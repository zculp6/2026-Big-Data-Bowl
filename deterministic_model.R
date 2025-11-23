library(tidyverse)
library(data.table)
library(foreach)
library(doParallel)

# Physics-Based Player Trajectory Model for NFL Tracking Data
# ==============================================================

# Helper Functions
# ----------------

angle_difference <- function(angle1, angle2) {
  # Calculate the smallest difference between two angles (in degrees)
  diff <- (angle2 - angle1) %% 360
  diff <- ifelse(diff > 180, diff - 360, diff)
  return(diff)
}

normalize_angle <- function(angle) {
  # Normalize angle to [0, 360)
  angle %% 360
}

euclidean_distance <- function(x1, y1, x2, y2) {
  # Calculate Euclidean distance
  sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

# Physics Model Class (using list structure in R)
# ------------------------------------------------

create_physics_model <- function(max_accel = 5.0, 
                                 max_decel = 8.0, 
                                 max_turn_rate = pi, 
                                 drag_coefficient = 0.1, 
                                 time_step = 0.1) {
  # Create a physics-based trajectory model
  
  model <- list(
    max_accel = max_accel,
    max_decel = max_decel,
    max_turn_rate = max_turn_rate,
    drag_coefficient = drag_coefficient,
    time_step = time_step,
    params = list(
      reaction_time = 0.2,      # Time before player reacts (seconds)
      anticipation = 0.5,       # How much player anticipates (0-1)
      aggressiveness = 0.8,     # How aggressively player accelerates (0-1)
      path_smoothness = 0.7     # How smooth the path is (0-1)
    )
  )
  
  class(model) <- "physics_trajectory_model"
  return(model)
}

# Single Trajectory Prediction
# -----------------------------

predict_single_trajectory <- function(model, x0, y0, s0, a0, dir0, o0,
                                      target_x, target_y, num_frames,
                                      player_position = "WR") {
  # Predict trajectory for a single player
  #
  # Parameters:
  # -----------
  # x0, y0 : initial position
  # s0 : initial speed (yards/s)
  # a0 : initial acceleration (yards/s^2)
  # dir0 : direction of motion (degrees)
  # o0 : player orientation (degrees)
  # target_x, target_y : ball landing position
  # num_frames : number of frames to predict
  # player_position : player position (affects behavior)
  
  # Initialize trajectory arrays
  x_pred <- numeric(num_frames)
  y_pred <- numeric(num_frames)
  
  # Current state
  x_curr <- x0
  y_curr <- y0
  s_curr <- s0
  dir_curr <- dir0 * pi / 180  # Convert to radians
  
  # Position-specific parameters
  max_speed <- ifelse(player_position %in% c("WR", "CB", "S"), 11.0, 9.0)
  
  for (frame in 1:num_frames) {
    # Calculate vector to target
    dx <- target_x - x_curr
    dy <- target_y - y_curr
    dist_to_target <- sqrt(dx^2 + dy^2)
    
    # Desired direction (toward ball)
    if (dist_to_target > 0.1) {
      desired_dir <- atan2(dy, dx)
    } else {
      desired_dir <- dir_curr
    }
    
    # Calculate angle difference (how much to turn)
    angle_diff <- atan2(sin(desired_dir - dir_curr), 
                        cos(desired_dir - dir_curr))
    
    # Apply turning rate constraint
    max_turn <- model$max_turn_rate * model$time_step
    turn <- sign(angle_diff) * min(abs(angle_diff), max_turn)
    
    # Smooth turning based on path_smoothness
    turn <- turn * model$params$path_smoothness
    dir_curr <- dir_curr + turn
    
    # Desired speed based on distance to target
    if (dist_to_target > 5) {
      desired_speed <- max_speed * model$params$aggressiveness
    } else {
      # Decelerate as approaching target
      desired_speed <- max(1.0, max_speed * (dist_to_target / 5))
    }
    
    # Calculate acceleration needed
    speed_diff <- desired_speed - s_curr
    
    if (speed_diff > 0) {
      accel <- min(speed_diff / model$time_step, 
                   model$max_accel * model$params$aggressiveness)
    } else {
      accel <- max(speed_diff / model$time_step, -model$max_decel)
    }
    
    # Update speed with acceleration and drag
    s_curr <- s_curr + accel * model$time_step
    s_curr <- s_curr * (1 - model$drag_coefficient * model$time_step)
    s_curr <- max(0, min(s_curr, max_speed))
    
    # Update position
    x_curr <- x_curr + s_curr * cos(dir_curr) * model$time_step
    y_curr <- y_curr + s_curr * sin(dir_curr) * model$time_step
    
    # Store prediction
    x_pred[frame] <- x_curr
    y_pred[frame] <- y_curr
  }
  
  return(data.frame(x = x_pred, y = y_pred))
}

# Batch Prediction Function
# --------------------------

predict_trajectories <- function(model, input_data) {
  # Predict trajectories for all players in input data
  #
  # Parameters:
  # -----------
  # input_data : data frame with last frame for each player before pass
  
  # Get last frame for each player (end of input data)
  last_frames <- input_data %>%
    group_by(game_id, play_id, nfl_id) %>%
    filter(frame_id == max(frame_id)) %>%
    ungroup()
  
  # Generate predictions
  predictions <- last_frames %>%
    rowwise() %>%
    mutate(
      trajectory = list(
        predict_single_trajectory(
          model = model,
          x0 = x, y0 = y, s0 = s, a0 = a,
          dir0 = dir, o0 = o,
          target_x = ball_land_x,
          target_y = ball_land_y,
          num_frames = num_frames_output,
          player_position = player_position
        )
      )
    ) %>%
    ungroup() %>%
    select(game_id, play_id, nfl_id, trajectory) %>%
    unnest(trajectory) %>%
    group_by(game_id, play_id, nfl_id) %>%
    mutate(frame_id = row_number()) %>%
    ungroup()
  
  return(predictions)
}

# RMSE Calculation
# ----------------

calculate_rmse <- function(predictions, ground_truth) {
  # Calculate RMSE between predictions and ground truth
  #
  # RMSE = sqrt(1/(2N) * sum((x_true - x_pred)^2 + (y_true - y_pred)^2))
  
  merged <- predictions %>%
    inner_join(ground_truth, 
               by = c("game_id", "play_id", "nfl_id", "frame_id"),
               suffix = c("_pred", "_true"))
  
  N <- nrow(merged)
  rmse <- sqrt(sum((merged$x_true - merged$x_pred)^2 + 
                     (merged$y_true - merged$y_pred)^2) / (2 * N))
  
  return(rmse)
}

# Parameter Optimization
# ----------------------

optimize_parameters <- function(input_data, output_data, 
                                param_grid = NULL,
                                n_folds = 5) {
  # Optimize model parameters using cross-validation
  
  if (is.null(param_grid)) {
    # Default parameter grid
    param_grid <- expand.grid(
      reaction_time = c(0.1, 0.2, 0.3),
      anticipation = c(0.3, 0.5, 0.7),
      aggressiveness = c(0.6, 0.8, 1.0),
      path_smoothness = c(0.5, 0.7, 0.9)
    )
  }
  
  # Get unique plays for cross-validation
  unique_plays <- input_data %>%
    distinct(game_id, play_id) %>%
    mutate(fold_id = sample(1:n_folds, n(), replace = TRUE))
  
  # Add fold IDs to data
  input_data <- input_data %>%
    left_join(unique_plays, by = c("game_id", "play_id"))
  
  output_data <- output_data %>%
    left_join(unique_plays, by = c("game_id", "play_id"))
  
  # Test each parameter combination
  results <- data.frame()
  
  cat("Testing", nrow(param_grid), "parameter combinations...\n")
  
  for (i in 1:nrow(param_grid)) {
    params <- param_grid[i, ]
    fold_rmse <- numeric(n_folds)
    
    for (fold in 1:n_folds) {
      # Split data
      train_input <- input_data %>% filter(fold_id != fold)
      val_input <- input_data %>% filter(fold_id == fold)
      val_output <- output_data %>% filter(fold_id == fold)
      
      # Create model with current parameters
      model <- create_physics_model()
      model$params <- as.list(params)
      
      # Predict on validation set
      predictions <- predict_trajectories(model, val_input)
      
      # Calculate RMSE
      fold_rmse[fold] <- calculate_rmse(predictions, val_output)
    }
    
    # Store results
    results <- rbind(results, cbind(params, 
                                    mean_rmse = mean(fold_rmse),
                                    sd_rmse = sd(fold_rmse)))
    
    if (i %% 10 == 0) {
      cat("Completed", i, "of", nrow(param_grid), "combinations\n")
    }
  }
  
  # Find best parameters
  best_idx <- which.min(results$mean_rmse)
  best_params <- results[best_idx, ]
  
  cat("\nBest parameters found:\n")
  print(best_params)
  
  return(list(
    best_params = best_params,
    all_results = results
  ))
}

# Main Training Function
# ----------------------

train_model <- function(input_files, output_files, 
                        optimize_params = TRUE,
                        n_folds = 5) {
  # Train the physics-based model
  
  cat("Loading data...\n")
  
  # Load all input files
  input_data <- map_dfr(input_files, fread)
  
  # Load all output files
  output_data <- map_dfr(output_files, fread)
  
  cat("Loaded", nrow(input_data), "input rows and", 
      nrow(output_data), "output rows\n")
  
  if (optimize_params) {
    cat("\nOptimizing parameters...\n")
    opt_results <- optimize_parameters(input_data, output_data, 
                                       n_folds = n_folds)
    
    # Create final model with best parameters
    model <- create_physics_model()
    model$params <- as.list(opt_results$best_params[1:4])
    
    return(list(
      model = model,
      optimization_results = opt_results
    ))
  } else {
    # Return model with default parameters
    model <- create_physics_model()
    return(list(model = model))
  }
}

# Prediction Function for Test Data
# ----------------------------------

make_predictions <- function(model, input_files, output_file = "predictions.csv") {
  # Make predictions on test data
  
  cat("Loading input data...\n")
  input_data <- map_dfr(input_files, fread)
  
  cat("Generating predictions...\n")
  predictions <- predict_trajectories(model, input_data)
  
  cat("Saving predictions to", output_file, "...\n")
  fwrite(predictions, output_file)
  
  cat("Done! Predicted", nrow(predictions), "positions\n")
  
  return(predictions)
}

# Example Usage
# =============

# # Train model with parameter optimization
input_files <- list.files(path = "train", 
                          pattern = "input_2023_w[0-9]{2}\\.csv",
                          full.names = TRUE)

output_files <- list.files(path = "train",
                           pattern = "output_2023_w[0-9]{2}\\.csv", 
                           full.names = TRUE)
# 
trained <- train_model(input_files, output_files, 
                       optimize_params = TRUE, 
                       n_folds = 5)

best_model <- trained$model
# 
# # Make predictions on new data

val_predictions <- make_predictions(best_model, "train/input_2023_w01.csv")
# 
# # Evaluate on validation set (if you have ground truth)
#val_output <- fread("output_validation.csv")
#val_predictions <- fread("predictions.csv")
rmse <- calculate_rmse(val_predictions, output_w01)
# cat("Validation RMSE:", rmse, "\n")