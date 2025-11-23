library(tidyverse)
library(data.table)
library(keras3)
library(reticulate)
library(xgboost)
library(ranger)
library(glmnet)
library(caret)
library(mgcv)

# ============================================================================
# MULTIPLE MODEL APPROACHES FOR NFL PLAYER TRAJECTORY PREDICTION
# ============================================================================

# Helper Functions
# ----------------

# Prepare features for modeling
prepare_features <- function(input_data, include_sequence = FALSE, seq_length = 5) {
  
  if (!include_sequence) {
    # Get last frame for each player
    features <- input_data %>%
      group_by(game_id, play_id, nfl_id) %>%
      arrange(frame_id) %>%
      slice_tail(n = 1) %>%
      ungroup() %>%
      mutate(
        # Distance to ball
        dist_to_ball = sqrt((ball_land_x - x)^2 + (ball_land_y - y)^2),
        
        # Angle to ball
        angle_to_ball = atan2(ball_land_y - y, ball_land_x - x) * 180 / pi,
        angle_diff = (angle_to_ball - dir) %% 360,
        angle_diff = ifelse(angle_diff > 180, angle_diff - 360, angle_diff),
        
        # Player characteristics
        player_age = as.numeric(Sys.Date() - as.Date(player_birth_date)) / 365.25,
        player_bmi = player_weight / (as.numeric(sub("(\\d+)-(\\d+)", "\\1", player_height)) * 12 + 
                                        as.numeric(sub("(\\d+)-(\\d+)", "\\2", player_height)))^2 * 703,
        
        # Position encoding
        is_receiver = as.numeric(player_position %in% c("WR", "TE")),
        is_defensive_back = as.numeric(player_position %in% c("CB", "S", "DB")),
        is_linebacker = as.numeric(player_position == "LB"),
        
        # Role encoding
        is_target = as.numeric(player_role == "Targeted Receiver"),
        is_passer = as.numeric(player_role == "Passer"),
        is_coverage = as.numeric(player_role == "Defensive Coverage"),
        
        # Direction features
        dir_rad = dir * pi / 180,
        o_rad = o * pi / 180,
        dir_x = cos(dir_rad),
        dir_y = sin(dir_rad),
        o_x = cos(o_rad),
        o_y = sin(o_rad)
      )
    
    return(features)
    
  } else {
    # Create sequences of last N frames for LSTM
    sequences <- input_data %>%
      group_by(game_id, play_id, nfl_id) %>%
      arrange(frame_id) %>%
      slice_tail(n = seq_length) %>%
      mutate(seq_idx = row_number()) %>%
      ungroup()
    
    return(sequences)
  }
}

calculate_rmse <- function(predictions, ground_truth) {
  # Calculate RMSE between predictions and ground truth
  merged <- predictions %>%
    inner_join(ground_truth, 
               by = c("game_id", "play_id", "nfl_id", "frame_id"),
               suffix = c("_pred", "_true"))
  
  N <- nrow(merged)
  rmse <- sqrt(sum((merged$x_true - merged$x_pred)^2 + 
                     (merged$y_true - merged$y_pred)^2) / (2 * N))
  
  return(rmse)
}
# ============================================================================
# MODEL 1: GAM (Generalized Additive Model) with Smoothing Splines
# ============================================================================

train_gam_model <- function(input_data, output_data) {
  cat("Training GAM Model...\n")
  
  # Prepare features
  features <- prepare_features(input_data)
  
  # Join with output to create training data for each frame
  train_data <- features %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    inner_join(output_data, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      # Target: change in position per frame
      delta_x = (x.y - x.x) / frame_id,
      delta_y = (y.y - y.x) / frame_id,
      # Normalize frame_id
      frame_norm = frame_id / num_frames_output
    )
  
  # Sample for faster training if dataset is very large
  if (nrow(train_data) > 100000) {
    train_data <- train_data %>% sample_n(100000)
    cat("Sampled 100k observations for GAM training\n")
  }
  
  # Train GAM models with smooth terms for continuous variables
  # Using thin plate regression splines (default) with k basis functions
  model_x <- gam(
    delta_x ~ 
      s(s, k = 10) +                    # Smooth term for speed
      s(a, k = 10) +                    # Smooth term for acceleration
      s(dist_to_ball, k = 10) +         # Smooth term for distance to ball
      s(angle_diff, k = 10) +           # Smooth term for angle difference
      s(player_age, k = 8) +            # Smooth term for age
      s(player_bmi, k = 8) +            # Smooth term for BMI
      s(frame_norm, k = 10) +           # Smooth term for frame position
      is_receiver + is_defensive_back + is_linebacker +
      is_target + is_passer + is_coverage +
      dir_x + dir_y + o_x + o_y,
    data = train_data,
    method = "REML"  # Restricted maximum likelihood for smoothing parameter selection
  )
  
  model_y <- gam(
    delta_y ~ 
      s(s, k = 10) +
      s(a, k = 10) +
      s(dist_to_ball, k = 10) +
      s(angle_diff, k = 10) +
      s(player_age, k = 8) +
      s(player_bmi, k = 8) +
      s(frame_norm, k = 10) +
      is_receiver + is_defensive_back + is_linebacker +
      is_target + is_passer + is_coverage +
      dir_x + dir_y + o_x + o_y,
    data = train_data,
    method = "REML"
  )
  
  cat("GAM Model trained. Deviance explained X:", 
      round(summary(model_x)$dev.expl * 100, 2), "%",
      "Y:", round(summary(model_y)$dev.expl * 100, 2), "%\n")
  
  return(list(model_x = model_x, model_y = model_y))
}

predict_gam_model <- function(models, input_data) {
  features <- prepare_features(input_data)
  
  predictions <- features %>%
    rowwise() %>%
    mutate(
      trajectories = list({
        X_new <- tibble(
          s = s, a = a, dist_to_ball = dist_to_ball, angle_diff = angle_diff,
          player_age = player_age, player_bmi = player_bmi,
          is_receiver = is_receiver, is_defensive_back = is_defensive_back,
          is_linebacker = is_linebacker,
          is_target = is_target, is_passer = is_passer, is_coverage = is_coverage,
          dir_x = dir_x, dir_y = dir_y, o_x = o_x, o_y = o_y,
          frame_norm = (1:num_frames_output) / num_frames_output
        )
        
        delta_x_pred <- predict(models$model_x, newdata = X_new)
        delta_y_pred <- predict(models$model_y, newdata = X_new)
        
        tibble(
          frame_id = 1:num_frames_output,
          x = x + cumsum(delta_x_pred),
          y = y + cumsum(delta_y_pred)
        )
      })
    ) %>%
    ungroup() %>%
    select(game_id, play_id, nfl_id, trajectories) %>%
    unnest(trajectories)
  
  return(predictions)
}

# ============================================================================
# MODEL 2: XGBoost (Gradient Boosted Trees)
# ============================================================================

train_xgboost_model <- function(input_data, output_data, nrounds = 100) {
  cat("Training XGBoost Model...\n")
  
  features <- prepare_features(input_data)
  
  train_data <- features %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    inner_join(output_data, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      delta_x = (x.y - x.x) / frame_id,
      delta_y = (y.y - y.x) / frame_id,
      frame_norm = frame_id / num_frames_output
    )
  
  X <- train_data %>%
    select(s, a, dist_to_ball, angle_diff, 
           player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           frame_norm) %>%
    as.matrix()
  
  # Train XGBoost models
  dtrain_x <- xgb.DMatrix(data = X, label = train_data$delta_x)
  dtrain_y <- xgb.DMatrix(data = X, label = train_data$delta_y)
  
  params <- list(
    objective = "reg:squarederror",
    eta = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  model_x <- xgb.train(params, dtrain_x, nrounds = nrounds, verbose = 0)
  model_y <- xgb.train(params, dtrain_y, nrounds = nrounds, verbose = 0)
  
  cat("XGBoost Model trained.\n")
  
  return(list(model_x = model_x, model_y = model_y, feature_names = colnames(X)))
}

predict_xgboost_model <- function(models, input_data) {
  # Handle if input_data is a data frame (already loaded)
  if (is.data.frame(input_data)) {
    data_to_use <- input_data
  } else if (is.character(input_data)) {
    # If it's a file path or vector of paths
    data_to_use <- map_dfr(input_data, fread)
  } else {
    stop("input_data must be a data frame or file path(s)")
  }
  
  features <- prepare_features(data_to_use)
  
  predictions <- features %>%
    rowwise() %>%
    mutate(
      trajectories = list({
        X_new <- matrix(c(
          rep(s, num_frames_output),
          rep(a, num_frames_output),
          rep(dist_to_ball, num_frames_output),
          rep(angle_diff, num_frames_output),
          rep(player_age, num_frames_output),
          rep(player_bmi, num_frames_output),
          rep(is_receiver, num_frames_output),
          rep(is_defensive_back, num_frames_output),
          rep(is_linebacker, num_frames_output),
          rep(is_target, num_frames_output),
          rep(is_passer, num_frames_output),
          rep(is_coverage, num_frames_output),
          rep(dir_x, num_frames_output),
          rep(dir_y, num_frames_output),
          rep(o_x, num_frames_output),
          rep(o_y, num_frames_output),
          (1:num_frames_output) / num_frames_output
        ), ncol = 17, byrow = FALSE)
        
        dtest <- xgb.DMatrix(data = X_new)
        
        delta_x_pred <- predict(models$model_x, dtest)
        delta_y_pred <- predict(models$model_y, dtest)
        
        tibble(
          frame_id = 1:num_frames_output,
          x = x + cumsum(delta_x_pred),
          y = y + cumsum(delta_y_pred)
        )
      })
    ) %>%
    ungroup() %>%
    select(game_id, play_id, nfl_id, trajectories) %>%
    unnest(trajectories)
  
  return(predictions)
}

# ============================================================================
# MODEL 3: Random Forest
# ============================================================================

train_random_forest_model <- function(input_data, output_data, num_trees = 100) {
  cat("Training Random Forest Model...\n")
  
  features <- prepare_features(input_data)
  
  train_data <- features %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    inner_join(output_data, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      delta_x = (x.y - x.x) / frame_id,
      delta_y = (y.y - y.x) / frame_id,
      frame_norm = frame_id / num_frames_output
    )
  
  # Sample for faster training if dataset is large
  if (nrow(train_data) > 50000) {
    train_data <- train_data %>% sample_n(50000)
  }
  
  formula_features <- ~ s + a + dist_to_ball + angle_diff + 
    player_age + player_bmi +
    is_receiver + is_defensive_back + is_linebacker +
    is_target + is_passer + is_coverage +
    dir_x + dir_y + o_x + o_y + frame_norm
  
  model_x <- ranger(
    delta_x ~ s + a + dist_to_ball + angle_diff + 
      player_age + player_bmi +
      is_receiver + is_defensive_back + is_linebacker +
      is_target + is_passer + is_coverage +
      dir_x + dir_y + o_x + o_y + frame_norm,
    data = train_data,
    num.trees = num_trees,
    importance = 'impurity'
  )
  
  model_y <- ranger(
    delta_y ~ s + a + dist_to_ball + angle_diff + 
      player_age + player_bmi +
      is_receiver + is_defensive_back + is_linebacker +
      is_target + is_passer + is_coverage +
      dir_x + dir_y + o_x + o_y + frame_norm,
    data = train_data,
    num.trees = num_trees,
    importance = 'impurity'
  )
  
  cat("Random Forest Model trained. OOB R-squared X:", model_x$r.squared,
      "Y:", model_y$r.squared, "\n")
  
  return(list(model_x = model_x, model_y = model_y))
}

predict_random_forest_model <- function(models, input_data) {
  features <- prepare_features(input_data)
  
  predictions <- features %>%
    rowwise() %>%
    mutate(
      trajectories = list({
        X_new <- tibble(
          s = s, a = a, dist_to_ball = dist_to_ball, angle_diff = angle_diff,
          player_age = player_age, player_bmi = player_bmi,
          is_receiver = is_receiver, is_defensive_back = is_defensive_back,
          is_linebacker = is_linebacker,
          is_target = is_target, is_passer = is_passer, is_coverage = is_coverage,
          dir_x = dir_x, dir_y = dir_y, o_x = o_x, o_y = o_y,
          frame_norm = (1:num_frames_output) / num_frames_output
        )
        
        delta_x_pred <- predict(models$model_x, X_new)$predictions
        delta_y_pred <- predict(models$model_y, X_new)$predictions
        
        tibble(
          frame_id = 1:num_frames_output,
          x = x + cumsum(delta_x_pred),
          y = y + cumsum(delta_y_pred)
        )
      })
    ) %>%
    ungroup() %>%
    select(game_id, play_id, nfl_id, trajectories) %>%
    unnest(trajectories)
  
  return(predictions)
}

# ============================================================================
# MODEL 4: Elastic Net (Ridge + Lasso Regression)
# ============================================================================

train_elasticnet_model <- function(input_data, output_data, alpha = 0.5) {
  cat("Training Elastic Net Model...\n")
  
  features <- prepare_features(input_data)
  
  train_data <- features %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    inner_join(output_data, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      delta_x = (x.y - x.x) / frame_id,
      delta_y = (y.y - y.x) / frame_id,
      frame_norm = frame_id / num_frames_output
    )
  
  X <- train_data %>%
    select(s, a, dist_to_ball, angle_diff, 
           player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           frame_norm) %>%
    as.matrix()
  
  # Cross-validated elastic net
  model_x <- cv.glmnet(X, train_data$delta_x, alpha = alpha, nfolds = 5)
  model_y <- cv.glmnet(X, train_data$delta_y, alpha = alpha, nfolds = 5)
  
  cat("Elastic Net Model trained. Lambda X:", model_x$lambda.min,
      "Y:", model_y$lambda.min, "\n")
  
  return(list(model_x = model_x, model_y = model_y))
}

predict_elasticnet_model <- function(models, input_data) {
  features <- prepare_features(input_data)
  
  predictions <- features %>%
    rowwise() %>%
    mutate(
      trajectories = list({
        X_new <- matrix(c(
          rep(s, num_frames_output),
          rep(a, num_frames_output),
          rep(dist_to_ball, num_frames_output),
          rep(angle_diff, num_frames_output),
          rep(player_age, num_frames_output),
          rep(player_bmi, num_frames_output),
          rep(is_receiver, num_frames_output),
          rep(is_defensive_back, num_frames_output),
          rep(is_linebacker, num_frames_output),
          rep(is_target, num_frames_output),
          rep(is_passer, num_frames_output),
          rep(is_coverage, num_frames_output),
          rep(dir_x, num_frames_output),
          rep(dir_y, num_frames_output),
          rep(o_x, num_frames_output),
          rep(o_y, num_frames_output),
          (1:num_frames_output) / num_frames_output
        ), ncol = 17, byrow = FALSE)
        
        delta_x_pred <- predict(models$model_x, newx = X_new, s = "lambda.min")
        delta_y_pred <- predict(models$model_y, newx = X_new, s = "lambda.min")
        
        tibble(
          frame_id = 1:num_frames_output,
          x = x + cumsum(delta_x_pred[, 1]),
          y = y + cumsum(delta_y_pred[, 1])
        )
      })
    ) %>%
    ungroup() %>%
    select(game_id, play_id, nfl_id, trajectories) %>%
    unnest(trajectories)
  
  return(predictions)
}

# ============================================================================
# MODEL 5: Physics Deterministic Model (Optimal Path to Ball)
# ============================================================================

train_physics_model <- function(train_input, train_output) {
  opt <- optimize_parameters(train_input, train_output, n_folds = 3)
  model <- create_physics_model()
  model$params <- as.list(opt$best_params[1:4])
  return(model)
}
predict_physics_model <- function(model, new_input) {
  predict_trajectories(model, new_input)
}

# ============================================================================
# MODEL 6: Transformers
# ============================================================================

# ================================================================
# Custom Transformer Encoder Layer (R keras)
layer_transformer_encoder <- function(num_heads, key_dim, ff_dim, dropout = 0.1) {
  function(x) {
    
    embedding_dim <- x$shape[[3]]
    
    attn_output <- layer_multi_head_attention(
      num_heads = num_heads,
      key_dim = key_dim,
      dropout = dropout
    )(x, x)
    
    attn_output <- layer_dropout(rate = dropout)(attn_output)
    out1 <- layer_layer_normalization(epsilon = 1e-6)(x + attn_output)
    
    ffn <- out1 %>%
      layer_dense(ff_dim, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_dense(units = embedding_dim)
    
    out2 <- layer_layer_normalization(epsilon = 1e-6)(out1 + ffn)
    out2
  }
}
train_transformer_model <- function(input_data, output_data,
                                    num_heads = 8, 
                                    key_dim = 64, 
                                    ff_dim = 128,
                                    epochs = 20, 
                                    batch_size = 64,
                                    sequence_length = 10,
                                    dropout_rate = 0.1) { 
  
  features <- prepare_features(input_data)
  
  train_data <- features %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    inner_join(output_data, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      delta_x = (x.y - x.x) / frame_id,
      delta_y = (y.y - y.x) / frame_id,
      frame_id = as.numeric(frame_id),
      frame_norm = frame_id / num_frames_output
    )
  
  # Group by player and create sequences
  train_sequences <- train_data %>%
    arrange(game_id, play_id, nfl_id, frame_id) %>%
    group_by(game_id, play_id, nfl_id) %>%
    mutate(
      seq_group = ceiling(row_number() / sequence_length)
    ) %>%
    filter(n() >= sequence_length) %>%  # Only keep complete sequences
    ungroup()
  
  # Create feature matrix for each sequence
  feature_cols <- c("s", "a", "dist_to_ball", "angle_diff", 
                    "player_age", "player_bmi",
                    "is_receiver", "is_defensive_back", "is_linebacker",
                    "is_target", "is_passer", "is_coverage",
                    "dir_x", "dir_y", "o_x", "o_y", "frame_norm")
  
  target_cols <- c("delta_x", "delta_y")
  
  num_features <- length(feature_cols)
  
  # Build sequences as 3D arrays
  sequences_list <- train_sequences %>%
    group_by(game_id, play_id, nfl_id, seq_group) %>%
    group_split()
  
  # Filter to only complete sequences
  sequences_list <- sequences_list[sapply(sequences_list, nrow) == sequence_length]
  
  num_sequences <- length(sequences_list)
  
  # Preallocate arrays
  X <- array(0, dim = c(num_sequences, sequence_length, num_features))
  Y <- array(0, dim = c(num_sequences, sequence_length, 2))
  
  # Fill arrays
  for (i in seq_along(sequences_list)) {
    seq_data <- sequences_list[[i]]
    X[i, , ] <- as.matrix(seq_data[, feature_cols])
    Y[i, , ] <- as.matrix(seq_data[, target_cols])
  }
  
  cat("Created", num_sequences, "sequences of length", sequence_length, "\n")
  cat("Input shape:", dim(X), "\n")
  
  # Model - NOW with proper sequence dimension
  inputs <- layer_input(shape = c(sequence_length, num_features))
  
  # Transformer encoder processes the sequence
  # First create the transformer encoder function
  transformer_encoder <- layer_transformer_encoder(
    num_heads = num_heads,
    key_dim = key_dim,
    ff_dim = ff_dim,
    dropout = dropout_rate
  )
  
  # Then apply it to inputs
  x <- transformer_encoder(inputs)
  
  # Predict for all timesteps in sequence
  outputs <- x %>%
    layer_dense(64, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(2)  # Output shape: (batch, sequence_length, 2)
  
  model <- keras_model(inputs, outputs)
  
  custom_optimizer <- optimizer_sgd(learning_rate = 0.01, momentum = 0.9)
  model |> compile(
    optimizer = custom_optimizer,
    loss = "mse"
  )
  
  model |> fit(
    X, Y,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.1,
    verbose = 1
  )
  
  return(model)
}

predict_transformer_model <- function(model, input_data, sequence_length = 10) {
  
  # Prepare input data
  if (is.data.frame(input_data)) {
    data_to_use <- input_data
  } else if (is.character(input_data)) {
    data_to_use <- map_dfr(input_data, fread)
  } else {
    stop("input_data must be a data frame or file paths")
  }
  
  features <- prepare_features(data_to_use)
  
  # Expand for all frames
  expanded <- features %>%
    rowwise() %>%
    mutate(frame_id_vec = list(1:num_frames_output)) %>%
    unnest(frame_id_vec) %>%
    ungroup() %>%
    mutate(frame_norm = frame_id_vec / num_frames_output) %>%
    arrange(game_id, play_id, nfl_id, frame_id_vec) %>%
    group_by(game_id, play_id, nfl_id) %>%
    mutate(seq_group = ceiling(row_number() / sequence_length)) %>%
    ungroup()
  
  feature_cols <- c("s", "a", "dist_to_ball", "angle_diff", 
                    "player_age", "player_bmi",
                    "is_receiver", "is_defensive_back", "is_linebacker",
                    "is_target", "is_passer", "is_coverage",
                    "dir_x", "dir_y", "o_x", "o_y", "frame_norm")
  
  # Build sequences
  sequences_list <- expanded %>%
    group_by(game_id, play_id, nfl_id, seq_group) %>%
    filter(n() == sequence_length) %>%  # Only complete sequences
    group_split()
  
  if (length(sequences_list) == 0) {
    stop("No complete sequences found for prediction")
  }
  
  num_sequences <- length(sequences_list)
  num_features <- length(feature_cols)
  
  # Preallocate
  X_seq <- array(0, dim = c(num_sequences, sequence_length, num_features))
  
  for (i in seq_along(sequences_list)) {
    seq_data <- sequences_list[[i]]
    X_seq[i, , ] <- as.matrix(seq_data %>% 
                                mutate(across(all_of(feature_cols), as.numeric)) %>%
                                select(all_of(feature_cols)))
  }
  
  # Predict all sequences at once with batching
  preds <- predict(model, X_seq, verbose = 0, batch_size = 1000)
  
  # Flatten predictions back to dataframe
  result <- tibble()
  for (i in seq_along(sequences_list)) {
    seq_info <- sequences_list[[i]] %>%
      select(game_id, play_id, nfl_id, frame_id = frame_id_vec, x, y)
    
    seq_info$delta_x <- preds[i, , 1]
    seq_info$delta_y <- preds[i, , 2]
    
    result <- bind_rows(result, seq_info)
  }
  
  # Compute cumulative trajectories
  trajectories <- result %>%
    group_by(game_id, play_id, nfl_id) %>%
    arrange(frame_id) %>%
    mutate(
      x = x[1] + cumsum(delta_x),  # Start from initial position
      y = y[1] + cumsum(delta_y)
    ) %>%
    select(game_id, play_id, nfl_id, frame_id, x, y) %>%
    ungroup()
  
  return(trajectories)
}

predict_transformer_autoregressive <- function(model, input_file, sequence_length = 10) {
  
  input_data <- read_csv(input_file, show_col_types = FALSE)
  features <- prepare_features(input_data)
  
  players_to_predict <- features %>%
    filter(player_to_predict == TRUE) %>%
    group_by(game_id, play_id, nfl_id) %>%
    slice_tail(n = 1) %>%
    select(game_id, play_id, nfl_id, x, y, s, a, dir, o,
           dist_to_ball, angle_diff, player_age, player_bmi,
           is_receiver, is_defensive_back, is_linebacker,
           is_target, is_passer, is_coverage,
           dir_x, dir_y, o_x, o_y,
           ball_land_x, ball_land_y, num_frames_output) %>%
    ungroup()
  
  feature_cols <- c("s", "a", "dist_to_ball", "angle_diff", 
                    "player_age", "player_bmi",
                    "is_receiver", "is_defensive_back", "is_linebacker",
                    "is_target", "is_passer", "is_coverage",
                    "dir_x", "dir_y", "o_x", "o_y", "frame_norm")
  
  num_features <- length(feature_cols)
  all_predictions <- list()
  
  for (i in 1:nrow(players_to_predict)) {
    player_row <- players_to_predict[i, ]
    num_output_frames <- player_row$num_frames_output
    
    # Start with last known position
    current_x <- player_row$x
    current_y <- player_row$y
    
    pred_frames <- data.frame(
      game_id = integer(),
      play_id = integer(),
      nfl_id = integer(),
      frame_id = integer(),
      x = numeric(),
      y = numeric()
    )
    
    # Autoregressive: use previous predictions to inform next ones
    for (frame in 1:num_output_frames) {
      frame_norm <- frame / num_output_frames
      
      # Update distance to ball based on current position
      dist_to_ball <- sqrt((current_x - player_row$ball_land_x)^2 + 
                             (current_y - player_row$ball_land_y)^2)
      
      input_features <- data.frame(
        s = player_row$s,
        a = player_row$a,
        dist_to_ball = dist_to_ball,  # Updated
        angle_diff = player_row$angle_diff,
        player_age = player_row$player_age,
        player_bmi = player_row$player_bmi,
        is_receiver = player_row$is_receiver,
        is_defensive_back = player_row$is_defensive_back,
        is_linebacker = player_row$is_linebacker,
        is_target = player_row$is_target,
        is_passer = player_row$is_passer,
        is_coverage = player_row$is_coverage,
        dir_x = player_row$dir_x,
        dir_y = player_row$dir_y,
        o_x = player_row$o_x,
        o_y = player_row$o_y,
        frame_norm = frame_norm
      )
      
      X_pred <- array(0, dim = c(1, sequence_length, num_features))
      for (t in 1:sequence_length) {
        X_pred[1, t, ] <- as.matrix(input_features[, feature_cols])
      }
      
      delta_pred <- predict(model, X_pred, verbose = 0)
      
      # Use last timestep prediction
      delta_x <- delta_pred[1, sequence_length, 1]
      delta_y <- delta_pred[1, sequence_length, 2]
      
      # Update position (single step)
      current_x <- current_x + delta_x
      current_y <- current_y + delta_y
      
      pred_frames <- bind_rows(pred_frames, data.frame(
        game_id = player_row$game_id,
        play_id = player_row$play_id,
        nfl_id = player_row$nfl_id,
        frame_id = frame,
        x = current_x,
        y = current_y
      ))
    }
    
    all_predictions[[i]] <- pred_frames
    
    if (i %% 10 == 0) {
      cat("Completed", i, "of", nrow(players_to_predict), "players\n")
    }
  }
  
  predictions_df <- bind_rows(all_predictions)
  
  cat("\nGenerated", nrow(predictions_df), "predictions (autoregressive)\n")
  return(predictions_df)
}


# ============================================================================
# CROSS-VALIDATION FRAMEWORK
# ============================================================================

cross_validate_models <- function(input_data, output_data, n_folds = 5) {
  cat("Running", n_folds, "-fold cross-validation across all models...\n")
  
  # Create folds
  unique_plays <- input_data %>%
    distinct(game_id, play_id) %>%
    mutate(fold = sample(1:n_folds, n(), replace = TRUE))
  
  input_data <- input_data %>%
    left_join(unique_plays, by = c("game_id", "play_id"))
  
  output_data <- output_data %>%
    left_join(unique_plays, by = c("game_id", "play_id"))
  
  results <- tibble()
  
  for (fold in 1:n_folds) {
    cat("\n=== Fold", fold, "of", n_folds, "===\n")
    
    train_input <- input_data %>% filter(fold != !!fold)
    train_output <- output_data %>% filter(fold != !!fold)
    val_input <- input_data %>% filter(fold == !!fold)
    val_output <- output_data %>% filter(fold == !!fold)
    
    # Train and evaluate each model
    models_to_test <- list(
      "GAM" = list(train = train_gam_model, predict = predict_gam_model),
      "XGBoost" = list(train = train_xgboost_model, predict = predict_xgboost_model),
      "RandomForest" = list(train = train_random_forest_model, predict = predict_random_forest_model),
      "ElasticNet" = list(train = train_elasticnet_model, predict = predict_elasticnet_model),
      "PhysicsModel" = list(train = train_physics_model, predict = predict_physics_model),
      "Transformer" = list(train = train_transformer_model, predict = predict_transformer_model)
    )
    
    for (model_name in names(models_to_test)) {
      tryCatch({
        model <- models_to_test[[model_name]]$train(train_input, output_data %>% filter(fold != !!fold))
        predictions <- models_to_test[[model_name]]$predict(model, val_input)
        rmse <- calculate_rmse(predictions, val_output)
        
        results <- bind_rows(results, tibble(
          fold = fold,
          model = model_name,
          rmse = rmse
        ))
        
        cat(model_name, "RMSE:", round(rmse, 4), "\n")
      }, error = function(e) {
        cat(model_name, "failed:", e$message, "\n")
      })
    }
  }
  
  # Summary statistics
  summary_stats <- results %>%
    group_by(model) %>%
    summarise(
      mean_rmse = mean(rmse),
      sd_rmse = sd(rmse),
      min_rmse = min(rmse),
      max_rmse = max(rmse)
    ) %>%
    arrange(mean_rmse)
  
  cat("\n=== CROSS-VALIDATION SUMMARY ===\n")
  print(summary_stats)
  
  return(list(fold_results = results, summary = summary_stats))
}

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# # Load data
input_files <- list.files("train", pattern = "input_2023_w[0-9]{2}\\.csv", full.names = TRUE)
output_files <- list.files("train", pattern = "output_2023_w[0-9]{2}\\.csv", full.names = TRUE)
# 
input_data <- map_dfr(input_files, fread)
output_data <- map_dfr(output_files, fread)
# 
# Compare all models with cross-validation
set.seed(123)
cv_results <- cross_validate_models(input_data, output_data, n_folds = 5)
 
deterministic_results <- cross_validate_models(input_data, output_data, n_folds = 5)
