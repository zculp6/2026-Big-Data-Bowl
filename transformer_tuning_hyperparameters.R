library(tidyverse)
library(keras3)
library(reticulate)

# Define hyperparameter grid
create_hyperparameter_grid <- function() {
  expand.grid(
    num_heads = c(4, 8),
    key_dim = c(32, 64),
    ff_dim = c(64, 128, 256),
    sequence_length = c(10),
    batch_size = c(64, 128),
    dropout = c(0.1, 0.2, 0.3, 0.4),
    learning_rate = c(0.0001, 0.001, 0.01),
    stringsAsFactors = FALSE
  )
}

# Random search (more efficient than grid search)
random_hyperparameter_search <- function(n_trials = 100) {
  tibble(
    num_heads = sample(c(2, 4, 8), n_trials, replace = TRUE),
    key_dim = sample(c(16, 32, 64, 128), n_trials, replace = TRUE),
    ff_dim = sample(c(64, 128, 256, 512), n_trials, replace = TRUE),
    sequence_length = sample(c(5, 10, 15, 20), n_trials, replace = TRUE),
    batch_size = sample(c(32, 64, 128), n_trials, replace = TRUE),
    dropout = runif(n_trials, 0.1, 0.4),
    learning_rate = 10^runif(n_trials, -4, -2),  # Log-uniform between 0.0001 and 0.01
    dense_units = sample(c(32, 64, 128), n_trials, replace = TRUE)
  )
}

# Modified training function with more hyperparameters
train_transformer_model_tunable <- function(input_data, output_data,
                                            num_heads = 4, 
                                            key_dim = 32, 
                                            ff_dim = 128,
                                            sequence_length = 10,
                                            epochs = 9, 
                                            batch_size = 64,
                                            dropout = 0.1,
                                            learning_rate = 0.001,
                                            dense_units = 64) {
  
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
  
  # Create sequences for each player/play
  train_sequences <- train_data %>%
    arrange(game_id, play_id, nfl_id, frame_id) %>%
    group_by(game_id, play_id, nfl_id) %>%
    mutate(seq_group = ceiling(row_number() / sequence_length)) %>%
    filter(n() >= sequence_length) %>%
    ungroup()
  
  feature_cols <- c("s", "a", "dist_to_ball", "angle_diff", 
                    "player_age", "player_bmi",
                    "is_receiver", "is_defensive_back", "is_linebacker",
                    "is_target", "is_passer", "is_coverage",
                    "dir_x", "dir_y", "o_x", "o_y", "frame_norm")
  target_cols <- c("delta_x", "delta_y")
  num_features <- length(feature_cols)
  
  # Build array
  sequences_list <- train_sequences %>%
    group_by(game_id, play_id, nfl_id, seq_group) %>%
    group_split()
  sequences_list <- sequences_list[sapply(sequences_list, nrow) == sequence_length]
  
  num_sequences <- length(sequences_list)
  X <- array(0, dim = c(num_sequences, sequence_length, num_features))
  Y <- array(0, dim = c(num_sequences, sequence_length, 2))
  
  for (i in seq_along(sequences_list)) {
    seq_data <- sequences_list[[i]]
    X[i, , ] <- as.matrix(seq_data[, feature_cols])
    Y[i, , ] <- as.matrix(seq_data[, target_cols])
  }
  
  # Build transformer model
  inputs <- layer_input(shape = c(sequence_length, num_features))
  
  x <- layer_transformer_encoder(
    num_heads = num_heads,
    key_dim = key_dim,
    ff_dim = ff_dim,
    dropout = dropout
  )(inputs)
  
  outputs <- x %>%
    layer_dense(dense_units, activation = "relu") %>%
    layer_dropout(dropout) %>%
    layer_dense(2)
  
  model <- keras_model(inputs, outputs)
  optimizer <- optimizer_adam(learning_rate = learning_rate)
  
  model |> compile(
    optimizer = optimizer,
    loss = "mse"
  )
  
  history <- model |> fit(
    X, Y,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = 0,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = 3, restore_best_weights = TRUE)
    )
  )
  
  return(list(model = model, history = history))
}


# Main tuning function
tune_transformer_hyperparameters <- function(input_data, output_data, 
                                             method = "random", 
                                             n_trials = 100,
                                             epochs = 20) {
  
  cat("Starting hyperparameter tuning with", method, "search...\n")
  
  # Generate random hyperparameters
  hp_grid <- random_hyperparameter_search(n_trials)
  cat("Testing", nrow(hp_grid), "hyperparameter combinations\n\n")
  
  results <- tibble()
  
  # Split train/val
  unique_plays <- input_data %>%
    distinct(game_id, play_id) %>%
    mutate(fold = sample(c("train", "val"), n(), replace = TRUE, prob = c(0.8, 0.2)))
  
  train_input <- input_data %>% left_join(unique_plays, by = c("game_id", "play_id")) %>% filter(fold == "train") %>% select(-fold)
  val_input   <- input_data %>% left_join(unique_plays, by = c("game_id", "play_id")) %>% filter(fold == "val")   %>% select(-fold)
  train_output <- output_data %>% left_join(unique_plays, by = c("game_id", "play_id")) %>% filter(fold == "train") %>% select(-fold)
  val_output   <- output_data %>% left_join(unique_plays, by = c("game_id", "play_id")) %>% filter(fold == "val")   %>% select(-fold)
  
  # Loop over hyperparameter trials
  for (i in 1:nrow(hp_grid)) {
    cat("Trial", i, "of", nrow(hp_grid), "\n")
    hp <- hp_grid[i, ]
    
    tryCatch({
      start_time <- Sys.time()
      
      result <- train_transformer_model_tunable(
        train_input, train_output,
        num_heads = hp$num_heads,
        key_dim = hp$key_dim,
        ff_dim = hp$ff_dim,
        sequence_length = hp$sequence_length,
        batch_size = hp$batch_size,
        dropout = hp$dropout,
        learning_rate = hp$learning_rate,
        dense_units = hp$dense_units,
        epochs = epochs
      )
      
      model <- result$model
      history <- result$history
      
      val_loss <- min(history$metrics$val_loss, na.rm = TRUE)
      train_loss <- history$metrics$loss[which.min(history$metrics$val_loss)]
      
      # Compute validation RMSE using your vectorized prediction function
      predictions <- predict_transformer_model(model, val_input, sequence_length = hp$sequence_length)
      val_rmse <- calculate_rmse(predictions, val_output)
      
      training_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      results <- bind_rows(results,
                           tibble(
                             trial = i,
                             num_heads = hp$num_heads,
                             key_dim = hp$key_dim,
                             ff_dim = hp$ff_dim,
                             sequence_length = hp$sequence_length,
                             batch_size = hp$batch_size,
                             dropout = hp$dropout,
                             learning_rate = hp$learning_rate,
                             dense_units = hp$dense_units,
                             val_loss = val_loss,
                             train_loss = train_loss,
                             val_rmse = val_rmse,
                             training_time = training_time
                           ))
      
      cat("  Val RMSE:", round(val_rmse, 4),
          "| Val Loss:", round(val_loss, 4),
          "| Time:", round(training_time, 1), "s\n\n")
      
      keras::k_clear_session()
      gc()
      
    }, error = function(e) {
      cat("  Trial failed:", e$message, "\n\n")
    })
  }
  
  results <- results %>% arrange(val_rmse)
  
  cat("\n=== TOP 5 HYPERPARAMETER COMBINATIONS ===\n")
  print(results %>% head(5) %>% select(-trial))
  
  return(results)
}


# Usage example:
results <- tune_transformer_hyperparameters(
  input_data, output_data, 
  method = "random", 
  n_trials = 50,
  epochs = 100
)

# Train final model with best hyperparameters
best_hp <- results[1, ]
final_model <- train_transformer_model(
  input_data, output_data,
  num_heads = 4, 
  key_dim = 16, 
  ff_dim = 512,
  epochs = 100, 
  batch_size = 32,
  sequence_length = 5,
  dropout_rate = 0.17
)

final_prediction <- predict_transformer_autoregressive(final_model, input_files, sequence_length = 5)
calculate_rmse(final_prediction, output_data)
