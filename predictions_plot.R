# packages ----------------------------------------------------------------
library(tidyverse)
library(gganimate)
library(gifski)
library(png)
library(sportyR)
library(ggstar)
library(data.table)

# creating the football field ---------------------------------------------
field_params <- list(
  field_apron = "springgreen3",
  field_border = "springgreen3",
  offensive_endzone = "springgreen3",
  defensive_endzone = "springgreen3",
  offensive_half = "springgreen3",
  defensive_half = "springgreen3"
)
field_background <- geom_football(
  league = "nfl",
  display_range = "in bounds only",
  x_trans = 60,
  y_trans = 26.6667,
  color_updates = field_params
)

# Function to combine input, predictions, and actual results ---------------
combine_all_data <- function(input_file, output_file, predictions_df) {
  # Read input and output data
  tracking_input <- read_csv(input_file)
  tracking_output <- read_csv(output_file)
  
  # Get the maximum frame_id from input for each player
  max_input_frames <- tracking_input %>%
    group_by(game_id, play_id, nfl_id) %>%
    summarise(max_input_frame = max(frame_id), .groups = 'drop')
  
  # Add phase column to input (before pass)
  tracking_input <- tracking_input %>%
    mutate(
      phase = "INPUT",
      data_type = "actual"
    )
  
  # Prepare actual output with adjusted frame_ids
  tracking_output_adjusted <- tracking_output %>%
    left_join(max_input_frames, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      frame_id = frame_id + max_input_frame,
      phase = "OUTPUT",
      data_type = "actual"
    )
  
  # Get player metadata from last frame of input (including play_direction)
  player_metadata <- tracking_input %>%
    group_by(game_id, play_id, nfl_id) %>%
    slice_tail(n = 1) %>%
    select(game_id, play_id, nfl_id, player_name, player_position, 
           player_side, player_role, player_to_predict, ball_land_x, ball_land_y,
           absolute_yardline_number, play_direction)
  
  # Add metadata to actual output
  tracking_output_adjusted <- tracking_output_adjusted %>%
    select(-any_of(c("player_name", "player_position", "player_side", 
                     "player_role", "player_to_predict", "ball_land_x", 
                     "ball_land_y", "absolute_yardline_number", "play_direction"))) %>%
    left_join(player_metadata, by = c("game_id", "play_id", "nfl_id"))
  
  # Prepare predictions with adjusted frame_ids
  predictions_adjusted <- predictions_df %>%
    left_join(max_input_frames, by = c("game_id", "play_id", "nfl_id")) %>%
    mutate(
      frame_id = frame_id + max_input_frame,
      phase = "OUTPUT",
      data_type = "predicted"
    ) %>%
    left_join(player_metadata, by = c("game_id", "play_id", "nfl_id"))
  
  # Combine all data
  combined <- bind_rows(
    tracking_input %>% select(game_id, play_id, nfl_id, frame_id, x, y, 
                              player_name, player_position, player_side, 
                              player_role, player_to_predict, ball_land_x, 
                              ball_land_y, absolute_yardline_number, 
                              play_direction, phase, data_type),
    tracking_output_adjusted %>% select(game_id, play_id, nfl_id, frame_id, x, y,
                                        player_name, player_position, player_side,
                                        player_role, player_to_predict, ball_land_x,
                                        ball_land_y, absolute_yardline_number,
                                        play_direction, phase, data_type),
    predictions_adjusted %>% select(game_id, play_id, nfl_id, frame_id, x, y,
                                    player_name, player_position, player_side,
                                    player_role, player_to_predict, ball_land_x,
                                    ball_land_y, absolute_yardline_number,
                                    play_direction, phase, data_type)
  )
  
  return(combined)
}

# Function to standardize field ------------------------------------------
standardize_field <- function(tracking_data) {
  tracking_data %>%
    mutate(
      x = ifelse(play_direction == "left", 120 - x, x),
      y = ifelse(play_direction == "left", 160/3 - y, y),
      ball_land_x = ifelse(play_direction == "left", 120 - ball_land_x, ball_land_x),
      ball_land_y = ifelse(play_direction == "left", 160/3 - ball_land_y, ball_land_y)
    )
}

# Function to animate with predictions and actual results -----------------
animate_prediction_vs_actual <- function(tracking_data, play_index = 1, 
                                         save_path = "prediction_vs_actual.gif") {
  
  # Standardize field
  tracking_data <- standardize_field(tracking_data)
  
  # Pick a single play
  unique_plays <- tracking_data %>%
    distinct(game_id, play_id)
  
  one_play <- tracking_data %>%
    filter(game_id == unique_plays$game_id[play_index],
           play_id == unique_plays$play_id[play_index])
  
  # Get the starting position based on absolute_yardline_number
  start_yard <- unique(one_play$absolute_yardline_number)[1]
  
  # Adjust coordinates to center the play at the starting yard line
  x_offset <- start_yard - mean(one_play$x[one_play$frame_id == min(one_play$frame_id)], na.rm = TRUE)
  one_play <- one_play %>%
    mutate(x = x + x_offset)
  
  # Get ball landing coordinates and adjust them the same way
  ball_land_data <- one_play %>%
    filter(!is.na(ball_land_x) & !is.na(ball_land_y)) %>%
    distinct(ball_land_x, ball_land_y) %>%
    mutate(
      ball_land_x_adj = ball_land_x + x_offset,
      ball_land_y_adj = ball_land_y
    )
  
  # Separate data by type and player
  # Normal players (not predicted) - only show actual
  players_normal <- one_play %>% 
    filter(!player_to_predict, data_type == "actual")
  
  # Predicted players - actual data
  players_predict_actual <- one_play %>% 
    filter(player_to_predict, data_type == "actual")
  
  # Predicted players - predictions
  players_predict_predicted <- one_play %>% 
    filter(player_to_predict, data_type == "predicted")
  
  # Calculate the frame where OUTPUT phase starts
  output_start_frame <- min(one_play$frame_id[one_play$phase == "OUTPUT"], na.rm = TRUE)
  
  # Calculate RMSE for display
  rmse_data <- players_predict_actual %>%
    filter(phase == "OUTPUT") %>%
    inner_join(
      players_predict_predicted %>% filter(phase == "OUTPUT"),
      by = c("game_id", "play_id", "nfl_id", "frame_id"),
      suffix = c("_actual", "_pred")
    ) %>%
    summarise(
      rmse = sqrt(mean((x_actual - x_pred)^2 + (y_actual - y_pred)^2))
    ) %>%
    pull(rmse)
  
  # Create a data frame for dynamic phase labels
  phase_labels <- one_play %>%
    distinct(frame_id, phase) %>%
    mutate(phase_label = ifelse(phase == "INPUT", "INPUT PHASE", "OUTPUT PHASE"))
  
  # Plot setup with field background
  p <- field_background +
    # Normal players (circles)
    geom_point(data = players_normal, 
               aes(x = x, y = y, color = player_side), 
               size = 8) +
    geom_text(data = players_normal,
              aes(x = x, y = y, label = player_position, color = player_side),
              vjust = -1.5, size = 5) +
    
    # Predicted players - ACTUAL positions (solid stars)
    geom_star(data = players_predict_actual,
              aes(x = x, y = y, color = player_side, fill = player_side),
              size = 8, alpha = 1) +
    geom_text(data = players_predict_actual,
              aes(x = x, y = y, label = player_position, color = player_side),
              vjust = -1.5, size = 5, fontface = "bold") +
    
    # Predicted players - PREDICTED positions (hollow stars in output phase)
    geom_star(data = players_predict_predicted %>% filter(phase == "OUTPUT"),
              aes(x = x, y = y, color = player_side),
              fill = NA, size = 8, alpha = 0.6) +
    
    # Ball landing X
    {if(nrow(ball_land_data) > 0) {
      geom_point(data = ball_land_data,
                 aes(x = ball_land_x_adj, y = ball_land_y_adj),
                 shape = 4, size = 8, color = "black",
                 inherit.aes = FALSE)
    }} +
    
    # Phase indicator - dynamic based on frame
    geom_text(data = phase_labels,
              aes(x = 10, y = 55, label = phase_label),
              size = 6, fontface = "bold", color = "darkred",
              inherit.aes = FALSE) +
    
    # Legend for prediction visualization
    annotate("text", x = 110, y = 55, 
             label = "★ Solid = Actual\n☆ Hollow = Predicted", 
             size = 4, hjust = 1, color = "black") +
    
    # RMSE display
    {if(!is.na(rmse_data) && !is.infinite(output_start_frame)) {
      annotate("text", x = 60, y = 55,
               label = sprintf("RMSE: %.2f yards", rmse_data),
               size = 5, fontface = "bold", color = "purple")
    }} +
    
    coord_fixed(xlim = c(0, 120), ylim = c(-5, 58.33)) +
    theme_void() +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    
    # Color scales
    scale_color_manual(
      values = c("Offense" = "blue", "Defense" = "red")
    ) +
    scale_fill_manual(
      values = c("Offense" = "blue", "Defense" = "red"),
      guide = "none"
    ) +
    
    labs(title = paste0("Game: ", unique(one_play$game_id), 
                        ", Play: ", unique(one_play$play_id), 
                        ", Frame: {frame_time}")) +
    transition_time(frame_id)
  
  # Animate
  anim <- animate(
    p,
    nframes = length(unique(one_play$frame_id)),
    fps = 10,
    width = 1000,
    height = 600,
    renderer = gifski_renderer()
  )
  
  # Save gif
  anim_save(save_path, animation = anim)
  
  cat("Animation saved to:", save_path, "\n")
  cat("Total frames:", length(unique(one_play$frame_id)), "\n")
  cat("Input frames:", sum(one_play$phase == "INPUT" & one_play$data_type == "actual"), "\n")
  cat("Output frames:", sum(one_play$phase == "OUTPUT" & one_play$data_type == "actual"), "\n")
  if (!is.na(rmse_data)) {
    cat("RMSE:", round(rmse_data, 3), "yards\n")
  }
  
  return(anim)
}

# Main workflow for first play from week 01 ------------------------------

# Step 1: Generate predictions (you need to have a trained model)
# If you have a trained model:
best_model_transformer <- train_transformer_model(input_data, output_data)
predictions_w01 <- predict_transformer_model(best_model_transformer, "train/input_2023_w01.csv")
new_predictions <- predict_transformer_autoregressive(final_model,"train/input_2023_w01.csv", sequence_length = 5)
# For demonstration, let's assume you have predictions saved or just generated
# predictions_w01 <- read_csv("predictions_w01.csv")

# Step 2: Combine all data
combined_data <- combine_all_data("train/input_2023_w01.csv", 
                                 "train/output_2023_w01.csv",
                                 new_predictions)

# Step 3: Animate the first play
animate_prediction_vs_actual(combined_data, play_index = 1, 
                             save_path = "first_play_comparison.gif")

