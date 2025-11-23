# Full Play Animation - Input Phase + Output Phase
# This combines the input tracking with the output tracking to show the complete play

library(tidyverse)
library(gganimate)
library(gifski)
library(sportyR)
library(ggstar)

# Creating the football field ---------------------------------------------
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

# Read input and output data ----------------------------------------------
tracking_input <- read_csv("train/input_2023_w01.csv")
tracking_output <- read_csv("train/output_2023_w01.csv")

# Standardize field coordinates for INPUT ---------------------------------
tracking_input <- tracking_input %>%
  mutate(
    x = ifelse(play_direction == "left", 120 - x, x),
    y = ifelse(play_direction == "left", 160/3 - y, y),
    dir = ifelse(play_direction == "left", dir + 180, dir),
    dir = ifelse(dir > 360, dir - 360, dir),
    o = ifelse(play_direction == "left", o + 180, o),
    o = ifelse(o > 360, o - 360, o),
    ball_land_x = ifelse(play_direction == "left", 120 - ball_land_x, ball_land_x),
    ball_land_y = ifelse(play_direction == "left", 160/3 - ball_land_y, ball_land_y)
  )

# Pick a single play ------------------------------------------------------
unique_plays <- tracking_input %>% distinct(game_id, play_id)
play_index <- 1  # Change this to view different plays

one_play_input <- tracking_input %>%
  filter(game_id == unique_plays$game_id[play_index],
         play_id == unique_plays$play_id[play_index])

one_play_output <- tracking_output %>%
  filter(game_id == unique_plays$game_id[play_index],
         play_id == unique_plays$play_id[play_index])

# Get metadata from input to apply to output -----------------------------
play_metadata <- one_play_input %>%
  group_by(nfl_id) %>%
  slice_tail(n = 1) %>%
  select(game_id, play_id, nfl_id, play_direction, player_name, 
         player_position, player_side, player_role, player_to_predict,
         ball_land_x, ball_land_y, absolute_yardline_number) %>%
  ungroup()

# Join metadata to output and standardize coordinates --------------------
one_play_output <- one_play_output %>%
  left_join(play_metadata, by = c("game_id", "play_id", "nfl_id")) %>%
  mutate(
    x = ifelse(play_direction == "left", 120 - x, x),
    y = ifelse(play_direction == "left", 160/3 - y, y)
  )

# Get the maximum frame_id from input for each player ---------------------
max_input_frames <- one_play_input %>%
  group_by(nfl_id) %>%
  summarise(max_frame = max(frame_id), .groups = 'drop')

# Adjust output frame_ids to continue from input
one_play_output <- one_play_output %>%
  left_join(max_input_frames, by = "nfl_id") %>%
  mutate(frame_id = frame_id + max_frame) %>%
  select(-max_frame)

# Add phase labels --------------------------------------------------------
one_play_input <- one_play_input %>% mutate(phase = "INPUT")
one_play_output <- one_play_output %>% mutate(phase = "OUTPUT")

# Combine input and output ------------------------------------------------
one_play_full <- bind_rows(one_play_input, one_play_output)

# Get the starting position based on absolute_yardline_number ------------
start_yard <- unique(one_play_full$absolute_yardline_number)[1]

# Adjust coordinates to center the play at the starting yard line
x_offset <- start_yard - mean(one_play_input$x[one_play_input$frame_id == min(one_play_input$frame_id)], na.rm = TRUE)
one_play_full <- one_play_full %>%
  mutate(x = x + x_offset)

# Get ball landing coordinates and adjust them the same way ---------------
ball_land_data <- one_play_full %>%
  filter(!is.na(ball_land_x) & !is.na(ball_land_y)) %>%
  distinct(ball_land_x, ball_land_y) %>%
  mutate(
    ball_land_x_adj = ball_land_x + x_offset,
    ball_land_y_adj = ball_land_y
  )

# Separate players to predict from others ---------------------------------
players_normal <- one_play_full %>% filter(!player_to_predict)
players_predict <- one_play_full %>% filter(player_to_predict)

# Create path traces for predicted players in output phase ----------------
path_predict <- players_predict %>%
  filter(phase == "OUTPUT") %>%
  arrange(nfl_id, frame_id)

# Find when output phase starts -------------------------------------------
output_start_frame <- min(one_play_full$frame_id[one_play_full$phase == "OUTPUT"], na.rm = TRUE)

# Plot setup with field background ----------------------------------------
p <- field_background +
  # Normal players (circles)
  geom_point(data = players_normal, 
             aes(x = x, y = y, color = player_side), 
             size = 8) +
  geom_text(data = players_normal,
            aes(x = x, y = y, label = player_position, color = player_side),
            vjust = -1.5, size = 5) +
  
  # Predicted players (stars)
  geom_star(data = players_predict,
            aes(x = x, y = y, color = player_side, fill = player_side),
            size = 8, alpha = 1) +
  geom_text(data = players_predict,
            aes(x = x, y = y, label = player_position, color = player_side),
            vjust = -1.5, size = 5, fontface = "bold") +
  
  # Path traces for predicted players during output phase
  geom_path(data = path_predict,
            aes(x = x, y = y, group = nfl_id, color = player_side),
            alpha = 0.4, linewidth = 1.5, linetype = "solid") +
  
  # Ball landing X
  {if(nrow(ball_land_data) > 0) {
    geom_point(data = ball_land_data,
               aes(x = ball_land_x_adj, y = ball_land_y_adj),
               shape = 4, size = 8, stroke = 2, color = "black",
               inherit.aes = FALSE)
  }} +
  
  # Phase indicator - Dynamic based on current frame
  geom_text(aes(x = 10, y = 55, 
                label = ifelse(frame_id >= output_start_frame, "OUTPUT PHASE", "INPUT PHASE")),
            data = one_play_full,
            size = 6, fontface = "bold", color = "darkred") +
  
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
  
  labs(title = paste0("Game: ", unique(one_play_full$game_id), 
                      ", Play: ", unique(one_play_full$play_id), 
                      ", Frame: {frame_time}")) +
  transition_time(frame_id)

# Animate -----------------------------------------------------------------
anim <- animate(
  p,
  nframes = length(unique(one_play_full$frame_id)),
  fps = 10,
  width = 1000,
  height = 600,
  renderer = gifski_renderer()
)

# Save gif ----------------------------------------------------------------
anim_save("full_play_animation.gif", animation = anim)

cat("Animation saved: full_play_animation.gif\n")
cat("Total frames:", length(unique(one_play_full$frame_id)), "\n")
cat("Input phase frames:", sum(one_play_full$phase == "INPUT"), "\n")
cat("Output phase frames:", sum(one_play_full$phase == "OUTPUT"), "\n")
cat("Unique players:", n_distinct(one_play_full$nfl_id), "\n")
cat("Players to predict:", sum(one_play_full$player_to_predict[one_play_full$frame_id == 1]), "\n")