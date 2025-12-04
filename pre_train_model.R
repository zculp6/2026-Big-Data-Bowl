# Load all training data
train_dir <- 'C:/Users/ASUS/OneDrive/Desktop/Big Data Bowl 2026/train/'
input_files <- list.files(train_dir, pattern = "input_2023_w.*\\.csv", full.names = TRUE)
output_files <- list.files(train_dir, pattern = "output_2023_w.*\\.csv", full.names = TRUE)

cat("Found", length(input_files), "input files\n")
cat("Found", length(output_files), "output files\n")

# Load data
input_data <- map_df(input_files, read_csv, show_col_types = FALSE)
output_data <- map_df(output_files, read_csv, show_col_types = FALSE)

cat("Loaded", nrow(input_data), "input rows\n")
cat("Loaded", nrow(output_data), "output rows\n")

# Train model
final_model <- train_transformer_model(
  input_data, output_data,
  num_heads = 4, 
  key_dim = 16, 
  ff_dim = 512,
  epochs = 25,  # Adjust as needed
  batch_size = 32,
  sequence_length = 5,
  dropout_rate = 0.17
)

# Save the model
save_model(final_model, "nfl_transformer_model.keras")
