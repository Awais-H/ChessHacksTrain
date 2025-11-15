"""Training configuration and constants."""

# HuggingFace settings
HF_REPO_ID = "ricfinity242/chess"
HF_MODEL_FILENAME = "chess_model.pth"

# Data processing settings
MAX_POSITIONS = 500000
MIN_RATING = 1800
SKIP_OPENING_MOVES = 10

# Lichess database URLs (add more datasets as needed)
LICHESS_DATASETS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2018-01.pgn.zst",
]

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning
EPOCHS = 5  # Fewer epochs for incremental training
TRAIN_SPLIT = 0.9

# Model architecture
INPUT_CHANNELS = 12  # 6 piece types × 2 colors
BOARD_SIZE = 8
OUTPUT_SIZE = 4096  # 64 from_squares × 64 to_squares

# Modal settings
MODAL_GPU = "T4"
MODAL_TIMEOUT = 7200  # 2 hours
MODAL_VOLUME_NAME = "chess-data"
