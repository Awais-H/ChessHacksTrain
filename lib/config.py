"""Training configuration and constants."""

# HuggingFace settings
HF_REPO_ID = "ricfinity242/chess"
HF_MODEL_FILENAME = "chess_model.pth"

# Data processing settings
MAX_POSITIONS = 500000
MIN_RATING = 2300
SKIP_OPENING_MOVES = 10

# Lichess database URLs (add more datasets as needed)
LICHESS_DATASETS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2016-06.pgn.zst",
]

# Training hyperparameters
TRAIN_SPLIT = 0.9

# Model architecture
INPUT_CHANNELS = 12  # 6 piece types × 2 colors
BOARD_SIZE = 8
OUTPUT_SIZE = 4096  # 64 from_squares × 64 to_squares

# Modal settings
MODAL_GPU = "A100"
MODAL_TIMEOUT = 7200  # 2 hours
MODAL_VOLUME_NAME = "chess-data"

USE_V2_MODEL = True  # Enable V2
USE_VALUE_HEAD = True  # Enable position evaluation
VALUE_LOSS_WEIGHT = 0.1  # Weight for value loss
NUM_RESIDUAL_BLOCKS = 4  # Depth of network
INCLUDE_PIECE_VALUES = True  # Include material channel

LEARNING_RATE = 0.00005  # Lower LR for deeper network
BATCH_SIZE = 64
EPOCHS = 5  # V2 needs more epochs initially