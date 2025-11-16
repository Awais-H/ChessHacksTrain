# ChessHacks Training

Incremental training pipeline for chess move prediction using Modal + HuggingFace.

## Process

1. Download model from HuggingFace (or start fresh)
2. Download Lichess PGN database
3. Process PGN data into training format
4. Train model incrementally on processed data
5. Upload trained model back to HuggingFace

## Quick Start

### 1. Setup Environment

**Recommended: Use a virtual environment** (best practice for Python projects)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Modal
modal setup
```

**Note:** While Modal runs in isolated containers (so venv isn't strictly required for Modal execution), using a venv is highly recommended for:

- Local development and testing
- Running `check_model_status.py` and other local scripts
- Avoiding dependency conflicts with other projects

### 2. Configure HuggingFace

```bash
# Create Modal secret with your HF write token
modal secret create huggingface-secret HF_TOKEN=your_write_token

# Edit lib/config.py to set your HF_REPO_ID
```

### 3. Run Training

```bash
# Train on default dataset (index 0)
modal run scripts/modal_incremental_train.py

# Train on specific dataset
modal run scripts/modal_incremental_train.py --dataset-index 0
```

### 4. Other Commands

```bash
# Upload existing model to HuggingFace
modal run scripts/upload_to_hf.py

# Check model status (requires venv/local setup)
python scripts/check_model_status.py
```

## Architecture

- **Modal**: Handles GPU training in the cloud with persistent volumes for data caching
- **HuggingFace Hub**: Stores and version-controls model checkpoints
- **Lichess Data**: High-quality chess games for training
- **Incremental Training**: Continues from previous checkpoints, tracks best accuracy
