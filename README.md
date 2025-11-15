# ChessHacks Training

Incremental training pipeline for chess move prediction using Modal + HuggingFace.

## Quick Start

```bash
# Setup
pip install -r requirements.txt
modal setup

# Configure HF token (write access required)
modal secret create huggingface-secret HF_TOKEN=your_write_token

# Edit lib/config.py to set your HF_REPO_ID

# Train
modal run scripts/modal_incremental_train.py

# Upload existing model
modal run scripts/upload_to_hf.py

# Check model status
python scripts/check_model_status.py
```
