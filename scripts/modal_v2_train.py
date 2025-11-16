"""
V2 Model Training on Modal - Fresh start with piece value awareness.
Usage: modal run scripts/modal_v2_train.py --dataset-index 0
"""

import modal
import os
import sys
from pathlib import Path

app = modal.App("chess-v2-training")

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create Modal image with all dependencies and copy lib files
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "python-chess",
        "tqdm",
        "huggingface-hub"
    )
    .apt_install("wget", "zstd")
    .add_local_dir(
        local_path=str(project_root / "lib"),
        remote_path="/root/lib",
        copy=True
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={"/data": modal.Volume.from_name("chess-data-v2", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_v2(dataset_index: int = 0):
    """
    Train V2 model from scratch.
    
    Args:
        dataset_index: Index of dataset to use from config.LICHESS_DATASETS
    """
    import sys
    import torch
    import numpy as np
    
    # Add lib to path for imports
    sys.path.insert(0, '/root/lib')
    
    # Import our modules
    from config import (
        HF_REPO_ID, HF_MODEL_FILENAME,
        LICHESS_DATASETS, MAX_POSITIONS, MIN_RATING, SKIP_OPENING_MOVES,
        BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT,
        USE_V2_MODEL, USE_VALUE_HEAD, VALUE_LOSS_WEIGHT, 
        NUM_RESIDUAL_BLOCKS, INCLUDE_PIECE_VALUES
    )
    from model_v2 import ChessModelV2
    from hf_utils import download_model, upload_model, load_checkpoint
    from data_processing_v2 import load_or_process_dataset_v2
    from trainer_v2 import ChessTrainerV2
    
    print("="*60)
    print("CHESS MODEL V2 TRAINING (FRESH START)")
    print("="*60)
    print(f"Model: V2 with {NUM_RESIDUAL_BLOCKS} residual blocks")
    print(f"Value Head: {'Enabled' if USE_VALUE_HEAD else 'Disabled'}")
    print(f"Piece Values: {'Included' if INCLUDE_PIECE_VALUES else 'Not included'}")
    print("="*60)
    
    # Setup paths
    model_path = "/data/chess_model_v2.pth"
    cache_dir = "/data/datasets_v2"
    
    # Step 1: Try to download existing V2 model (if continuing V2 training)
    print("\n[1/5] Checking for existing V2 model...")
    downloaded_path = download_model(HF_REPO_ID, "chess_model_v2.pth", model_path)
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing V2 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessModelV2(
        use_piece_values=INCLUDE_PIECE_VALUES,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS
    )
    
    # Load checkpoint if available (V2 checkpoint only)
    starting_epoch = 0
    best_acc = 0.0
    if downloaded_path and os.path.exists(model_path):
        checkpoint = load_checkpoint(model, model_path, device)
        starting_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"  Resuming V2 training from epoch: {starting_epoch}")
        print(f"  Best accuracy so far: {best_acc:.2f}%")
    else:
        print("  Starting fresh V2 training")
    
    # Step 3: Load or process dataset
    print("\n[3/5] Loading dataset...")
    if dataset_index >= len(LICHESS_DATASETS):
        print(f"⚠ Invalid dataset index {dataset_index}. Using index 0.")
        dataset_index = 0
    
    dataset_url = LICHESS_DATASETS[dataset_index]
    print(f"  Dataset: {dataset_url.split('/')[-1]}")
    
    positions, moves, values = load_or_process_dataset_v2(
        dataset_url=dataset_url,
        cache_dir=cache_dir,
        max_positions=MAX_POSITIONS,
        min_rating=MIN_RATING,
        skip_moves=SKIP_OPENING_MOVES,
        include_values=INCLUDE_PIECE_VALUES,
        include_position_eval=USE_VALUE_HEAD
    )
    
    if positions is None:
        print("✗ Failed to load dataset. Exiting.")
        return {"success": False, "error": "Dataset loading failed"}
    
    print(f"  Loaded {len(positions)} positions")
    print(f"  Input shape: {positions.shape}")  # Should be (N, 8, 8, 19)
    
    # Step 4: Train model
    print("\n[4/5] Training V2 model...")
    trainer = ChessTrainerV2(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        use_value_head=USE_VALUE_HEAD,
        value_loss_weight=VALUE_LOSS_WEIGHT
    )
    
    # Load optimizer state if available
    if downloaded_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        trainer.load_optimizer_state(checkpoint)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        positions=positions,
        moves=moves,
        values=values,
        train_split=TRAIN_SPLIT
    )
    
    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path=model_path,
        starting_epoch=starting_epoch,
        best_acc=best_acc
    )
    
    # Step 5: Upload to HuggingFace
    print("\n[5/5] Uploading V2 model to HuggingFace...")
    commit_msg = f"V2 Model - Epoch {results['final_epoch']} - Acc: {results['best_acc']:.2f}%"
    
    upload_success = upload_model(
        model_path=model_path,
        repo_id=HF_REPO_ID,
        filename="chess_model_v2.pth",  # Different filename for V2
        commit_message=commit_msg
    )
    
    # Final summary
    print("\n" + "="*60)
    print("V2 TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset_url.split('/')[-1]}")
    print(f"Final Epoch: {results['final_epoch']}")
    print(f"Best Accuracy: {results['best_acc']:.2f}%")
    print(f"Upload Success: {upload_success}")
    print(f"Model saved as: chess_model_v2.pth")
    print("="*60 + "\n")
    
    return {
        "success": True,
        "best_acc": results['best_acc'],
        "final_epoch": results['final_epoch'],
        "uploaded": upload_success,
        "model_version": "v2"
    }


@app.local_entrypoint()
def main(dataset_index: int = 0):
    """
    Local entrypoint for Modal.
    
    Args:
        dataset_index: Which dataset to train on (0, 1, 2, etc.)
    """
    print(f"Starting V2 model training with dataset index {dataset_index}...")
    result = train_v2.remote(dataset_index)
    
    if result["success"]:
        print(f"\n✓ V2 Training completed successfully!")
        print(f"  Best Accuracy: {result['best_acc']:.2f}%")
        print(f"  Final Epoch: {result['final_epoch']}")
        print(f"  Model Version: {result['model_version']}")
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")

