"""
Incremental training on Modal: downloads model from HF → trains on Lichess data → uploads back to HF.
Usage: modal run scripts/modal_incremental_train.py --dataset-index 0
"""

import modal
import os
import sys
from pathlib import Path

app = modal.App("chess-incremental-training")

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
    volumes={"/data": modal.Volume.from_name("chess-data", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")]  # Store HF_TOKEN in Modal secrets
)
def train_incremental(dataset_index: int = 0):
    """
    Incremental training function.
    
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
        BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT
    )
    from model import ChessModel
    from hf_utils import download_model, upload_model, load_checkpoint
    from data_processing import load_or_process_dataset
    from trainer import ChessTrainer
    
    print("="*60)
    print("CHESS MODEL INCREMENTAL TRAINING")
    print("="*60)
    
    # Setup paths
    model_path = "/data/chess_model.pth"
    cache_dir = "/data/datasets"
    
    # Step 1: Download existing model from HuggingFace
    print("\n[1/5] Downloading existing model from HuggingFace...")
    downloaded_path = download_model(HF_REPO_ID, HF_MODEL_FILENAME, model_path)
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessModel()
    
    # Load checkpoint if available
    checkpoint = load_checkpoint(model, model_path, device)
    starting_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"  Starting from epoch: {starting_epoch}")
    print(f"  Best accuracy so far: {best_acc:.2f}%")
    
    # Step 3: Load or process dataset
    print("\n[3/5] Loading dataset...")
    if dataset_index >= len(LICHESS_DATASETS):
        print(f"⚠ Invalid dataset index {dataset_index}. Using index 0.")
        dataset_index = 0
    
    dataset_url = LICHESS_DATASETS[dataset_index]
    print(f"  Dataset: {dataset_url.split('/')[-1]}")
    
    positions, moves = load_or_process_dataset(
        dataset_url=dataset_url,
        cache_dir=cache_dir,
        max_positions=MAX_POSITIONS,
        min_rating=MIN_RATING,
        skip_moves=SKIP_OPENING_MOVES
    )
    
    if positions is None:
        print("✗ Failed to load dataset. Exiting.")
        return {"success": False, "error": "Dataset loading failed"}
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    trainer = ChessTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )
    
    # Load optimizer state if available
    if checkpoint:
        trainer.load_optimizer_state(checkpoint)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        positions=positions,
        moves=moves,
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
    print("\n[5/5] Uploading model to HuggingFace...")
    commit_msg = f"Incremental training - Epoch {results['final_epoch']} - Acc: {results['best_acc']:.2f}%"
    
    upload_success = upload_model(
        model_path=model_path,
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        commit_message=commit_msg
    )
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset_url.split('/')[-1]}")
    print(f"Final Epoch: {results['final_epoch']}")
    print(f"Best Accuracy: {results['best_acc']:.2f}%")
    print(f"Upload Success: {upload_success}")
    print("="*60 + "\n")
    
    return {
        "success": True,
        "best_acc": results['best_acc'],
        "final_epoch": results['final_epoch'],
        "uploaded": upload_success
    }


@app.local_entrypoint()
def main(dataset_index: int = 0):
    """
    Local entrypoint for Modal.
    
    Args:
        dataset_index: Which dataset to train on (0, 1, 2, etc.)
    """
    print(f"Starting incremental training with dataset index {dataset_index}...")
    result = train_incremental.remote(dataset_index)
    
    if result["success"]:
        print(f"\n✓ Training completed successfully!")
        print(f"  Best Accuracy: {result['best_acc']:.2f}%")
        print(f"  Final Epoch: {result['final_epoch']}")
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
