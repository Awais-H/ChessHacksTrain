"""
Train on Lichess Puzzle Database - PERFECT for winning intent!
Every puzzle has a correct winning move.

Usage: modal run scripts/modal_puzzle_train.py --theme mate
"""

import modal
import os
import sys
from pathlib import Path

app = modal.App("chess-puzzle-training")

project_root = Path(__file__).parent.parent

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
    timeout=7200,
    volumes={"/data": modal.Volume.from_name("chess-data-puzzles", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_on_puzzles(
    theme: str = "all_winning",
    continue_from_v2: bool = True,
    max_positions: int = 500000,
    min_rating: int = 1500,
    max_rating: int = 2500
):
    """
    Train on Lichess puzzles.
    
    Args:
        theme: Puzzle theme filter ("mate", "endgame", "tactical", "all_winning")
        continue_from_v2: Load V2/V3 checkpoint and continue
        max_positions: Maximum puzzles to use
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
    """
    import sys
    import torch
    
    sys.path.insert(0, '/root/lib')
    
    from config import (
        HF_REPO_ID,
        BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_SPLIT,
        USE_VALUE_HEAD, VALUE_LOSS_WEIGHT,
        NUM_RESIDUAL_BLOCKS, INCLUDE_PIECE_VALUES
    )
    from model_v2 import ChessModelV2
    from hf_utils import download_model, upload_model, load_checkpoint
    from data_processing_puzzles import (
        load_or_process_puzzle_database,
        THEME_FILTERS
    )
    from trainer_v2 import ChessTrainerV2
    
    print("="*60)
    print("LICHESS PUZZLE TRAINING")
    print("="*60)
    print(f"Theme: {theme}")
    print(f"Rating range: {min_rating}-{max_rating}")
    print(f"Max puzzles: {max_positions}")
    print(f"Continue from V2/V3: {continue_from_v2}")
    print("="*60)
    
    # Setup
    model_path = "/data/chess_model_v2.pth"  # Continue V2 evolution
    cache_dir = "/data/puzzle_cache"
    
    # Initialize model
    print("\n[1/5] Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessModelV2(
        use_piece_values=INCLUDE_PIECE_VALUES,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS
    )
    
    starting_epoch = 0
    best_acc = 0.0
    
    if continue_from_v2:
        # Load existing V2 checkpoint
        downloaded = download_model(HF_REPO_ID, "chess_model_v2.pth", model_path)
        if downloaded and os.path.exists(model_path):
            checkpoint = load_checkpoint(model, model_path, device)
            starting_epoch = checkpoint.get('epoch', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"  ✓ Loaded chess_model_v2.pth")
            print(f"  Starting from epoch: {starting_epoch}")
            print(f"  Best accuracy: {best_acc:.2f}%")
            print(f"  Will continue training with puzzle data")
        else:
            print("  ⚠ No V2 checkpoint found, starting fresh")
    
    # Load puzzle database
    print("\n[2/5] Loading Lichess puzzles...")
    themes_filter = THEME_FILTERS.get(theme)
    if themes_filter:
        print(f"  Filtering for themes: {themes_filter}")
    
    positions, moves, values = load_or_process_puzzle_database(
        cache_dir=cache_dir,
        max_positions=max_positions,
        min_rating=min_rating,
        max_rating=max_rating,
        include_values=INCLUDE_PIECE_VALUES,
        include_position_eval=USE_VALUE_HEAD,
        themes_filter=themes_filter
    )
    
    if positions is None:
        print("✗ Failed to load puzzles")
        return {"success": False, "error": "Puzzle loading failed"}
    
    print(f"  ✓ Loaded {len(positions)} puzzles")
    print(f"  Every puzzle has a correct winning move!")
    
    # Initialize trainer
    print("\n[3/5] Initializing trainer...")
    trainer = ChessTrainerV2(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        use_value_head=USE_VALUE_HEAD,
        value_loss_weight=VALUE_LOSS_WEIGHT
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        positions=positions,
        moves=moves,
        values=values,
        train_split=TRAIN_SPLIT
    )
    
    # Train
    print("\n[4/5] Training on puzzles...")
    print("  Goal: Learn winning moves from tactical puzzles")
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        save_path=model_path,
        starting_epoch=starting_epoch,
        best_acc=best_acc
    )
    
    # Upload
    print("\n[5/5] Uploading puzzle-trained model...")
    commit_msg = f"Puzzle Training ({theme}) - Epoch {results['final_epoch']} - Acc: {results['best_acc']:.2f}%"
    
    upload_success = upload_model(
        model_path=model_path,
        repo_id=HF_REPO_ID,
        filename="chess_model_v2.pth",  # Same file, continuous evolution
        commit_message=commit_msg
    )
    
    # Summary
    print("\n" + "="*60)
    print("PUZZLE TRAINING SUMMARY")
    print("="*60)
    print(f"Theme: {theme}")
    print(f"Puzzles trained: {len(positions)}")
    print(f"Final Epoch: {results['final_epoch']}")
    print(f"Best Accuracy: {results['best_acc']:.2f}%")
    print(f"Upload Success: {upload_success}")
    print(f"Model saved as: chess_model_v2.pth (continuous evolution)")
    print("\n💡 Model trained on winning moves from tactical puzzles!")
    print("="*60 + "\n")
    
    return {
        "success": True,
        "best_acc": results['best_acc'],
        "final_epoch": results['final_epoch'],
        "uploaded": upload_success,
        "theme": theme,
        "puzzles_count": len(positions)
    }


@app.local_entrypoint()
def main(
    theme: str = "all_winning",
    continue_from_v2: bool = True,
    max_positions: int = 500000,
    min_rating: int = 1500,
    max_rating: int = 2500
):
    """
    Train on Lichess puzzles.
    
    Args:
        theme: "mate", "endgame", "tactical", "advantage", or "all_winning"
        continue_from_v2: Continue from V2/V3 checkpoint
        max_positions: Maximum puzzles
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
    """
    print(f"Starting puzzle training...")
    print(f"  Theme: {theme}")
    print(f"  Continue from V2/V3: {continue_from_v2}")
    
    result = train_on_puzzles.remote(
        theme, continue_from_v2, max_positions, min_rating, max_rating
    )
    
    if result["success"]:
        print(f"\n✓ Puzzle training completed!")
        print(f"  Theme: {result['theme']}")
        print(f"  Puzzles: {result['puzzles_count']}")
        print(f"  Best Accuracy: {result['best_acc']:.2f}%")
        print(f"  Final Epoch: {result['final_epoch']}")
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")

