"""HuggingFace utilities for model download and upload."""

import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import torch


def download_model(repo_id: str, filename: str, local_path: str = None) -> str:
    """
    Download model from HuggingFace Hub.
    
    Args:
        repo_id: HF repository ID (e.g., "username/repo")
        filename: Model filename in the repo
        local_path: Optional local path to save the model
        
    Returns:
        Path to the downloaded model file
    """
    try:
        print(f"Downloading {filename} from {repo_id}...")
        
        if local_path:
            # Download to specific location
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(local_path),
                local_dir_use_symlinks=False
            )
            # Rename if needed
            if model_path != local_path:
                os.rename(model_path, local_path)
                model_path = local_path
        else:
            # Download to cache
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename
            )
        
        print(f"✓ Model downloaded to: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"⚠ Could not download model: {e}")
        print("Starting fresh training...")
        return None


def upload_model(
    model_path: str,
    repo_id: str,
    filename: str,
    token: str = None,
    commit_message: str = None
) -> bool:
    """
    Upload model to HuggingFace Hub.
    
    Args:
        model_path: Local path to the model file
        repo_id: HF repository ID
        filename: Filename in the repo
        token: HF API token (reads from HF_TOKEN env var if not provided)
        commit_message: Optional commit message
        
    Returns:
        True if upload successful, False otherwise
    """
    token = token or os.getenv("HF_TOKEN")
    
    if not token:
        print("⚠ No HF_TOKEN found. Skipping upload.")
        return False
    
    if not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        return False
    
    try:
        print(f"Uploading {model_path} to {repo_id}/{filename}...")
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or f"Update {filename}"
        )
        
        print(f"✓ Model uploaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False


def load_checkpoint(model, checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Dictionary with checkpoint metadata (if available)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh training.")
        return {}
    
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both raw state_dict and full checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
            print(f"✓ Loaded checkpoint with metadata: {metadata}")
            return metadata
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded model weights")
            return {}
            
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        print("Starting fresh training...")
        return {}


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_acc: float,
    save_path: str,
    additional_info: dict = None
) -> None:
    """
    Save training checkpoint with metadata.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        best_acc: Best validation accuracy
        save_path: Path to save checkpoint
        additional_info: Additional metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved to {save_path}")
