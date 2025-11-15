#!/usr/bin/env python3
"""
Check model status from HuggingFace (shows epoch, accuracy, etc).
Usage: python scripts/check_model_status.py
"""

import sys
import os

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from config import HF_REPO_ID, HF_MODEL_FILENAME
from hf_utils import download_model, load_checkpoint
from model import ChessModel
import torch


def main():
    print("="*60)
    print("CHESS MODEL STATUS CHECK")
    print("="*60)
    print(f"\nRepository: {HF_REPO_ID}")
    print(f"Model file: {HF_MODEL_FILENAME}\n")
    
    # Download model
    print("Downloading model from HuggingFace...")
    model_path = download_model(HF_REPO_ID, HF_MODEL_FILENAME)
    
    if not model_path:
        print("\n✗ Could not download model")
        print("  The model may not exist yet, or there's a connection issue.")
        return
    
    print(f"\n✓ Model downloaded to: {model_path}\n")
    
    # Load checkpoint info
    print("Loading checkpoint information...")
    model = ChessModel()
    checkpoint = load_checkpoint(model, model_path)
    
    if not checkpoint:
        print("\n⚠ No checkpoint metadata found")
        print("  This might be a raw state_dict without training info.")
        return
    
    # Display checkpoint info
    print("\n" + "="*60)
    print("CHECKPOINT INFORMATION")
    print("="*60)
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'best_acc' in checkpoint:
        print(f"Best Accuracy: {checkpoint['best_acc']:.2f}%")
    
    if 'val_acc' in checkpoint:
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    if 'train_loss' in checkpoint:
        print(f"Training Loss: {checkpoint['train_loss']:.4f}")
    
    if 'val_loss' in checkpoint:
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Model size
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel Size: {file_size:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
