#!/usr/bin/env python3
"""
Compare V1 and V2 model architectures and parameter counts.
Usage: python scripts/compare_models.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import torch
from model import ChessModel as ChessModelV1
from model_v2 import ChessModelV2


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Estimate model size in MB (float32)."""
    return count_parameters(model) * 4 / (1024 * 1024)


def main():
    print("="*70)
    print("CHESS MODEL COMPARISON: V1 vs V2")
    print("="*70)
    
    # V1 Model
    print("\n📦 V1 MODEL (Original)")
    print("-" * 70)
    model_v1 = ChessModelV1()
    params_v1 = count_parameters(model_v1)
    size_v1 = get_model_size_mb(model_v1)
    
    print(f"Architecture:")
    print(f"  Input: 12 channels (piece positions only)")
    print(f"  Layers: Conv(12→64) → Conv(64→128) → Conv(128→128) → Pool → FC")
    print(f"  Output: 4096 move logits")
    print(f"\nParameters: {params_v1:,}")
    print(f"Model Size: {size_v1:.2f} MB")
    print(f"Features: Basic piece positions")
    
    # V2 Model (without value head)
    print("\n\n📦 V2 MODEL (Enhanced - Policy Only)")
    print("-" * 70)
    model_v2_policy = ChessModelV2(use_piece_values=True, num_residual_blocks=4)
    params_v2_policy = count_parameters(model_v2_policy)
    size_v2_policy = get_model_size_mb(model_v2_policy)
    
    print(f"Architecture:")
    print(f"  Input: 19 channels (pieces + game state + material)")
    print(f"  Layers: Conv(19→128) → 4× ResBlock(128) → Policy Head")
    print(f"  Output: 4096 move logits")
    print(f"\nParameters: {params_v2_policy:,}")
    print(f"Model Size: {size_v2_policy:.2f} MB")
    print(f"Features: Piece positions + castling + en passant + turn + material")
    print(f"Improvements: Residual blocks, batch normalization, deeper network")
    
    # V2 Model (with value head)
    print("\n\n📦 V2 MODEL (Enhanced - Policy + Value)")
    print("-" * 70)
    # Note: Same model, just using value head
    print(f"Architecture:")
    print(f"  Input: 19 channels (pieces + game state + material)")
    print(f"  Layers: Conv(19→128) → 4× ResBlock(128) → Policy + Value Heads")
    print(f"  Output: 4096 move logits + position evaluation")
    print(f"\nParameters: {params_v2_policy:,} (same as policy-only)")
    print(f"Model Size: {size_v2_policy:.2f} MB")
    print(f"Features: Same as policy-only + position evaluation head")
    print(f"Improvements: Multi-task learning, better feature representations")
    
    # Comparison
    print("\n\n📊 COMPARISON")
    print("="*70)
    param_increase = (params_v2_policy - params_v1) / params_v1 * 100
    size_increase = (size_v2_policy - size_v1) / size_v1 * 100
    
    print(f"Parameter Increase: +{param_increase:.1f}% ({params_v2_policy - params_v1:,} more)")
    print(f"Size Increase: +{size_increase:.1f}% ({size_v2_policy - size_v1:.2f} MB more)")
    print(f"\nExpected Performance:")
    print(f"  V1 @ 19 epochs: ~23% accuracy (current)")
    print(f"  V2 @ 19 epochs: ~30-35% accuracy (estimated)")
    print(f"  V2 @ 50 epochs: ~40-45% accuracy (estimated)")
    print(f"\nTraining Time:")
    print(f"  V2 is ~1.5-2x slower per epoch (more parameters + batch norm)")
    print(f"  But reaches higher accuracy faster (better features)")
    
    # Input comparison
    print("\n\n🔍 INPUT FEATURES COMPARISON")
    print("="*70)
    print(f"{'Feature':<30} {'V1':<10} {'V2':<10}")
    print("-" * 70)
    features = [
        ("Piece positions", "✓", "✓"),
        ("Castling rights", "✗", "✓"),
        ("En passant", "✗", "✓"),
        ("Turn indicator", "✗", "✓"),
        ("Material advantage", "✗", "✓"),
        ("Residual connections", "✗", "✓"),
        ("Batch normalization", "✗", "✓"),
        ("Value head (optional)", "✗", "✓"),
    ]
    
    for feature, v1, v2 in features:
        print(f"{feature:<30} {v1:<10} {v2:<10}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    print("Use V2 for:")
    print("  ✓ Better understanding of piece values")
    print("  ✓ Faster learning (fewer epochs to reach same accuracy)")
    print("  ✓ Higher final accuracy potential")
    print("  ✓ More robust to different positions")
    print("\nKeep V1 if:")
    print("  ✓ GPU memory is very limited")
    print("  ✓ Training time is critical")
    print("  ✓ Current accuracy (23%) is sufficient")
    print("="*70 + "\n")
    
    # Test forward pass
    print("🧪 TESTING FORWARD PASS")
    print("-" * 70)
    
    # V1 test
    dummy_input_v1 = torch.randn(1, 12, 8, 8)
    output_v1 = model_v1(dummy_input_v1)
    print(f"V1 Input shape: {dummy_input_v1.shape}")
    print(f"V1 Output shape: {output_v1.shape}")
    
    # V2 test (policy only)
    dummy_input_v2 = torch.randn(1, 19, 8, 8)
    output_v2_policy = model_v2_policy(dummy_input_v2, return_value=False)
    print(f"\nV2 Input shape: {dummy_input_v2.shape}")
    print(f"V2 Policy output shape: {output_v2_policy.shape}")
    
    # V2 test (policy + value)
    output_v2_policy, output_v2_value = model_v2_policy(dummy_input_v2, return_value=True)
    print(f"V2 Policy output shape: {output_v2_policy.shape}")
    print(f"V2 Value output shape: {output_v2_value.shape}")
    
    print("\n✓ All forward passes successful!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

