"""Training utilities for chess model."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional


class ChessTrainer:
    """Handles training loop for chess model."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 64
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"✓ Trainer initialized on {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
    
    def prepare_data(
        self,
        positions: np.ndarray,
        moves: np.ndarray,
        train_split: float = 0.9
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders from numpy arrays.
        
        Args:
            positions: Numpy array of board positions (N, 8, 8, 12)
            moves: Numpy array of move indices (N,)
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"\nPreparing data loaders...")
        print(f"  Total samples: {len(positions)}")
        
        # Convert to tensors and permute positions to (N, 12, 8, 8)
        X = torch.FloatTensor(positions).permute(0, 3, 1, 2)
        y = torch.LongTensor(moves)
        
        # Split data
        split_idx = int(train_split * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (batch_X, batch_y) in enumerate(pbar):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if (i + 1) % 50 == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                predictions = outputs.argmax(1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / num_batches
        
        return accuracy, avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: str,
        starting_epoch: int = 0,
        best_acc: float = 0.0
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save best model
            starting_epoch: Starting epoch number (for resuming)
            best_acc: Best accuracy so far (for resuming)
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(starting_epoch + 1, starting_epoch + epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_acc, val_loss = self.validate(val_loader)
            
            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)
                print(f"  ✓ New best model saved! (Acc: {best_acc:.2f}%)")
            
            print(f"  Best Accuracy: {best_acc:.2f}%")
            print()
        
        print(f"{'='*60}")
        print(f"Training complete!")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return {
            'best_acc': best_acc,
            'final_epoch': starting_epoch + epochs,
            'history': history
        }
    
    def load_optimizer_state(self, checkpoint: dict) -> None:
        """
        Load optimizer state from checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Loaded optimizer state")
            except Exception as e:
                print(f"⚠ Could not load optimizer state: {e}")
