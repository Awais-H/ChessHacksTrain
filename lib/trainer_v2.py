"""Enhanced training utilities with value head support."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional


class ChessTrainerV2:
    """Handles training loop for enhanced chess model with optional value head."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        use_value_head: bool = False,
        value_loss_weight: float = 0.1
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            use_value_head: Whether to train value head alongside policy
            value_loss_weight: Weight for value loss (relative to policy loss)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.use_value_head = use_value_head
        self.value_loss_weight = value_loss_weight
        
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"✓ Trainer initialized on {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Use value head: {use_value_head}")
        if use_value_head:
            print(f"  Value loss weight: {value_loss_weight}")
    
    def prepare_data(
        self,
        positions: np.ndarray,
        moves: np.ndarray,
        values: Optional[np.ndarray] = None,
        train_split: float = 0.9
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders from numpy arrays.
        
        Args:
            positions: Numpy array of board positions (N, 8, 8, C)
            moves: Numpy array of move indices (N,)
            values: Optional numpy array of position evaluations (N,)
            train_split: Fraction of data for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"\nPreparing data loaders...")
        print(f"  Total samples: {len(positions)}")
        
        # Convert to tensors and permute positions to (N, C, 8, 8)
        X = torch.FloatTensor(positions).permute(0, 3, 1, 2)
        y_policy = torch.LongTensor(moves)
        
        # Split data
        split_idx = int(train_split * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_policy_train, y_policy_val = y_policy[:split_idx], y_policy[split_idx:]
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        
        # Create datasets
        if self.use_value_head and values is not None:
            y_value = torch.FloatTensor(values).unsqueeze(1)
            y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]
            
            train_dataset = TensorDataset(X_train, y_policy_train, y_value_train)
            val_dataset = TensorDataset(X_val, y_policy_val, y_value_val)
        else:
            train_dataset = TensorDataset(X_train, y_policy_train)
            val_dataset = TensorDataset(X_val, y_policy_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (avg_total_loss, avg_policy_loss, avg_value_loss)
        """
        self.model.train()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            if self.use_value_head and len(batch) == 3:
                batch_X, batch_y_policy, batch_y_value = batch
                batch_X = batch_X.to(self.device)
                batch_y_policy = batch_y_policy.to(self.device)
                batch_y_value = batch_y_value.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                policy_output, value_output = self.model(batch_X, return_value=True)
                
                # Calculate losses
                policy_loss = self.policy_criterion(policy_output, batch_y_policy)
                value_loss = self.value_criterion(value_output, batch_y_value)
                total_loss = policy_loss + self.value_loss_weight * value_loss
                
                value_loss_sum += value_loss.item()
            else:
                batch_X, batch_y_policy = batch
                batch_X = batch_X.to(self.device)
                batch_y_policy = batch_y_policy.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                policy_output = self.model(batch_X)
                
                # Calculate loss
                policy_loss = self.policy_criterion(policy_output, batch_y_policy)
                total_loss = policy_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            num_batches += 1
            
            # Update progress bar
            if (i + 1) % 50 == 0:
                avg_total = total_loss_sum / num_batches
                avg_policy = policy_loss_sum / num_batches
                if self.use_value_head:
                    avg_value = value_loss_sum / num_batches
                    pbar.set_postfix({
                        'loss': f'{avg_total:.4f}',
                        'policy': f'{avg_policy:.4f}',
                        'value': f'{avg_value:.4f}'
                    })
                else:
                    pbar.set_postfix({'loss': f'{avg_total:.4f}'})
        
        avg_total = total_loss_sum / num_batches
        avg_policy = policy_loss_sum / num_batches
        avg_value = value_loss_sum / num_batches if self.use_value_head else 0.0
        
        return avg_total, avg_policy, avg_value
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (accuracy, total_loss, policy_loss, value_loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if self.use_value_head and len(batch) == 3:
                    batch_X, batch_y_policy, batch_y_value = batch
                    batch_X = batch_X.to(self.device)
                    batch_y_policy = batch_y_policy.to(self.device)
                    batch_y_value = batch_y_value.to(self.device)
                    
                    policy_output, value_output = self.model(batch_X, return_value=True)
                    
                    policy_loss = self.policy_criterion(policy_output, batch_y_policy)
                    value_loss = self.value_criterion(value_output, batch_y_value)
                    total_loss = policy_loss + self.value_loss_weight * value_loss
                    
                    value_loss_sum += value_loss.item()
                else:
                    batch_X, batch_y_policy = batch
                    batch_X = batch_X.to(self.device)
                    batch_y_policy = batch_y_policy.to(self.device)
                    
                    policy_output = self.model(batch_X)
                    policy_loss = self.policy_criterion(policy_output, batch_y_policy)
                    total_loss = policy_loss
                
                predictions = policy_output.argmax(1)
                correct += (predictions == batch_y_policy).sum().item()
                total += batch_y_policy.size(0)
                
                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                num_batches += 1
        
        accuracy = 100.0 * correct / total
        avg_total = total_loss_sum / num_batches
        avg_policy = policy_loss_sum / num_batches
        avg_value = value_loss_sum / num_batches if self.use_value_head else 0.0
        
        return accuracy, avg_total, avg_policy, avg_value
    
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
            'train_policy_loss': [],
            'train_value_loss': [],
            'val_loss': [],
            'val_policy_loss': [],
            'val_value_loss': [],
            'val_acc': []
        }
        
        for epoch in range(starting_epoch + 1, starting_epoch + epochs + 1):
            # Train
            train_total, train_policy, train_value = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_acc, val_total, val_policy, val_value = self.validate(val_loader)
            
            # Track history
            history['train_loss'].append(train_total)
            history['train_policy_loss'].append(train_policy)
            history['train_value_loss'].append(train_value)
            history['val_loss'].append(val_total)
            history['val_policy_loss'].append(val_policy)
            history['val_value_loss'].append(val_value)
            history['val_acc'].append(val_acc)
            
            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_total:.4f} (policy: {train_policy:.4f}", end="")
            if self.use_value_head:
                print(f", value: {train_value:.4f})")
            else:
                print(")")
            print(f"  Val Loss: {val_total:.4f} (policy: {val_policy:.4f}", end="")
            if self.use_value_head:
                print(f", value: {val_value:.4f})")
            else:
                print(")")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'train_loss': train_total,
                    'val_loss': val_total,
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

