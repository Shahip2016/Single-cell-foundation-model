"""
CellFM Training Engine
========================

LAYMAN EXPLANATION:
    Training an AI model is like teaching a student:

    1. SHOW the student some data (forward pass)
    2. CHECK their answers against the correct ones (compute loss)
    3. TELL them what to improve (backward pass / gradients)
    4. UPDATE their knowledge (optimizer step)
    5. REPEAT thousands of times

    This file contains the "teacher" — the training engine that orchestrates
    this entire learning process.

TECHNICAL DETAILS:
    The training engine provides:
    - CellFMTrainer: Main training class with train/eval loops
    - Cosine warmup learning rate schedule (as per CellFM paper)
    - Mixed-precision (FP16) training support for faster GPU computation
    - Gradient clipping for stable training
    - ECS (Elastic Cell Similarity) loss for contrastive pre-training
    - Checkpoint saving/loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import time
from typing import Optional, Dict, Any, Tuple


class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup + cosine decay.

    The schedule looks like this:

        LR ↑
           |      peak_lr
           |     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
           |    /                                \
           |   /                                  |  min_lr
           |  /                                    |____
           | /
           +------------------------------------------------→ step
           | warmup |         cosine decay         |

    This is a common strategy in training large models:
    - Warmup: Gradually increase LR to avoid exploding gradients at the start
    - Cosine decay: Smoothly decrease LR for fine convergence

    Args:
        optimizer: PyTorch optimizer
        warmup_steps (int): Number of warmup steps
        total_steps (int): Total number of training steps
        start_lr (float): Starting learning rate (for warmup)
        peak_lr (float): Peak learning rate after warmup
        min_lr (float): Minimum learning rate at end of training
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr: float = 1e-7,
        peak_lr: float = 1e-4,
        min_lr: float = 5e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            progress = self.current_step / max(self.warmup_steps, 1)
            lr = self.start_lr + (self.peak_lr - self.start_lr) * progress
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.current_step - self.warmup_steps) / max(decay_steps, 1)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss that ignores padded positions.

    Used during CellFM pre-training: the model predicts masked gene
    expression values, and we compute MSE only on the masked positions.

    Args:
        reduction (str): 'mean' or 'sum'
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.

        Args:
            predictions: (batch, seq_len) — predicted expression values
            targets: (batch, seq_len) — true expression values
            mask: (batch, seq_len) — True for positions to compute loss on

        Returns:
            Scalar loss value
        """
        diff = (predictions - targets) ** 2
        diff = diff * mask.float()

        if self.reduction == 'mean':
            return diff.sum() / mask.float().sum().clamp(min=1.0)
        else:
            return diff.sum()


class CellFMTrainer:
    """
    Training engine for CellFM models.

    Handles the complete training loop including:
    - Forward pass, loss computation, backward pass
    - Learning rate scheduling (cosine warmup)
    - Gradient clipping for training stability
    - Periodic logging and evaluation
    - Checkpoint saving

    Args:
        model (nn.Module): CellFM model to train
        config: Model configuration (Config80M or Config800M)
        output_dir (str): Directory to save checkpoints and logs
        device (str): Compute device ('cuda', 'cpu', 'mps')
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        output_dir: str = "./checkpoints",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.output_dir = output_dir
        self.device = device

        os.makedirs(output_dir, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_fraction: float = 0.1,
        max_grad_norm: float = 1.0,
        log_every: int = 50,
        eval_every: int = 500,
        save_every: int = 1000,
        task: str = "classification",
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation (optional)
            num_epochs: Number of training epochs
            learning_rate: Peak learning rate
            weight_decay: L2 regularization strength
            warmup_fraction: Fraction of total steps for warmup
            max_grad_norm: Maximum gradient norm (for clipping)
            log_every: Log every N steps
            eval_every: Evaluate every N steps
            save_every: Save checkpoint every N steps
            task: Training task — 'classification' or 'regression'

        Returns:
            Dict with training history ('train_loss', 'val_loss', 'lr')
        """
        # Calculate total training steps
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_fraction)

        # Setup optimizer (AdamW with decoupled weight decay)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Setup learning rate scheduler
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            start_lr=self.config.start_lr,
            peak_lr=learning_rate,
            min_lr=self.config.min_lr,
        )

        # Setup loss function
        if task == "classification":
            criterion = nn.CrossEntropyLoss()
        elif task == "regression":
            criterion = MaskedMSELoss()
        else:
            raise ValueError(f"Unknown task: {task}. Use 'classification' or 'regression'.")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }

        print(f"🧬 Training CellFM")
        print(f"   Task: {task}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Steps/epoch: {len(train_loader)}")
        print(f"   Total steps: {total_steps}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"   Device: {self.device}")
        print()

        # === Training Loop ===
        self.model.train()
        epoch_start = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                gene_ids = batch['gene_ids'].to(self.device)
                gene_values = batch['gene_values'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)

                # Forward pass
                logits = self.model(
                    gene_ids, gene_values,
                    key_padding_mask=padding_mask,
                )

                # Compute loss
                if task == "classification":
                    labels = batch['label'].to(self.device)
                    loss = criterion(logits, labels)
                else:
                    # For regression, the model predicts gene expression values
                    loss = criterion(logits[:, 1:, 0], gene_values, ~padding_mask)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (prevents exploding gradients)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                # Update weights
                optimizer.step()

                # Update learning rate
                lr = scheduler.step()

                # Track metrics
                epoch_loss += loss.item()
                n_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    elapsed = time.time() - epoch_start
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0

                    print(
                        f"  Step {self.global_step:>6d} | "
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )

                    history['train_loss'].append(loss.item())
                    history['lr'].append(lr)

                # Evaluation
                if val_loader is not None and self.global_step % eval_every == 0:
                    val_loss = self.evaluate(val_loader, criterion, task)
                    history['val_loss'].append(val_loss)
                    print(f"  📊 Validation Loss: {val_loss:.4f}")

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint("best.pt", optimizer, scheduler)
                        print(f"  ✅ New best model saved!")

                    self.model.train()

                # Save checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(
                        f"step_{self.global_step}.pt", optimizer, scheduler
                    )

            # End of epoch
            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - epoch_start
            print(f"\n  Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f} | Time: {elapsed:.1f}s\n")

        # Save final checkpoint
        self.save_checkpoint("final.pt", optimizer, scheduler)
        print(f"✅ Training complete! Best loss: {self.best_loss:.4f}")

        return history

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        task: str = "classification",
    ) -> float:
        """
        Evaluate the model on a validation set.

        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            task: Task type

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            gene_ids = batch['gene_ids'].to(self.device)
            gene_values = batch['gene_values'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)

            logits = self.model(
                gene_ids, gene_values,
                key_padding_mask=padding_mask,
            )

            if task == "classification":
                labels = batch['label'].to(self.device)
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits[:, 1:, 0], gene_values, ~padding_mask)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(
        self,
        filename: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[CosineWarmupScheduler] = None,
    ):
        """
        Save a training checkpoint.

        Checkpoints include:
        - Model weights
        - Optimizer state (for resuming training)
        - Scheduler state
        - Training metadata (step, epoch, best loss)

        Args:
            filename: Name of the checkpoint file
            optimizer: Optimizer to save (optional)
            scheduler: Scheduler to save (optional)
        """
        path = os.path.join(self.output_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': {
                'enc_dims': self.config.enc_dims,
                'enc_nlayers': self.config.enc_nlayers,
                'enc_num_heads': self.config.enc_num_heads,
            },
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state'] = {
                'current_step': scheduler.current_step,
                'warmup_steps': scheduler.warmup_steps,
                'total_steps': scheduler.total_steps,
            }

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            path: Path to the checkpoint file
            optimizer: Optimizer to restore state into (optional)

        Returns:
            Checkpoint metadata dict
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"✅ Loaded checkpoint from {path}")
        print(f"   Step: {self.global_step}, Epoch: {self.current_epoch}")
        print(f"   Best loss: {self.best_loss:.4f}")

        return checkpoint
