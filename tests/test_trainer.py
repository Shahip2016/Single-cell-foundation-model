"""
Unit Tests for CellFM Training Engine
========================================

Tests for the trainer, learning rate scheduler, and loss functions.

Run with: python -m pytest tests/test_trainer.py -v
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.config import Config80M
from cellfm.model import CellFM
from cellfm.trainer import CosineWarmupScheduler, MaskedMSELoss, CellFMTrainer
from cellfm.data import SyntheticCellDataset, collate_cells
from torch.utils.data import DataLoader


# === Small model for fast testing ===
def make_test_model(num_classes=5):
    cfg = Config80M()
    cfg.enc_dims = 64
    cfg.enc_nlayers = 2
    cfg.enc_num_heads = 4
    cfg.recompute = False
    return CellFM(n_genes=500, config=cfg, num_classes=num_classes), cfg


class TestCosineWarmupScheduler:
    """Test the learning rate scheduler."""

    def test_warmup_phase(self):
        """LR should increase during warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=100, total_steps=1000,
            start_lr=1e-7, peak_lr=1e-4, min_lr=1e-5,
        )

        lrs = []
        for _ in range(100):
            lr = scheduler.step()
            lrs.append(lr)

        # LR should be increasing during warmup
        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_peak_at_warmup_end(self):
        """LR should reach peak at end of warmup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=100, total_steps=1000,
            start_lr=1e-7, peak_lr=1e-4, min_lr=1e-5,
        )

        for _ in range(100):
            scheduler.step()

        lr = scheduler.get_lr()
        assert abs(lr - 1e-4) < 1e-6, f"LR should be peak (1e-4), got {lr}"

    def test_decay_phase(self):
        """LR should decrease during cosine decay."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=100, total_steps=1000,
            start_lr=1e-7, peak_lr=1e-4, min_lr=1e-5,
        )

        # Complete warmup
        for _ in range(100):
            scheduler.step()

        # Check decay
        lr_at_warmup_end = scheduler.get_lr()
        for _ in range(500):
            scheduler.step()
        lr_at_midpoint = scheduler.get_lr()

        assert lr_at_midpoint < lr_at_warmup_end, "LR should decrease during decay"

    def test_min_lr_reached(self):
        """LR should approach min_lr at end of training."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=100, total_steps=1000,
            start_lr=1e-7, peak_lr=1e-4, min_lr=1e-5,
        )

        for _ in range(1000):
            scheduler.step()

        lr = scheduler.get_lr()
        assert abs(lr - 1e-5) < 1e-6, f"LR should be min (1e-5), got {lr}"


class TestMaskedMSELoss:
    """Test the masked MSE loss function."""

    def test_basic_computation(self):
        """Masked MSE should compute correctly."""
        loss_fn = MaskedMSELoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, True, True]])

        loss = loss_fn(predictions, targets, mask)
        assert loss.item() == 0.0, "Loss should be 0 for perfect predictions"

    def test_masking_ignores_positions(self):
        """Masked positions should not contribute to loss."""
        loss_fn = MaskedMSELoss()
        predictions = torch.tensor([[1.0, 100.0, 3.0]])  # Position 1 is way off
        targets = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True]])  # Ignore position 1

        loss = loss_fn(predictions, targets, mask)
        assert loss.item() == 0.0, "Masked positions should be ignored"

    def test_nonzero_loss(self):
        """Loss should be positive for incorrect predictions."""
        loss_fn = MaskedMSELoss()
        predictions = torch.tensor([[1.0, 3.0, 5.0]])
        targets = torch.tensor([[2.0, 4.0, 6.0]])
        mask = torch.tensor([[True, True, True]])

        loss = loss_fn(predictions, targets, mask)
        assert loss.item() > 0.0


class TestCellFMTrainer:
    """Test the full training engine."""

    def test_train_one_epoch(self):
        """Should complete one epoch without errors."""
        model, cfg = make_test_model(num_classes=5)

        ds = SyntheticCellDataset(
            n_cells=32, n_genes=500, max_genes=64, n_classes=5
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        # Use temp directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CellFMTrainer(
                model=model, config=cfg,
                output_dir=tmpdir, device="cpu",
            )

            history = trainer.train(
                train_loader=loader,
                num_epochs=1,
                learning_rate=1e-3,
                log_every=2,
                save_every=1000,
                task="classification",
            )

            assert len(history['train_loss']) > 0
            assert all(l > 0 for l in history['train_loss'])

    def test_checkpoint_save_load(self):
        """Should save and load checkpoints correctly."""
        model, cfg = make_test_model(num_classes=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CellFMTrainer(
                model=model, config=cfg,
                output_dir=tmpdir, device="cpu",
            )

            # Save
            trainer.global_step = 42
            trainer.best_loss = 0.123
            trainer.save_checkpoint("test.pt")

            # Check file exists
            assert os.path.exists(os.path.join(tmpdir, "test.pt"))

            # Load into fresh model
            model2, cfg2 = make_test_model(num_classes=5)
            trainer2 = CellFMTrainer(
                model=model2, config=cfg2,
                output_dir=tmpdir, device="cpu",
            )
            trainer2.load_checkpoint(os.path.join(tmpdir, "test.pt"))

            assert trainer2.global_step == 42
            assert abs(trainer2.best_loss - 0.123) < 1e-6

    def test_evaluation(self):
        """Evaluation should return a valid loss."""
        model, cfg = make_test_model(num_classes=5)

        ds = SyntheticCellDataset(
            n_cells=16, n_genes=500, max_genes=64, n_classes=5
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        trainer = CellFMTrainer(
            model=model, config=cfg, device="cpu",
        )

        criterion = nn.CrossEntropyLoss()
        val_loss = trainer.evaluate(loader, criterion, task="classification")

        assert val_loss > 0, "Validation loss should be positive"
        assert not torch.isnan(torch.tensor(val_loss)), "Loss should not be NaN"

    def test_gradient_clipping(self):
        """Training with gradient clipping should not error."""
        model, cfg = make_test_model(num_classes=3)

        ds = SyntheticCellDataset(
            n_cells=16, n_genes=500, max_genes=64, n_classes=3
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CellFMTrainer(
                model=model, config=cfg,
                output_dir=tmpdir, device="cpu",
            )

            history = trainer.train(
                train_loader=loader,
                num_epochs=1,
                max_grad_norm=0.5,
                log_every=1,
                save_every=1000,
                task="classification",
            )

            assert len(history['train_loss']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
