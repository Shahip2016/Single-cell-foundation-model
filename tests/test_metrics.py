"""
Unit Tests for CellFM Evaluation Metrics
==========================================

Tests for accuracy, precision/recall/F1, confusion matrix, and the evaluator.

Run with: python -m pytest tests/test_metrics.py -v
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.metrics import (
    accuracy,
    precision_recall_f1,
    confusion_matrix,
    CellFMEvaluator,
)
from cellfm.config import Config80M
from cellfm.model import CellFM
from cellfm.data import SyntheticCellDataset, collate_cells
from torch.utils.data import DataLoader


# === Test Configuration ===
N_SAMPLES = 100
N_CLASSES = 5
N_GENES = 500
SEQ_LEN = 64


class TestAccuracy:
    """Test accuracy computation."""

    def test_perfect_accuracy(self):
        """100% accuracy when all predictions match targets."""
        targets = torch.tensor([0, 1, 2, 3, 4])
        preds = torch.tensor([0, 1, 2, 3, 4])
        assert accuracy(preds, targets) == 1.0

    def test_zero_accuracy(self):
        """0% accuracy when no predictions match."""
        targets = torch.tensor([0, 0, 0, 0, 0])
        preds = torch.tensor([1, 1, 1, 1, 1])
        assert accuracy(preds, targets) == 0.0

    def test_partial_accuracy(self):
        """50% accuracy with half correct."""
        targets = torch.tensor([0, 1, 2, 3])
        preds = torch.tensor([0, 1, 0, 0])
        assert accuracy(preds, targets) == 0.5

    def test_accuracy_from_logits(self):
        """Accuracy should work with logits (N, C) input."""
        targets = torch.tensor([0, 1, 2])
        # Logits where argmax gives correct predictions
        logits = torch.tensor([
            [10.0, 0.0, 0.0],  # Predicts 0 ✓
            [0.0, 10.0, 0.0],  # Predicts 1 ✓
            [0.0, 0.0, 10.0],  # Predicts 2 ✓
        ])
        assert accuracy(logits, targets) == 1.0

    def test_top_k_accuracy(self):
        """Top-k accuracy should be >= top-1 accuracy."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)

        acc_1 = accuracy(logits, targets, top_k=1)
        acc_3 = accuracy(logits, targets, top_k=3)

        assert acc_3 >= acc_1, "Top-3 accuracy should be >= top-1"


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 computation."""

    def test_perfect_classification(self):
        """Perfect predictions should give precision=recall=f1=1.0."""
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        metrics = precision_recall_f1(preds, targets, num_classes=3)

        assert abs(metrics['precision'] - 1.0) < 1e-6
        assert abs(metrics['recall'] - 1.0) < 1e-6
        assert abs(metrics['f1'] - 1.0) < 1e-6

    def test_returns_per_class(self):
        """Should return per-class breakdowns."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)
        metrics = precision_recall_f1(logits, targets, num_classes=N_CLASSES)

        assert len(metrics['per_class_precision']) == N_CLASSES
        assert len(metrics['per_class_recall']) == N_CLASSES
        assert len(metrics['per_class_f1']) == N_CLASSES
        assert len(metrics['per_class_support']) == N_CLASSES

    def test_precision_recall_bounds(self):
        """All metrics should be in [0, 1]."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)
        metrics = precision_recall_f1(logits, targets, num_classes=N_CLASSES)

        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1'] <= 1.0

    def test_weighted_average(self):
        """Weighted average should work."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)
        metrics = precision_recall_f1(logits, targets, average="weighted")

        assert 0.0 <= metrics['f1'] <= 1.0


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_shape(self):
        """Confusion matrix should be (num_classes, num_classes)."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)
        cm = confusion_matrix(logits, targets, num_classes=N_CLASSES)

        assert cm.shape == (N_CLASSES, N_CLASSES)

    def test_sum_equals_n_samples(self):
        """Sum of confusion matrix should equal number of samples."""
        targets = torch.randint(0, N_CLASSES, (N_SAMPLES,))
        logits = torch.randn(N_SAMPLES, N_CLASSES)
        cm = confusion_matrix(logits, targets, num_classes=N_CLASSES)

        assert cm.sum() == N_SAMPLES

    def test_perfect_is_diagonal(self):
        """Perfect predictions should produce a diagonal matrix."""
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        cm = confusion_matrix(preds, targets, num_classes=3)

        # Off-diagonal elements should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert cm[i, j] == 0, f"Off-diagonal cm[{i},{j}] should be 0"

    def test_diagonal_sum_equals_correct(self):
        """Diagonal sum should equal number of correct predictions."""
        targets = torch.tensor([0, 1, 2, 0, 1])
        preds = torch.tensor([0, 1, 0, 0, 2])  # 3 correct
        cm = confusion_matrix(preds, targets, num_classes=3)

        assert np.trace(cm) == 3


class TestCellFMEvaluator:
    """Test the high-level evaluator."""

    def test_evaluate_returns_metrics(self):
        """Evaluator should return comprehensive metrics dict."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=N_CLASSES)
        evaluator = CellFMEvaluator(model, device="cpu")

        ds = SyntheticCellDataset(
            n_cells=32, n_genes=N_GENES, max_genes=SEQ_LEN, n_classes=N_CLASSES
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        results = evaluator.evaluate(loader)

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'confusion_matrix' in results
        assert results['n_samples'] == 32

    def test_evaluate_metrics_bounds(self):
        """All metrics should be within valid bounds."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=N_CLASSES)
        evaluator = CellFMEvaluator(model, device="cpu")

        ds = SyntheticCellDataset(
            n_cells=32, n_genes=N_GENES, max_genes=SEQ_LEN, n_classes=N_CLASSES
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        results = evaluator.evaluate(loader)

        assert 0.0 <= results['accuracy'] <= 1.0
        assert 0.0 <= results['f1'] <= 1.0

    def test_print_report(self, capsys):
        """print_report should produce output without errors."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=N_CLASSES)
        evaluator = CellFMEvaluator(model, device="cpu")

        ds = SyntheticCellDataset(
            n_cells=16, n_genes=N_GENES, max_genes=SEQ_LEN, n_classes=N_CLASSES
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        results = evaluator.evaluate(loader)
        evaluator.print_report(results)  # Should not raise

        captured = capsys.readouterr()
        assert "Evaluation Report" in captured.out
        assert "Accuracy" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
