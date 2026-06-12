"""
Unit Tests for CellFM Inference Utilities
============================================

Tests for model loading, prediction, embedding extraction, and export.

Run with: python -m pytest tests/test_inference.py -v
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.config import Config80M
from cellfm.model import CellFM
from cellfm.inference import (
    load_pretrained,
    CellFMPredictor,
    extract_embeddings,
)
from cellfm.data import SyntheticCellDataset, collate_cells
from torch.utils.data import DataLoader


# === Test Configuration ===
BATCH_SIZE = 4
SEQ_LEN = 64
N_GENES = 500
N_CLASSES = 5


def make_test_model(num_classes=0):
    cfg = Config80M()
    cfg.enc_dims = 64
    cfg.enc_nlayers = 2
    cfg.enc_num_heads = 4
    cfg.recompute = False
    return CellFM(n_genes=N_GENES, config=cfg, num_classes=num_classes), cfg


class TestLoadPretrained:
    """Test checkpoint loading."""

    def test_load_from_checkpoint(self):
        """Should load a model from a saved checkpoint."""
        model, cfg = make_test_model(num_classes=N_CLASSES)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            path = os.path.join(tmpdir, "test.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'global_step': 100,
                'config': {
                    'enc_dims': 64,
                    'enc_nlayers': 2,
                    'enc_num_heads': 4,
                },
            }, path)

            # Load it
            loaded = load_pretrained(
                path, n_genes=N_GENES, config=cfg,
                num_classes=N_CLASSES, device="cpu",
            )

            assert isinstance(loaded, CellFM)
            assert not loaded.training  # Should be in eval mode

    def test_loaded_model_produces_output(self):
        """Loaded model should produce valid output."""
        model, cfg = make_test_model(num_classes=N_CLASSES)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'global_step': 0,
                'config': {
                    'enc_dims': 64,
                    'enc_nlayers': 2,
                    'enc_num_heads': 4,
                },
            }, path)

            loaded = load_pretrained(
                path, n_genes=N_GENES, config=cfg,
                num_classes=N_CLASSES, device="cpu",
            )

            gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
            gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

            with torch.no_grad():
                output = loaded(gene_ids, gene_values)

            assert output.shape == (BATCH_SIZE, N_CLASSES)


class TestCellFMPredictor:
    """Test the high-level predictor API."""

    def test_predict_cell_types(self):
        """Should return class predictions."""
        model, _ = make_test_model(num_classes=N_CLASSES)
        predictor = CellFMPredictor(model, device="cpu", max_genes=SEQ_LEN)

        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        predictions = predictor.predict_cell_types(gene_ids, gene_values)

        assert predictions.shape == (BATCH_SIZE,)
        assert (predictions >= 0).all()
        assert (predictions < N_CLASSES).all()

    def test_predict_probabilities(self):
        """Should return valid probability distributions."""
        model, _ = make_test_model(num_classes=N_CLASSES)
        predictor = CellFMPredictor(model, device="cpu")

        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        probs = predictor.predict_probabilities(gene_ids, gene_values)

        assert probs.shape == (BATCH_SIZE, N_CLASSES)
        # Should sum to 1 (softmax output)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)
        # Should be non-negative
        assert (probs >= 0).all()

    def test_get_cell_embeddings(self):
        """Should return cell-level embeddings."""
        model, _ = make_test_model(num_classes=0)  # Encoder-only
        predictor = CellFMPredictor(model, device="cpu")

        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        embeddings = predictor.get_cell_embeddings(gene_ids, gene_values)

        assert embeddings.shape == (BATCH_SIZE, 64)

    def test_with_padding_mask(self):
        """Predictions should work with padding masks."""
        model, _ = make_test_model(num_classes=N_CLASSES)
        predictor = CellFMPredictor(model, device="cpu")

        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)
        mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, -10:] = True  # Last 10 positions padded

        predictions = predictor.predict_cell_types(
            gene_ids, gene_values, key_padding_mask=mask
        )
        assert predictions.shape == (BATCH_SIZE,)


class TestExtractEmbeddings:
    """Test batch embedding extraction."""

    def test_extract_from_dataloader(self):
        """Should extract embeddings for all cells in the dataset."""
        model, _ = make_test_model(num_classes=0)

        ds = SyntheticCellDataset(
            n_cells=32, n_genes=N_GENES, max_genes=SEQ_LEN
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        embeddings = extract_embeddings(model, loader, device="cpu")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (32, 64)

    def test_embeddings_are_finite(self):
        """All embedding values should be finite."""
        model, _ = make_test_model(num_classes=0)

        ds = SyntheticCellDataset(
            n_cells=16, n_genes=N_GENES, max_genes=SEQ_LEN
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        embeddings = extract_embeddings(model, loader, device="cpu")

        assert np.all(np.isfinite(embeddings))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
