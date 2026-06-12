"""
Unit Tests for CellFM Data Pipeline
=======================================

Tests for the data loading, preprocessing, and DataLoader creation.

Run with: python -m pytest tests/test_data.py -v
"""

import torch
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.data import (
    CellDataset,
    SyntheticCellDataset,
    collate_cells,
)
from torch.utils.data import DataLoader


# === Test Configuration ===
N_CELLS = 50
N_GENES = 500
MAX_GENES = 64
N_CLASSES = 5


class TestSyntheticCellDataset:
    """Test the synthetic dataset for correctness."""

    def test_length(self):
        """Dataset length should match n_cells."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        assert len(ds) == N_CELLS

    def test_item_keys(self):
        """Each item should have gene_ids, gene_values, and padding_mask."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        item = ds[0]
        assert 'gene_ids' in item
        assert 'gene_values' in item
        assert 'padding_mask' in item

    def test_item_shapes(self):
        """All tensors should have shape (max_genes,)."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        item = ds[0]
        assert item['gene_ids'].shape == (MAX_GENES,)
        assert item['gene_values'].shape == (MAX_GENES,)
        assert item['padding_mask'].shape == (MAX_GENES,)

    def test_item_dtypes(self):
        """Check that dtypes are correct."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        item = ds[0]
        assert item['gene_ids'].dtype == torch.long
        assert item['gene_values'].dtype == torch.float32
        assert item['padding_mask'].dtype == torch.bool

    def test_labels_present(self):
        """When n_classes > 0, items should include labels."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES, n_classes=N_CLASSES
        )
        item = ds[0]
        assert 'label' in item
        assert 0 <= item['label'].item() < N_CLASSES

    def test_no_labels_when_zero_classes(self):
        """When n_classes = 0, items should NOT include labels."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES, n_classes=0
        )
        item = ds[0]
        assert 'label' not in item

    def test_gene_ids_positive(self):
        """Non-padded gene IDs should be positive (0 is padding)."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        item = ds[0]
        non_padded = item['gene_ids'][~item['padding_mask']]
        assert (non_padded > 0).all()

    def test_padding_mask_consistency(self):
        """Padding mask should be True exactly where gene_ids == 0."""
        ds = SyntheticCellDataset(n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES)
        item = ds[0]
        assert torch.all(item['padding_mask'] == (item['gene_ids'] == 0))


class TestCollation:
    """Test the custom collation function."""

    def test_collate_shapes(self):
        """Collated batch should have correct batched shapes."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES, n_classes=N_CLASSES
        )
        batch = [ds[i] for i in range(4)]
        collated = collate_cells(batch)

        assert collated['gene_ids'].shape == (4, MAX_GENES)
        assert collated['gene_values'].shape == (4, MAX_GENES)
        assert collated['padding_mask'].shape == (4, MAX_GENES)
        assert collated['label'].shape == (4,)

    def test_collate_without_labels(self):
        """Collation should work without labels."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES, n_classes=0
        )
        batch = [ds[i] for i in range(4)]
        collated = collate_cells(batch)

        assert 'label' not in collated


class TestDataLoader:
    """Test DataLoader integration."""

    def test_dataloader_iteration(self):
        """DataLoader should produce correct batch shapes."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES, n_classes=N_CLASSES
        )
        loader = DataLoader(ds, batch_size=8, collate_fn=collate_cells)

        batch = next(iter(loader))
        assert batch['gene_ids'].shape[0] == 8
        assert batch['gene_ids'].shape[1] == MAX_GENES

    def test_dataloader_full_iteration(self):
        """DataLoader should iterate through all data."""
        ds = SyntheticCellDataset(
            n_cells=N_CELLS, n_genes=N_GENES, max_genes=MAX_GENES
        )
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_cells)

        total_cells = 0
        for batch in loader:
            total_cells += batch['gene_ids'].shape[0]

        assert total_cells == N_CELLS


class TestCellDatasetWithMockAdata:
    """Test CellDataset with a mock AnnData-like object."""

    class MockAdata:
        """Minimal mock of AnnData for testing without scanpy dependency."""
        def __init__(self, n_cells, n_genes, label_col=None):
            self.X = np.random.rand(n_cells, n_genes).astype(np.float32)
            # Make it sparse-ish (like real scRNA-seq)
            mask = np.random.rand(n_cells, n_genes) < 0.8
            self.X[mask] = 0.0

            import types
            self.obs = types.SimpleNamespace()
            if label_col:
                labels = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=n_cells)
                self.obs.columns = [label_col]
                self.obs.__getitem__ = lambda _, key: types.SimpleNamespace(
                    values=labels
                )
            else:
                self.obs.columns = []

    def test_basic_loading(self):
        """CellDataset should load from mock AnnData."""
        adata = self.MockAdata(n_cells=N_CELLS, n_genes=N_GENES)
        ds = CellDataset(adata, max_genes=MAX_GENES, normalize=True)

        assert len(ds) == N_CELLS

    def test_item_format(self):
        """Items should have correct format."""
        adata = self.MockAdata(n_cells=N_CELLS, n_genes=N_GENES)
        ds = CellDataset(adata, max_genes=MAX_GENES, normalize=True)

        item = ds[0]
        assert item['gene_ids'].shape == (MAX_GENES,)
        assert item['gene_values'].shape == (MAX_GENES,)
        assert item['padding_mask'].shape == (MAX_GENES,)

    def test_gene_ranking(self):
        """Genes should be sorted by expression value (descending)."""
        adata = self.MockAdata(n_cells=N_CELLS, n_genes=N_GENES)
        ds = CellDataset(adata, max_genes=MAX_GENES, normalize=True)

        item = ds[0]
        values = item['gene_values']
        non_padded = values[~item['padding_mask']]

        # Values should be in descending order
        if len(non_padded) > 1:
            diffs = non_padded[1:] - non_padded[:-1]
            assert (diffs <= 0).all(), "Gene values should be sorted descending"

    def test_normalization(self):
        """Normalized values should be non-negative (log1p output)."""
        adata = self.MockAdata(n_cells=N_CELLS, n_genes=N_GENES)
        ds = CellDataset(adata, max_genes=MAX_GENES, normalize=True)

        item = ds[0]
        assert (item['gene_values'] >= 0).all(), "Log1p values should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
