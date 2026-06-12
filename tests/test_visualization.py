"""
Unit Tests for CellFM Visualization Utilities
=================================================

Tests for plotting functions (dimensionality reduction, training curves,
confusion matrix heatmaps, gene importance charts).

Run with: python -m pytest tests/test_visualization.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.visualization import (
    reduce_dimensions,
    plot_embeddings,
    plot_training_history,
    plot_confusion_matrix,
    plot_gene_importance,
    plot_model_summary,
)

# Use non-interactive backend for testing (no display needed)
import matplotlib
matplotlib.use('Agg')


# === Test Configuration ===
N_SAMPLES = 50
EMBED_DIM = 32
N_CLASSES = 5


class TestReduceDimensions:
    """Test dimensionality reduction utilities."""

    def test_pca_output_shape(self):
        """PCA should reduce to (n_samples, 2)."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        coords = reduce_dimensions(embeddings, method="pca")
        assert coords.shape == (N_SAMPLES, 2)

    def test_tsne_output_shape(self):
        """t-SNE should reduce to (n_samples, 2)."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        coords = reduce_dimensions(embeddings, method="tsne")
        assert coords.shape == (N_SAMPLES, 2)

    def test_pca_deterministic(self):
        """PCA should be deterministic."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        coords1 = reduce_dimensions(embeddings, method="pca")
        coords2 = reduce_dimensions(embeddings, method="pca")
        np.testing.assert_array_almost_equal(coords1, coords2)

    def test_custom_n_components(self):
        """Should support arbitrary n_components."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        coords = reduce_dimensions(embeddings, method="pca", n_components=3)
        assert coords.shape == (N_SAMPLES, 3)

    def test_invalid_method(self):
        """Should raise for unknown methods."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_dimensions(embeddings, method="invalid")


class TestPlotEmbeddings:
    """Test embedding scatter plots."""

    def test_plot_without_labels(self):
        """Should create plot without labels."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        fig, coords = plot_embeddings(embeddings, method="pca")
        assert fig is not None
        assert coords.shape == (N_SAMPLES, 2)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_labels(self):
        """Should create colored plot with labels."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        labels = np.random.randint(0, N_CLASSES, N_SAMPLES)
        fig, coords = plot_embeddings(embeddings, labels=labels, method="pca")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_label_names(self):
        """Should create plot with custom label names."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        labels = np.random.randint(0, 3, N_SAMPLES)
        names = ["T-cell", "B-cell", "NK cell"]
        fig, _ = plot_embeddings(
            embeddings, labels=labels, label_names=names, method="pca"
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_save(self, tmp_path):
        """Should save plot to file."""
        embeddings = np.random.randn(N_SAMPLES, EMBED_DIM).astype(np.float32)
        save_path = str(tmp_path / "test_embeddings.png")
        fig, _ = plot_embeddings(embeddings, method="pca", save_path=save_path)
        assert os.path.exists(save_path)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotTrainingHistory:
    """Test training history plots."""

    def test_plot_with_all_curves(self):
        """Should plot train loss, val loss, and LR."""
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.7],
            'lr': [1e-5, 5e-5, 1e-4, 8e-5, 5e-5],
        }
        fig = plot_training_history(history)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_train_only(self):
        """Should work with just training loss."""
        history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [],
            'lr': [],
        }
        fig = plot_training_history(history)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotConfusionMatrix:
    """Test confusion matrix visualization."""

    def test_basic_plot(self):
        """Should create a confusion matrix heatmap."""
        cm = np.array([[10, 2, 1], [3, 15, 0], [1, 1, 12]])
        fig = plot_confusion_matrix(cm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_normalized_plot(self):
        """Should create a normalized confusion matrix."""
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, normalize=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_label_names(self):
        """Should work with custom label names."""
        cm = np.array([[10, 2], [3, 15]])
        fig = plot_confusion_matrix(cm, label_names=["T-cell", "B-cell"])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotGeneImportance:
    """Test gene importance bar charts."""

    def test_basic_plot(self):
        """Should create a gene importance bar chart."""
        gene_ids = np.arange(50)
        scores = np.random.rand(50)
        fig = plot_gene_importance(gene_ids, scores, top_k=10)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPlotModelSummary:
    """Test model parameter distribution charts."""

    def test_basic_plot(self):
        """Should create a parameter distribution donut chart."""
        counts = {
            'embedding': 5_000_000,
            'retention_total': 30_000_000,
            'sglu_total': 40_000_000,
            'norms_total': 500_000,
            'classifier': 1_000_000,
            'total': 76_500_000,
        }
        fig = plot_model_summary(counts)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
