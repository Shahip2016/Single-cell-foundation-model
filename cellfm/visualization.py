"""
CellFM Visualization Utilities
=================================

LAYMAN EXPLANATION:
    Visualization is crucial for understanding what the model has learned.
    This module provides tools to create publication-quality plots:

    1. UMAP/t-SNE EMBEDDINGS: Scatter plots showing how cells cluster
       in 2D space — similar cell types should form distinct groups
    2. TRAINING CURVES: Loss and learning rate over time — is the model
       still learning, or has it converged?
    3. CONFUSION MATRIX HEATMAP: A color-coded grid showing classification
       performance per cell type
    4. ATTENTION HEATMAPS: Visualize which genes the model pays attention to

TECHNICAL DETAILS:
    All functions return matplotlib Figure objects for maximum flexibility.
    Users can either display them interactively or save to files.

    Dependencies: matplotlib, numpy (optional: seaborn for enhanced styling)
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple

# Lazy imports for matplotlib (not always installed)
_MPL_AVAILABLE = None


def _check_matplotlib():
    """Check if matplotlib is available, raise helpful error if not."""
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False

    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib\n"
            "Or install the full viz extras: pip install cellfm[viz]"
        )


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    method: str = "umap",
    title: str = "Cell Embeddings",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 5.0,
    alpha: float = 0.7,
    cmap: str = "tab20",
    save_path: Optional[str] = None,
):
    """
    Plot cell embeddings in 2D using UMAP or t-SNE dimensionality reduction.

    This creates a scatter plot where each point is a cell, colored by cell
    type (if labels are provided). Cells that the model considers similar
    will be plotted close together.

    Args:
        embeddings: (n_cells, embed_dim) — cell embeddings from the model
        labels: (n_cells,) — integer labels for coloring (optional)
        label_names: Human-readable names for each label class
        method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
        title: Plot title
        figsize: Figure size in inches
        point_size: Size of scatter points
        alpha: Transparency of points
        cmap: Matplotlib colormap name
        save_path: If provided, save the figure to this path

    Returns:
        (fig, coords_2d): Matplotlib Figure and 2D coordinates

    Example:
        >>> embeddings = extract_embeddings(model, dataloader)
        >>> fig, coords = plot_embeddings(embeddings, labels=cell_types)
        >>> fig.savefig("embeddings.png", dpi=300)
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    # Dimensionality reduction
    coords_2d = reduce_dimensions(embeddings, method=method)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        # Get colormap
        colormap = plt.colormaps.get_cmap(cmap).resampled(n_classes)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[label] if label_names else f"Class {label}"
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colormap(i)],
                label=name,
                s=point_size,
                alpha=alpha,
                edgecolors="none",
            )

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            markerscale=3,
            frameon=True,
            fontsize=9,
        )
    else:
        ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c="steelblue",
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

    method_name = {"umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}.get(method, method)
    ax.set_xlabel(f"{method_name} 1", fontsize=12)
    ax.set_ylabel(f"{method_name} 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, coords_2d


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2D for visualization.

    Supports three methods:
    - PCA: Fast, linear reduction (always available)
    - t-SNE: Non-linear, good for cluster visualization (requires sklearn)
    - UMAP: Non-linear, preserves global structure (requires umap-learn)

    Args:
        embeddings: (n_samples, n_dims) — high-dimensional embeddings
        method: 'pca', 'tsne', or 'umap'
        n_components: Number of output dimensions (usually 2)
        random_state: Random seed for reproducibility

    Returns:
        coords: (n_samples, n_components) — reduced coordinates
    """
    method = method.lower()

    if method == "pca":
        # PCA: Simple, fast, always available
        centered = embeddings - embeddings.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ Vt[:n_components].T
        return coords

    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError(
                "scikit-learn is required for t-SNE. "
                "Install with: pip install scikit-learn"
            )
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, embeddings.shape[0] - 1),
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            # Fall back to PCA with a warning
            print("⚠️  umap-learn not installed. Falling back to PCA.")
            print("   Install with: pip install umap-learn")
            return reduce_dimensions(embeddings, method="pca", n_components=n_components)
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")


def plot_training_history(
    history: Dict[str, list],
    title: str = "CellFM Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
):
    """
    Plot training loss, validation loss, and learning rate curves.

    Args:
        history: Dict from CellFMTrainer.train() with keys:
                 'train_loss', 'val_loss', 'lr'
        title: Overall title for the figure
        figsize: Figure size
        save_path: If provided, save the figure

    Returns:
        Matplotlib Figure

    Example:
        >>> history = trainer.train(train_loader, val_loader, num_epochs=10)
        >>> fig = plot_training_history(history)
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    n_plots = sum(1 for k in ['train_loss', 'val_loss', 'lr'] if history.get(k))
    n_plots = max(n_plots, 1)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot training loss
    if history.get('train_loss'):
        ax = axes[plot_idx]
        ax.plot(history['train_loss'], color='#2196F3', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_idx += 1

    # Plot validation loss
    if history.get('val_loss'):
        ax = axes[plot_idx]
        ax.plot(history['val_loss'], color='#FF5722', linewidth=1.5,
                marker='o', markersize=4, alpha=0.8)
        ax.set_xlabel('Evaluation Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_idx += 1

    # Plot learning rate
    if history.get('lr'):
        ax = axes[plot_idx]
        ax.plot(history['lr'], color='#4CAF50', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot a confusion matrix as a color-coded heatmap.

    Args:
        cm: (num_classes, num_classes) confusion matrix from metrics.confusion_matrix()
        label_names: Human-readable names for each class
        title: Plot title
        figsize: Figure size
        cmap: Colormap for the heatmap
        normalize: If True, normalize rows to show percentages
        save_path: If provided, save the figure

    Returns:
        Matplotlib Figure

    Example:
        >>> from cellfm.metrics import confusion_matrix
        >>> cm = confusion_matrix(predictions, targets)
        >>> fig = plot_confusion_matrix(cm, label_names=["T-cell", "B-cell", "NK"])
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    n_classes = cm.shape[0]
    if label_names is None:
        label_names = [f"Class {i}" for i in range(n_classes)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)  # Avoid division by zero
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_display[i, j]
            text = f"{value:{fmt}}" if isinstance(value, float) else f"{value}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=10,
            )

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(label_names, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_gene_importance(
    gene_ids: np.ndarray,
    importance_scores: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Top Gene Importance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot a horizontal bar chart of the most important genes.

    Useful for understanding which genes the model relies on most
    for classification or prediction tasks.

    Args:
        gene_ids: (N,) gene indices
        importance_scores: (N,) importance scores (e.g., attention weights)
        gene_names: Optional mapping from gene index to name
        top_k: Number of top genes to display
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure

    Returns:
        Matplotlib Figure
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    # Get top-k genes by importance
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    top_ids = gene_ids[top_indices]
    top_scores = importance_scores[top_indices]

    # Generate labels
    if gene_names is not None:
        labels = [gene_names[gid] if gid < len(gene_names) else f"Gene {gid}"
                  for gid in top_ids]
    else:
        labels = [f"Gene {gid}" for gid in top_ids]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_scores)))

    bars = ax.barh(
        range(len(top_scores) - 1, -1, -1),
        top_scores,
        color=colors,
        edgecolor="none",
        height=0.7,
    )

    ax.set_yticks(range(len(top_scores) - 1, -1, -1))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_model_summary(
    param_counts: Dict[str, int],
    title: str = "CellFM Parameter Distribution",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    Plot a pie chart showing parameter distribution across model components.

    Args:
        param_counts: Dict from model.count_parameters()
        title: Plot title
        figsize: Figure size
        save_path: If provided, save the figure

    Returns:
        Matplotlib Figure
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    # Filter out 'total' and zero entries
    components = {
        k: v for k, v in param_counts.items()
        if k != 'total' and v > 0
    }

    labels = list(components.keys())
    sizes = list(components.values())
    total = sum(sizes)

    # Format labels with percentages
    formatted_labels = [
        f"{l}\n({v/1e6:.1f}M, {100*v/total:.1f}%)"
        for l, v in zip(labels, sizes)
    ]

    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=formatted_labels,
        colors=colors,
        autopct='',
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 10},
    )

    # Draw center circle for donut chart effect
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)

    # Center text with total
    ax.text(
        0, 0,
        f"Total\n{total/1e6:.1f}M",
        ha='center', va='center',
        fontsize=14, fontweight='bold',
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
