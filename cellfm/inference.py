"""
CellFM Inference Utilities
=============================

LAYMAN EXPLANATION:
    Once the model is trained, we need tools to USE it. This file provides
    utilities for:

    1. LOADING a trained model from a checkpoint file
    2. PREDICTING cell types from gene expression data
    3. EXTRACTING cell embeddings (numerical "fingerprints" of cells)
    4. EXPORTING the model for deployment

    Think of it like this:
        Training = The chef learning recipes (expensive, done once)
        Inference = The chef cooking meals from the recipes (fast, done repeatedly)

TECHNICAL DETAILS:
    Key utilities:
    - load_pretrained(): Load CellFM from a checkpoint file
    - CellFMPredictor: High-level inference wrapper with batched prediction
    - extract_embeddings(): Extract cell embeddings for downstream analysis
    - export_model(): Export for deployment (TorchScript)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union


def load_pretrained(
    checkpoint_path: str,
    n_genes: int = 20000,
    config=None,
    num_classes: int = 0,
    device: str = "cpu",
) -> nn.Module:
    """
    Load a pre-trained CellFM model from a checkpoint.

    This is the primary entry point for using a trained model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        n_genes: Number of genes in the vocabulary
        config: Model configuration. If None, will attempt to infer from checkpoint.
        num_classes: Number of output classes (0 = encoder-only)
        device: Device to load the model onto

    Returns:
        Loaded CellFM model in eval mode

    Example:
        >>> from cellfm.inference import load_pretrained
        >>> from cellfm.config import Config80M
        >>> model = load_pretrained("checkpoints/best.pt", config=Config80M())
        >>> model.eval()
    """
    # Lazy import to avoid circular dependency
    from .model import CellFM
    from .config import Config80M, Config800M

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer config if not provided
    if config is None:
        ckpt_config = checkpoint.get('config', {})
        enc_nlayers = ckpt_config.get('enc_nlayers', 12)

        if enc_nlayers >= 40:
            config = Config800M()
        else:
            config = Config80M()

        # Override with saved values
        for key, value in ckpt_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create model
    model = CellFM(n_genes=n_genes, config=config, num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    step = checkpoint.get('global_step', '?')
    print(f"✅ Loaded pre-trained CellFM from {checkpoint_path} (step {step})")

    return model


class CellFMPredictor:
    """
    High-level inference wrapper for CellFM.

    Provides a clean API for making predictions on single-cell data,
    handling batching, padding, and device management internally.

    Args:
        model (nn.Module): Trained CellFM model
        device (str): Compute device
        max_genes (int): Maximum genes per cell

    Example:
        >>> predictor = CellFMPredictor(model, device="cuda")
        >>> predictions = predictor.predict_cell_types(gene_ids, gene_values)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        max_genes: int = 2048,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.max_genes = max_genes

    @torch.no_grad()
    def predict_cell_types(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict cell types from gene expression data.

        Args:
            gene_ids: (batch, seq_len) — gene indices
            gene_values: (batch, seq_len) — expression values
            key_padding_mask: (batch, seq_len) — padding mask

        Returns:
            predictions: (batch,) — predicted class indices
        """
        gene_ids = gene_ids.to(self.device)
        gene_values = gene_values.to(self.device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(self.device)

        logits = self.model(gene_ids, gene_values, key_padding_mask=key_padding_mask)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def predict_probabilities(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get class probabilities for each cell.

        Args:
            gene_ids: (batch, seq_len) — gene indices
            gene_values: (batch, seq_len) — expression values
            key_padding_mask: (batch, seq_len) — padding mask

        Returns:
            probabilities: (batch, num_classes) — softmax probabilities
        """
        gene_ids = gene_ids.to(self.device)
        gene_values = gene_values.to(self.device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(self.device)

        logits = self.model(gene_ids, gene_values, key_padding_mask=key_padding_mask)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def get_cell_embeddings(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract cell-level embeddings (CLS token representations).

        These embeddings are useful for:
        - Dimensionality reduction (UMAP, t-SNE)
        - Clustering analysis
        - Downstream classification with a simple head

        Args:
            gene_ids: (batch, seq_len) — gene indices
            gene_values: (batch, seq_len) — expression values
            key_padding_mask: (batch, seq_len) — padding mask

        Returns:
            embeddings: (batch, embed_dim) — cell-level embeddings
        """
        gene_ids = gene_ids.to(self.device)
        gene_values = gene_values.to(self.device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(self.device)

        return self.model(
            gene_ids, gene_values,
            key_padding_mask=key_padding_mask,
            return_cell_embedding=True,
        )


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract cell embeddings from an entire dataset.

    Iterates through the dataloader, extracts CLS token embeddings
    from each batch, and concatenates them into a single array.

    Useful for downstream analysis:
    - Visualization with UMAP/t-SNE
    - Clustering (Leiden, Louvain)
    - Batch integration
    - Trajectory inference

    Args:
        model: Trained CellFM model
        dataloader: DataLoader with cell data
        device: Compute device

    Returns:
        embeddings: (n_cells, embed_dim) numpy array

    Example:
        >>> embeddings = extract_embeddings(model, val_loader, device="cuda")
        >>> import umap
        >>> reducer = umap.UMAP()
        >>> coords_2d = reducer.fit_transform(embeddings)
    """
    model = model.to(device).eval()
    all_embeddings = []

    for batch in dataloader:
        gene_ids = batch['gene_ids'].to(device)
        gene_values = batch['gene_values'].to(device)
        padding_mask = batch['padding_mask'].to(device)

        emb = model(
            gene_ids, gene_values,
            key_padding_mask=padding_mask,
            return_cell_embedding=True,
        )

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def export_model(
    model: nn.Module,
    output_path: str,
    example_gene_ids: Optional[torch.Tensor] = None,
    example_gene_values: Optional[torch.Tensor] = None,
    seq_len: int = 2048,
):
    """
    Export CellFM model for deployment using TorchScript.

    TorchScript allows running the model without Python, which is
    useful for:
    - Production deployment (C++, Java)
    - Mobile/edge deployment
    - Serving with TorchServe

    Args:
        model: Trained CellFM model
        output_path: Path to save the exported model (.pt)
        example_gene_ids: Example input for tracing
        example_gene_values: Example input for tracing
        seq_len: Sequence length for example inputs
    """
    model.eval()

    # Create example inputs if not provided
    if example_gene_ids is None:
        example_gene_ids = torch.randint(1, 1000, (1, seq_len))
    if example_gene_values is None:
        example_gene_values = torch.randn(1, seq_len)

    # Trace the model
    traced = torch.jit.trace(
        model,
        (example_gene_ids, example_gene_values),
    )

    # Save
    traced.save(output_path)
    print(f"✅ Model exported to {output_path}")
