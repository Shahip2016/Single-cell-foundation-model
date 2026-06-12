"""
CellFM Data Pipeline
======================

LAYMAN EXPLANATION:
    Before the AI can learn from cells, we need to prepare the data.
    This is like organizing a messy library:

    1. LOADING: Read the single-cell data files (h5ad format)
    2. NORMALIZING: Scale the numbers so they're comparable across experiments
    3. RANKING GENES: Sort genes by expression (most active first)
    4. BATCHING: Group cells together for efficient processing

    The h5ad file format (AnnData) is the standard in single-cell biology.
    It stores:
        - X: The gene expression matrix (cells × genes)
        - obs: Cell metadata (cell type, patient ID, etc.)
        - var: Gene metadata (gene name, chromosome, etc.)

TECHNICAL DETAILS:
    CellFM requires a specific input format:
        - gene_ids: Indices of non-zero genes, sorted by expression (descending)
        - gene_values: Corresponding expression values (log-normalized)
        - Sequences are truncated to nonz_len (default 2048) most-expressed genes

    Preprocessing pipeline:
        1. Load AnnData (.h5ad)
        2. Normalize per cell (total counts → 10,000)
        3. Log-transform: log1p(x)
        4. Rank genes by expression per cell
        5. Select top-k non-zero genes
        6. Pad shorter sequences to uniform length
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class CellDataset(Dataset):
    """
    PyTorch Dataset for single-cell gene expression data.

    Converts an AnnData object into a format suitable for CellFM:
        - Ranks genes by expression value (highest first)
        - Returns (gene_ids, gene_values, label) tuples

    This is designed to work with CellFM's embedding layer, which
    expects separate gene_ids and gene_values tensors.

    Args:
        adata: AnnData object (from scanpy) with expression matrix
        max_genes (int): Maximum number of genes to keep per cell (default: 2048)
        label_col (str): Column in adata.obs to use as labels (optional)
        normalize (bool): Whether to apply library-size normalization + log1p
    """

    def __init__(
        self,
        adata,
        max_genes: int = 2048,
        label_col: Optional[str] = None,
        normalize: bool = True,
    ):
        self.max_genes = max_genes
        self.label_col = label_col

        # Store the expression matrix as a dense numpy array
        # (AnnData can store sparse matrices — we need dense for indexing)
        if hasattr(adata.X, 'toarray'):
            self.expression_matrix = adata.X.toarray()
        else:
            self.expression_matrix = np.array(adata.X)

        # Optional normalization
        if normalize:
            self.expression_matrix = self._normalize(self.expression_matrix)

        # Extract labels if specified
        self.labels = None
        self.label_encoder = None
        if label_col is not None and label_col in adata.obs.columns:
            raw_labels = adata.obs[label_col].values
            if not np.issubdtype(raw_labels.dtype, np.number):
                # String labels → integer encoding
                unique_labels = sorted(set(raw_labels))
                self.label_encoder = {l: i for i, l in enumerate(unique_labels)}
                self.labels = np.array([self.label_encoder[l] for l in raw_labels])
            else:
                self.labels = raw_labels.astype(np.int64)

        self.n_cells = self.expression_matrix.shape[0]
        self.n_genes = self.expression_matrix.shape[1]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Library-size normalization + log1p transform.

        Steps:
            1. Divide each cell by its total expression (normalize to same scale)
            2. Multiply by 10,000 (common convention in scRNA-seq)
            3. log1p(x) = log(1 + x) — compresses the range of values

        Why log1p?
            Gene expression is highly skewed — a few genes might have
            expression values of 10,000+ while most are 0-10.
            log1p compresses this range so the model can learn effectively.
        """
        # Step 1: Library-size normalize
        cell_totals = X.sum(axis=1, keepdims=True)
        cell_totals = np.clip(cell_totals, a_min=1e-6, a_max=None)
        X_norm = X / cell_totals * 1e4

        # Step 2: Log-transform
        X_norm = np.log1p(X_norm)

        return X_norm.astype(np.float32)

    def _rank_genes(self, expression: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank genes by expression value (descending) and select top-k.

        This is a key step in CellFM's data preparation:
            - Most cells only express ~2000-4000 genes (rest are zero)
            - We sort by expression level so the most active genes come first
            - This gives the model a consistent ordering to learn from

        Args:
            expression: (n_genes,) — expression values for one cell

        Returns:
            gene_ids: (max_genes,) — indices of top genes
            gene_values: (max_genes,) — expression values of top genes
        """
        # Find non-zero genes
        nonzero_mask = expression > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = expression[nonzero_mask]

        # Sort by expression value (highest first)
        if len(nonzero_values) > 0:
            sort_order = np.argsort(-nonzero_values)
            sorted_indices = nonzero_indices[sort_order]
            sorted_values = nonzero_values[sort_order]
        else:
            sorted_indices = np.array([], dtype=np.int64)
            sorted_values = np.array([], dtype=np.float32)

        # Truncate to max_genes
        n_keep = min(len(sorted_indices), self.max_genes)
        gene_ids = np.zeros(self.max_genes, dtype=np.int64)
        gene_values = np.zeros(self.max_genes, dtype=np.float32)

        if n_keep > 0:
            # +1 because index 0 is reserved for padding in the embedding
            gene_ids[:n_keep] = sorted_indices[:n_keep] + 1
            gene_values[:n_keep] = sorted_values[:n_keep]

        return gene_ids, gene_values

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single cell's data.

        Returns:
            Dict with:
                - gene_ids: (max_genes,) — gene indices sorted by expression
                - gene_values: (max_genes,) — corresponding expression values
                - padding_mask: (max_genes,) — True for padded positions
                - label: scalar — class label (if label_col was specified)
        """
        expression = self.expression_matrix[idx]
        gene_ids, gene_values = self._rank_genes(expression)

        # Create padding mask (True = padded, False = real data)
        padding_mask = gene_ids == 0

        result = {
            'gene_ids': torch.from_numpy(gene_ids).long(),
            'gene_values': torch.from_numpy(gene_values).float(),
            'padding_mask': torch.from_numpy(padding_mask).bool(),
        }

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result


def collate_cells(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collation function for batching cells.

    Stacks individual cell tensors into batched tensors.
    Used with PyTorch DataLoader:
        loader = DataLoader(dataset, collate_fn=collate_cells)

    Args:
        batch: List of dicts from CellDataset.__getitem__

    Returns:
        Dict with batched tensors
    """
    result = {
        'gene_ids': torch.stack([b['gene_ids'] for b in batch]),
        'gene_values': torch.stack([b['gene_values'] for b in batch]),
        'padding_mask': torch.stack([b['padding_mask'] for b in batch]),
    }

    if 'label' in batch[0]:
        result['label'] = torch.stack([b['label'] for b in batch])

    return result


def create_dataloader(
    adata,
    batch_size: int = 16,
    max_genes: int = 2048,
    label_col: Optional[str] = None,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from an AnnData object.

    This is the main entry point for feeding data into CellFM.

    Args:
        adata: AnnData object with single-cell expression data
        batch_size: Number of cells per batch
        max_genes: Maximum genes to keep per cell
        label_col: Column in adata.obs for labels (for classification)
        normalize: Whether to apply library-size normalization
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)

    Returns:
        DataLoader ready for training/inference

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad("my_data.h5ad")
        >>> loader = create_dataloader(adata, batch_size=32, label_col="cell_type")
        >>> for batch in loader:
        ...     logits = model(batch['gene_ids'], batch['gene_values'])
    """
    dataset = CellDataset(
        adata=adata,
        max_genes=max_genes,
        label_col=label_col,
        normalize=normalize,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cells,
        drop_last=False,
    )


class SyntheticCellDataset(Dataset):
    """
    Synthetic dataset for testing and debugging.

    Generates fake single-cell data without requiring real h5ad files.
    Useful for:
        - Unit testing the training pipeline
        - Benchmarking model speed
        - Debugging without downloading real data

    Args:
        n_cells (int): Number of synthetic cells
        n_genes (int): Number of genes in the vocabulary
        max_genes (int): Maximum non-zero genes per cell
        n_classes (int): Number of classes for classification (0 = no labels)
        sparsity (float): Fraction of genes that are zero (0.0 to 1.0)
    """

    def __init__(
        self,
        n_cells: int = 1000,
        n_genes: int = 20000,
        max_genes: int = 2048,
        n_classes: int = 0,
        sparsity: float = 0.9,
    ):
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.max_genes = max_genes
        self.n_classes = n_classes
        self.sparsity = sparsity

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single synthetic cell."""
        # Determine number of expressed genes
        n_expressed = int(self.max_genes * (1 - self.sparsity * 0.5))
        n_expressed = max(10, min(n_expressed, self.max_genes))

        # Random gene indices (+1 to skip padding index 0)
        gene_ids = torch.zeros(self.max_genes, dtype=torch.long)
        expressed_ids = torch.randperm(self.n_genes)[:n_expressed] + 1
        gene_ids[:n_expressed] = expressed_ids

        # Random expression values (log-normal distribution)
        gene_values = torch.zeros(self.max_genes, dtype=torch.float32)
        gene_values[:n_expressed] = torch.randn(n_expressed).abs() * 2 + 0.1

        # Sort by expression (descending) — matching real data pipeline
        sort_idx = gene_values.argsort(descending=True)
        gene_ids = gene_ids[sort_idx]
        gene_values = gene_values[sort_idx]

        # Padding mask
        padding_mask = gene_ids == 0

        result = {
            'gene_ids': gene_ids,
            'gene_values': gene_values,
            'padding_mask': padding_mask,
        }

        if self.n_classes > 0:
            result['label'] = torch.randint(0, self.n_classes, (1,)).squeeze()

        return result
