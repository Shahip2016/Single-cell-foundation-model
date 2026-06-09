"""
CellFM Full Model Assembly
============================

LAYMAN EXPLANATION:
    This file puts ALL the pieces together into the complete CellFM model.
    Think of it as assembling a car from parts:

    1. Embedding (the fuel intake) — converts raw gene data into vectors
    2. ERetNet Layers (the engine) — processes gene relationships
       Each layer has:
       a. Multi-Scale Retention — "which genes relate to each other?"
       b. SGLU Feed-Forward — "what features are important?"
       c. Layer Normalization — "keep the numbers in a reasonable range"
    3. Output Head (the wheels) — produces the final prediction

    The model processes data in a pipeline:
        Raw genes → Embed → Layer1 → Layer2 → ... → LayerN → Output

    For the 800M model: N = 40 layers
    For the 80M model:  N = 12 layers

TECHNICAL DETAILS:
    The full CellFM model architecture:

    Input: (gene_ids, gene_values) — shape (batch, seq_len)
      ↓
    GeneExpressionEmbedding → (batch, seq_len+1, embed_dim)
      ↓
    N × ERetNetBlock:
      ├── LayerNorm
      ├── MultiScaleRetention (+ residual connection)
      ├── LayerNorm
      └── SGLU (+ residual connection)
      ↓
    LayerNorm → (batch, seq_len+1, embed_dim)
      ↓
    Classification/Regression Head → task-specific output

    Key design choices:
    - Pre-normalization (LayerNorm before each sub-layer)
    - Residual connections around both retention and SGLU
    - Optional gradient checkpointing for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .embedding import GeneExpressionEmbedding
from .retention import MultiScaleRetention
from .sglu import SGLU


class ERetNetBlock(nn.Module):
    """
    A single ERetNet block — one "layer" of CellFM's brain.

    Each block processes gene embeddings through two sub-layers:
    1. Multi-Scale Retention: discovers gene-gene relationships
    2. SGLU Feed-Forward: extracts and filters important features

    Both sub-layers use:
    - Pre-normalization (LayerNorm before the sub-layer)
    - Residual connections (output = input + sub_layer(input))

    The residual connection is crucial — it allows gradients to flow
    directly through the network, making it possible to train very
    deep models (40 layers!) without vanishing gradients.

    Args:
        embed_dim (int): Embedding dimension (e.g., 512 or 1536)
        num_heads (int): Number of retention heads (e.g., 16 or 48)
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # === Sub-layer 1: Multi-Scale Retention ===
        self.retention = MultiScaleRetention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.post_norm1 = nn.LayerNorm(embed_dim)

        # === Sub-layer 2: SGLU Feed-Forward ===
        self.sglu = SGLU(
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.post_norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input through one ERetNet block.

        The flow:
            x → Norm → Retention → +x → Norm → SGLU → +residual → output
                                    ↑ residual                ↑ residual

        Args:
            x: (batch_size, seq_len, embed_dim) — input embeddings
            key_padding_mask: (batch_size, seq_len) — mask for padding

        Returns:
            output: (batch_size, seq_len, embed_dim) — processed embeddings
        """
        # Sub-layer 1: Retention with residual connection
        residual = x
        x = self.post_norm1(x)               # Pre-normalization
        x = self.retention(x, key_padding_mask)  # Multi-scale retention
        x = x + residual                       # Residual connection

        # Sub-layer 2: SGLU with residual connection
        residual = x
        x = self.post_norm2(x)                # Pre-normalization
        x = self.sglu(x)                       # SwiGLU feed-forward
        x = x + residual                       # Residual connection

        return x


class CellFM(nn.Module):
    """
    CellFM — A Foundation Model for Single-Cell Transcriptomics.

    This is the complete model that processes single-cell gene expression
    data through an embedding layer and a stack of ERetNet blocks.

    The model can be used in two modes:
    1. **Encoder mode**: Returns hidden representations for each gene
       (useful for fine-tuning with a custom head)
    2. **Classification mode**: Returns class predictions
       (when num_classes > 0)

    Architecture:
        GeneExpressionEmbedding → N × ERetNetBlock → LayerNorm → [Head]

    Args:
        n_genes (int): Number of genes in the vocabulary (~20,000 for human)
        config: Configuration object (Config80M or Config800M)
        num_classes (int): Number of output classes (0 = encoder-only mode)
    """

    def __init__(self, n_genes: int, config, num_classes: int = 0):
        super().__init__()
        self.config = config
        self.n_genes = n_genes
        self.embed_dim = config.enc_dims
        self.num_layers = config.enc_nlayers
        self.num_classes = num_classes

        # === Embedding Layer ===
        # Converts raw gene expression data into vector representations
        self.embedding = GeneExpressionEmbedding(
            n_genes=n_genes,
            embed_dim=config.enc_dims,
            dropout=config.dropout,
        )

        # === ERetNet Backbone ===
        # Stack of N ERetNet blocks — this is the "brain" of the model
        self.layers = nn.ModuleList([
            ERetNetBlock(
                embed_dim=config.enc_dims,
                num_heads=config.enc_num_heads,
                dropout=config.enc_dropout,
            )
            for _ in range(config.enc_nlayers)
        ])

        # === Final Layer Normalization ===
        self.final_norm = nn.LayerNorm(config.enc_dims)

        # === Classification Head (optional) ===
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(config.enc_dims, config.enc_dims),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.enc_dims, num_classes),
            )
        else:
            self.classifier = None

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🧬 CellFM initialized:")
        print(f"   Layers: {config.enc_nlayers}")
        print(f"   Dims: {config.enc_dims}")
        print(f"   Heads: {config.enc_num_heads}")
        print(f"   Parameters: {total_params:,}")
        print(f"   ≈ {total_params / 1e6:.1f}M parameters")

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_cell_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through CellFM.

        Args:
            gene_ids: (batch, seq_len) — indices of expressed genes
            gene_values: (batch, seq_len) — expression levels
            key_padding_mask: (batch, seq_len) — True for padded positions
            return_cell_embedding: If True, return only the [CLS] token embedding

        Returns:
            If classifier is set and not return_cell_embedding:
                logits: (batch, num_classes) — class predictions
            If return_cell_embedding:
                cell_emb: (batch, embed_dim) — cell-level representation
            Else:
                hidden: (batch, seq_len+1, embed_dim) — all gene representations
        """
        # Step 1: Embed gene expression data
        # Output shape: (batch, seq_len+1, embed_dim) — +1 for CLS token
        x = self.embedding(gene_ids, gene_values)

        # Adjust padding mask for CLS token (CLS is never padded)
        if key_padding_mask is not None:
            # Prepend False for CLS token
            cls_mask = torch.zeros(
                key_padding_mask.shape[0], 1,
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        # Step 2: Pass through N ERetNet blocks
        for layer in self.layers:
            if self.config.recompute and self.training:
                # Gradient checkpointing: trade compute for memory
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, key_padding_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, key_padding_mask)

        # Step 3: Final normalization
        x = self.final_norm(x)

        # Step 4: Extract cell-level representation (CLS token = first position)
        if return_cell_embedding:
            return x[:, 0, :]  # (batch, embed_dim)

        # Step 5: Classification (if head is attached)
        if self.classifier is not None:
            cls_output = x[:, 0, :]  # Use CLS token for classification
            logits = self.classifier(cls_output)  # (batch, num_classes)
            return logits

        # Default: return all hidden states
        return x

    def get_gene_embeddings(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract per-gene embeddings (useful for gene function prediction).

        Returns embeddings for each gene position (excluding CLS token).

        Args:
            gene_ids: (batch, seq_len) — gene indices
            gene_values: (batch, seq_len) — expression values
            key_padding_mask: (batch, seq_len) — padding mask

        Returns:
            gene_embs: (batch, seq_len, embed_dim) — per-gene representations
        """
        hidden = self.forward(
            gene_ids, gene_values, key_padding_mask,
            return_cell_embedding=False,
        )
        # Skip CLS token (position 0)
        return hidden[:, 1:, :]

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component (useful for understanding model size).

        Returns:
            Dict with parameter counts for each component.
        """
        counts = {
            "embedding": sum(
                p.numel() for p in self.embedding.parameters()
            ),
            "retention_total": 0,
            "sglu_total": 0,
            "norms_total": 0,
        }

        for layer in self.layers:
            counts["retention_total"] += sum(
                p.numel() for p in layer.retention.parameters()
            )
            counts["sglu_total"] += sum(
                p.numel() for p in layer.sglu.parameters()
            )
            counts["norms_total"] += sum(
                p.numel() for p in layer.post_norm1.parameters()
            )
            counts["norms_total"] += sum(
                p.numel() for p in layer.post_norm2.parameters()
            )

        counts["final_norm"] = sum(
            p.numel() for p in self.final_norm.parameters()
        )

        if self.classifier is not None:
            counts["classifier"] = sum(
                p.numel() for p in self.classifier.parameters()
            )

        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts
