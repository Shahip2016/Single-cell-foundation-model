"""
CellFM Embedding Module
========================

LAYMAN EXPLANATION:
    Imagine you have a spreadsheet where each row is a cell and each column
    is a gene. The values are just numbers like 0.0, 3.2, 150.5, etc.

    The AI can't work directly with these raw numbers — it needs to convert
    them into rich, multi-dimensional representations called "embeddings."

    Think of it like this:
        Raw number:  3.2 (just a single number — not very informative)
        Embedding:   [0.12, -0.34, 0.78, 0.01, ...] (a 512-dimensional vector
                      that captures the MEANING of "gene X expressed at level 3.2")

    The embedding module does TWO things:
    1. Creates a unique "identity card" for each gene (gene_encoder)
    2. Encodes how much each gene is expressed (value_encoder)
    Then it combines them: "Gene TP53 expressed at level 3.2" → one rich vector.

TECHNICAL DETAILS:
    Following the CellFM paper, the embedding consists of:
    - gene_encoder: nn.Embedding(n_genes, enc_dims) — maps gene index to a vector
    - value_encoder: nn.Linear(1, enc_dims) — maps scalar expression value to a vector
    - The final embedding = gene_embedding + value_embedding (element-wise addition)

    Additionally, a special [CLS] token is prepended to represent the entire cell.
"""

import torch
import torch.nn as nn
import math


class GeneExpressionEmbedding(nn.Module):
    """
    Converts raw gene expression data into rich vector representations.

    Takes two inputs:
        - gene_ids: WHICH genes are expressed (indices into the gene vocabulary)
        - gene_values: HOW MUCH each gene is expressed (scalar values)

    Produces:
        - A sequence of embedding vectors, one per gene, plus a [CLS] token
          representing the entire cell.

    Args:
        n_genes (int): Total number of genes in the vocabulary (~20,000 for human)
        embed_dim (int): Dimensionality of the embedding vectors (e.g., 512 or 1536)
        dropout (float): Dropout rate for regularization (default: 0.1)
    """

    def __init__(self, n_genes: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # === Gene Identity Embedding ===
        # Each gene gets a unique vector representation.
        # Think of it as a "name badge" for each gene.
        # n_genes+1 because index 0 is reserved for padding.
        self.gene_encoder = nn.Embedding(n_genes + 1, embed_dim, padding_idx=0)

        # === Expression Value Embedding ===
        # Converts the scalar expression value (e.g., 3.2) into a vector.
        # This captures "how much" a gene is active.
        self.value_encoder = nn.Linear(1, embed_dim)

        # === CLS Token ===
        # A special learnable token prepended to the sequence.
        # After passing through the model, this token represents
        # the entire cell's "summary" — used for classification tasks.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # === Position Encoding (optional, for gene ordering) ===
        # Unlike language, genes don't have a natural "order."
        # CellFM sorts genes by expression level (highest first),
        # so positional encoding helps the model know the rank.
        self.pos_embed = nn.Parameter(torch.zeros(1, 2049, embed_dim))  # 2048 + 1 cls
        nn.init.normal_(self.pos_embed, std=0.02)

        # === Regularization ===
        self.dropout = nn.Dropout(dropout)

        # === Layer Normalization ===
        # Normalizes the embeddings to stabilize training.
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform (good for embeddings)."""
        nn.init.xavier_uniform_(self.gene_encoder.weight[1:])  # Skip padding idx
        nn.init.xavier_uniform_(self.value_encoder.weight)
        nn.init.zeros_(self.value_encoder.bias)

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_values: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert gene expression data into embeddings.

        Args:
            gene_ids: (batch_size, seq_len) — indices of expressed genes
            gene_values: (batch_size, seq_len) — expression levels
            mask: (batch_size, seq_len) — optional mask (1=real, 0=padding)

        Returns:
            embeddings: (batch_size, seq_len + 1, embed_dim) — +1 for CLS token

        Example:
            >>> embed = GeneExpressionEmbedding(n_genes=20000, embed_dim=512)
            >>> gene_ids = torch.randint(1, 20000, (4, 2048))
            >>> gene_values = torch.randn(4, 2048)
            >>> output = embed(gene_ids, gene_values)
            >>> output.shape  # torch.Size([4, 2049, 512])
        """
        batch_size, seq_len = gene_ids.shape

        # Step 1: Get gene identity embeddings
        # Shape: (batch_size, seq_len, embed_dim)
        gene_emb = self.gene_encoder(gene_ids)

        # Step 2: Get expression value embeddings
        # Unsqueeze to make values (batch, seq, 1) for the linear layer
        # Shape: (batch_size, seq_len, embed_dim)
        value_emb = self.value_encoder(gene_values.unsqueeze(-1))

        # Step 3: Combine gene identity + expression value
        # "Gene TP53 at expression 3.2" = identity_of_TP53 + encoding_of_3.2
        combined = gene_emb + value_emb

        # Step 4: Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        combined = torch.cat([cls_tokens, combined], dim=1)  # (batch, seq+1, dim)

        # Step 5: Add positional encoding
        combined = combined + self.pos_embed[:, :seq_len + 1, :]

        # Step 6: Normalize and apply dropout
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        return combined
