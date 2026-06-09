"""
CellFM Multi-Scale Retention (RetNet)
======================================

LAYMAN EXPLANATION:
    In a cell, genes don't work in isolation — they interact with each other.
    Gene A might activate Gene B, which then suppresses Gene C. Understanding
    these relationships is crucial for understanding cell behavior.

    The "Retention" mechanism is how CellFM discovers these gene-gene relationships.
    It's similar to "Attention" in ChatGPT, but more efficient.

    ATTENTION (Transformer):
        "Compare every gene with every other gene" → N² comparisons
        For 2048 genes: 2048² = 4,194,304 comparisons! 💀

    RETENTION (RetNet):
        "Remember a running summary as you go" → N comparisons
        For 2048 genes: 2048 operations! 🚀

    The "Multi-Scale" part means we use multiple "perspectives" (heads),
    each with a different decay rate. Fast-decaying heads focus on nearby
    genes, while slow-decaying heads capture long-range relationships.

TECHNICAL DETAILS:
    RetNet replaces softmax attention with a retention mechanism:

    Standard Attention:
        Attention(Q, K, V) = softmax(QK^T / √d) V

    Retention (parallel form):
        Retention(Q, K, V) = (QK^T ⊙ D) V

    where D is a causal decay matrix:
        D[n,m] = γ^(n-m) if n ≥ m, else 0

    γ (gamma) is the decay rate, set differently for each head using:
        γ_h = 1 - 2^(-5 - h * range/num_heads)

    This creates a spectrum from fast decay (short-range) to slow decay
    (long-range), allowing the model to capture patterns at multiple scales.

    Reference: "Retentive Network: A Successor to Transformer for Large
    Language Models" (Sun et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiScaleRetention(nn.Module):
    """
    Multi-Scale Retention layer — the core attention mechanism of CellFM.

    Instead of computing full quadratic attention like Transformers,
    RetNet uses an exponential decay to weight interactions between genes.
    Genes that are "nearby" in the input sequence get stronger connections.

    Each head uses a different decay rate (γ), creating a "multi-scale"
    view of gene-gene relationships:
        - High γ (close to 1): captures LONG-range dependencies
        - Low γ (close to 0): captures SHORT-range, local patterns

    Args:
        embed_dim (int): Total embedding dimension (e.g., 512 or 1536)
        num_heads (int): Number of retention heads (e.g., 16 or 48)
        dropout (float): Dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # e.g., 1536/48 = 32

        # === Q, K, V projections ===
        # Just like in Transformers, we project the input into
        # Query, Key, and Value matrices.
        # Q = "What am I looking for?"
        # K = "What do I contain?"
        # V = "What information should I pass along?"
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # === Output projection ===
        # Combines multi-head outputs back into a single representation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # === Group normalization per head ===
        # Normalizes within each head independently
        self.group_norm = nn.GroupNorm(num_heads, embed_dim)

        # === Decay rates (γ) for each head ===
        # These are FIXED (not learned) — each head gets a different decay
        # creating a spectrum from short-range to long-range focus.
        self._init_decay_rates()

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_decay_rates(self):
        """
        Initialize per-head decay rates γ following the RetNet paper.

        Creates a spectrum of decay rates:
            γ_0 ≈ 0.97 (slow decay → long-range focus)
            γ_1 ≈ 0.96
            ...
            γ_N ≈ 0.85 (fast decay → short-range focus)

        The formula: γ_h = 1 - 2^(-5 - h * range / num_heads)
        where range is chosen to give a good spread.
        """
        decay_range = 8  # Controls the spread of decay rates
        gammas = []
        for h in range(self.num_heads):
            gamma = 1 - 2 ** (-5 - h * decay_range / self.num_heads)
            gammas.append(gamma)

        # Register as a buffer (not a parameter — won't be trained)
        self.register_buffer(
            "gammas", torch.tensor(gammas, dtype=torch.float32)
        )

    def _init_weights(self):
        """Initialize projection weights with Xavier initialization."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _build_decay_matrix(self, seq_len: int) -> torch.Tensor:
        """
        Build the causal decay matrix D.

        D[n, m] = γ^(n-m) if n >= m, else 0

        This is the key difference from Transformer attention:
        instead of softmax, we use exponential decay.

        Args:
            seq_len: Length of the input sequence

        Returns:
            D: (num_heads, seq_len, seq_len) decay matrix
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.gammas.device)

        # Compute n - m for all pairs
        # Shape: (seq_len, seq_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Create causal mask: only allow n >= m (can't look into the future)
        causal_mask = (distances >= 0).float()

        # Compute decay: γ^(n-m) for each head
        # gammas: (num_heads,) → (num_heads, 1, 1)
        gammas = self.gammas.view(-1, 1, 1)

        # D[h, n, m] = gamma_h ^ max(n-m, 0) * causal_mask
        # Shape: (num_heads, seq_len, seq_len)
        D = gammas ** distances.abs().unsqueeze(0).float() * causal_mask.unsqueeze(0)

        return D

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-scale retention to the input.

        The parallel form of retention:
            Retention(Q, K, V) = (QK^T ⊙ D) V

        where D is the per-head decay matrix.

        Args:
            x: (batch_size, seq_len, embed_dim) — input embeddings
            key_padding_mask: (batch_size, seq_len) — True for padded positions

        Returns:
            output: (batch_size, seq_len, embed_dim) — retained representations
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Step 2: Reshape for multi-head: (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Build decay matrix D
        # Shape: (num_heads, seq_len, seq_len)
        D = self._build_decay_matrix(seq_len)

        # Step 4: Compute retention scores
        # QK^T: (batch, heads, seq, seq) — "how much does gene_i relate to gene_j?"
        retention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 5: Apply decay mask
        # This is the key innovation: instead of softmax, use element-wise
        # multiplication with the decay matrix
        retention = retention * D.unsqueeze(0)  # (batch, heads, seq, seq)

        # Step 6: Apply padding mask if provided
        if key_padding_mask is not None:
            # Expand mask: (batch, 1, 1, seq) — mask out padded keys
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            retention = retention.masked_fill(mask, 0.0)

        # Step 7: Normalize retention weights
        # Use sum normalization instead of softmax
        retention_sum = retention.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        retention = retention / retention_sum

        # Step 8: Apply retention to values
        # output = (retention_weights) @ V
        output = torch.matmul(retention, V)  # (batch, heads, seq, head_dim)

        # Step 9: Reshape back: (batch, seq, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Step 10: Group normalization across heads
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)

        # Step 11: Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        return output
