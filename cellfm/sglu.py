"""
CellFM SGLU (SwiGLU) Feed-Forward Module
==========================================

LAYMAN EXPLANATION:
    After the retention layer figures out which genes relate to each other,
    the SGLU module processes this information to extract useful features.

    Think of it as a "smart filter" with two pathways:
        Path 1 ("Gate"): Decides WHAT is important (outputs values 0-1)
        Path 2 ("Content"): Learns the actual information
        Output: Gate × Content (only important information passes through)

    This is like a security checkpoint at an airport:
        - The gate scanner decides if each item is allowed through
        - Only approved items (important features) make it to the output

TECHNICAL DETAILS:
    SGLU = Simple Gated Linear Unit with SiLU (Swish) activation.

    The architecture follows SwiGLU (Shazeer, 2020):
        SGLU(x) = (W₁x ⊙ SiLU(W₂x)) · W₃

    Where:
        - W₁, W₂ project input to a larger hidden dimension (typically 4× or 2.67×)
        - SiLU(x) = x * sigmoid(x) — a smooth, non-linear activation
        - ⊙ is element-wise multiplication (the "gating")
        - W₃ projects back to the original dimension

    SwiGLU has been shown to outperform standard FFN (ReLU) in Transformers:
        - Used in LLaMA, PaLM, and now CellFM
        - Better gradient flow due to the gating mechanism
        - The SiLU activation provides smoother gradients than ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network — the non-linear processing layer of CellFM.

    After the retention layer captures gene-gene relationships, SGLU
    processes these representations through a gated mechanism:

        output = (Linear₁(x) ⊙ SiLU(Linear₂(x))) · Linear₃

    The gating mechanism (⊙) allows the network to selectively amplify
    important features and suppress irrelevant ones.

    Args:
        embed_dim (int): Input and output dimension (e.g., 512 or 1536)
        hidden_dim (int): Hidden dimension of the FFN (typically 2.67× embed_dim).
                         If None, defaults to int(8/3 * embed_dim).
        dropout (float): Dropout rate for regularization
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Default hidden dim: 8/3 × embed_dim (following SwiGLU convention)
        # This is ≈2.67×, which balances capacity with parameter count
        if hidden_dim is None:
            hidden_dim = int(8 / 3 * embed_dim)
            # Round to nearest multiple of 64 for GPU efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # === Gate pathway ===
        # Projects input to hidden dim, then applies SiLU activation.
        # This produces values in [0, ∞) that control the "gate."
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=False)

        # === Content pathway ===
        # Projects input to hidden dim — the actual content to be gated.
        self.w_up = nn.Linear(embed_dim, hidden_dim, bias=False)

        # === Output projection ===
        # Projects back from hidden dim to embed dim.
        self.w_down = nn.Linear(hidden_dim, embed_dim, bias=False)

        # === Regularization ===
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with Xavier uniform for stable training."""
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward processing.

        Args:
            x: (batch_size, seq_len, embed_dim) — input from retention layer

        Returns:
            output: (batch_size, seq_len, embed_dim) — processed features

        The computation:
            gate = SiLU(W_gate @ x)     # "What's important?"
            content = W_up @ x          # "What's the content?"
            hidden = gate ⊙ content     # "Filter content through gate"
            output = W_down @ hidden    # "Project back to embed_dim"
        """
        # Gate pathway: Learn what's important, apply SiLU activation
        # SiLU(x) = x * sigmoid(x) — smooth, non-linear
        gate = F.silu(self.w_gate(x))  # (batch, seq, hidden_dim)

        # Content pathway: Learn the actual information
        content = self.w_up(x)  # (batch, seq, hidden_dim)

        # Gating: Only let important content through
        hidden = gate * content  # (batch, seq, hidden_dim)

        # Project back to original dimension
        output = self.w_down(hidden)  # (batch, seq, embed_dim)

        # Apply dropout for regularization
        output = self.dropout(output)

        return output
