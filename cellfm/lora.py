"""
CellFM LoRA (Low-Rank Adaptation) Module
==========================================

LAYMAN EXPLANATION:
    Imagine you've hired an expert chef (the pre-trained model) who knows
    how to cook 10,000 dishes. Now you want them to specialize in Italian
    cuisine. You have two options:

    Option A: Send them back to cooking school for years (full fine-tuning)
        → Expensive, slow, modifies everything they know

    Option B: Give them a small Italian recipe book to reference (LoRA)
        → Cheap, fast, their core skills stay intact

    LoRA is Option B. Instead of modifying all 800 million parameters,
    we add tiny "adapter" matrices that only have ~1% of the parameters.
    The original model stays frozen (unchanged), and we only train
    the small adapters.

TECHNICAL DETAILS:
    For a pre-trained weight matrix W ∈ ℝ^{d×d}, LoRA approximates
    the update ΔW as a low-rank decomposition:

        W' = W + (α/r) · A · B

    Where:
        - W: Original frozen weight (not trained)
        - A ∈ ℝ^{d×r}: Down-projection (random init)
        - B ∈ ℝ^{r×d}: Up-projection (zero init)
        - r: Rank (typically 4-16, much smaller than d)
        - α: Scaling factor

    Benefits:
        - Trainable params: 2 × d × r (vs d × d for full fine-tuning)
        - For d=1536, r=8: 24,576 params vs 2,359,296 (99% reduction!)
        - Original model unchanged — can easily switch between tasks

    Reference: "LoRA: Low-Rank Adaptation of Large Language Models"
    (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    A linear layer augmented with LoRA (Low-Rank Adaptation).

    This wraps an existing nn.Linear layer and adds a low-rank update:
        output = W·x + (α/r) · A·B·x

    The original weight W is FROZEN (not trained). Only A and B are trained.

    Args:
        original_linear (nn.Linear): The pre-trained linear layer to adapt
        rank (int): Rank of the LoRA decomposition (typically 4-16)
        alpha (float): Scaling factor for the LoRA update
        dropout (float): Dropout applied to the LoRA path
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # The scaling factor for the LoRA update

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # === Freeze the original weight ===
        # The pre-trained knowledge stays untouched
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # === LoRA down-projection (A) ===
        # Projects from full dimension to low rank
        # Initialized with random Kaiming normal (for good gradient flow)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

        # === LoRA up-projection (B) ===
        # Projects from low rank back to full dimension
        # Initialized to ZERO — this ensures the model starts
        # identical to the original (ΔW = A·B = A·0 = 0)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # === Optional dropout on LoRA path ===
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Computes: output = original(x) + scaling * x @ A @ B

        Args:
            x: (..., in_features) — input tensor

        Returns:
            output: (..., out_features) — adapted output
        """
        # Original (frozen) output
        original_output = self.original_linear(x)

        # LoRA (trainable) update
        lora_input = self.lora_dropout(x)
        lora_output = lora_input @ self.lora_A @ self.lora_B  # Low-rank path

        # Combine: original + scaled LoRA update
        return original_output + self.scaling * lora_output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.original_linear.in_features}, "
            f"out_features={self.original_linear.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}"
        )


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA adapters to specific linear layers in a model.

    This function walks through the model and replaces targeted nn.Linear
    layers with LoRALinear wrappers. The original weights are frozen and
    only the small LoRA matrices are trainable.

    Args:
        model: The pre-trained model to adapt
        rank: LoRA rank (higher = more capacity but more params)
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to apply LoRA to.
                       Default: ["q_proj", "k_proj", "v_proj", "out_proj"]
        dropout: Dropout rate for LoRA path

    Returns:
        The modified model with LoRA adapters

    Example:
        >>> model = CellFM(n_genes=20000, config=Config80M())
        >>> model = apply_lora_to_model(model, rank=8, alpha=16)
        >>> trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        >>> total = sum(p.numel() for p in model.parameters())
        >>> print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    """
    if target_modules is None:
        # Default: apply LoRA to attention projections
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    lora_count = 0

    for name, module in model.named_modules():
        # Check if any part of the module name matches our targets
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, child_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    child_name = name

                # Replace with LoRA version
                lora_layer = LoRALinear(
                    original_linear=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(parent, child_name, lora_layer)
                lora_count += 1

    print(f"✅ Applied LoRA (rank={rank}, α={alpha}) to {lora_count} layers")
    return model


def get_lora_params(model: nn.Module):
    """
    Get only the LoRA parameters (for the optimizer).

    Use this to create an optimizer that only trains LoRA params:
        optimizer = AdamW(get_lora_params(model), lr=1e-4)

    Args:
        model: Model with LoRA adapters

    Returns:
        List of LoRA parameters (A and B matrices)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params
