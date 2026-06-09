"""
CellFM Model Configurations
============================

This file defines the hyperparameters for different model sizes.

LAYMAN EXPLANATION:
    Think of these configs as "blueprints" for different-sized brains.
    The 80M brain is like a small student — good for learning and experimenting.
    The 800M brain is like a professor — powerful but needs more resources.

TECHNICAL DETAILS:
    The key hyperparameters control:
    - enc_dims: The "width" of each layer (how much info each layer can hold)
    - enc_nlayers: The "depth" (how many layers of processing)
    - enc_num_heads: How many "perspectives" each layer uses to look at data
    - nonz_len: Maximum number of non-zero genes to process per cell

    The 800M config matches the paper's full model:
        800M params = 40 layers × 1536 dims × 48 heads

    The 80M config is the smaller model used for fine-tuning demos:
        80M params = 12 layers × 512 dims × 16 heads
"""


class Config800M:
    """
    Full-scale CellFM configuration (800 million parameters).

    This matches the model described in the paper:
        - 40 ERetNet layers
        - 1536 embedding dimensions
        - 48 attention heads (each head has 1536/48 = 32 dims)

    Requires: ~32GB GPU memory for training, ~16GB for inference.
    """
    # Learning rate schedule
    start_lr = 1e-7       # Initial LR (very small, for warmup)
    max_lr = 1e-6         # Peak LR after warmup
    min_lr = 5e-7         # Minimum LR (for cosine decay)

    # LoRA settings
    lora = 0              # LoRA rank (0 = disabled, >0 = use LoRA)
    alpha = 0             # LoRA scaling factor

    # Sequence lengths
    nonz_len = 2048       # Max number of non-zero genes per cell
    mask_len = 2048       # Number of genes to consider for masking
    filt_len = 200        # Filter length for gene selection

    # Model architecture
    dropout = 0.1         # General dropout rate
    enc_dims = 1536       # Hidden dimension of each layer
    enc_nlayers = 40      # Number of ERetNet layers (depth)
    enc_num_heads = 48    # Number of attention heads per layer
    enc_dropout = 0.1     # Dropout in encoder layers

    # Training parameters
    temp = 0.2            # Temperature for contrastive learning
    eps = 1e-2            # Epsilon for numerical stability
    recompute = True      # Gradient checkpointing (saves memory)
    sim = 0.8             # Similarity threshold
    add_zero = False      # Whether to add zero-expression genes

    # Masking for pre-training
    mask_ratio = 0.5      # Fraction of genes to mask during pre-training
    ecs_threshold = 0.8   # Elastic Cell Similarity threshold
    ecs = True            # Enable ECS loss
    pad_zero = True       # Pad shorter sequences with zeros

    # Batch size
    use_bs = 4            # Default batch size


class Config80M:
    """
    Smaller CellFM configuration (80 million parameters).

    This is the "student" version — much faster to train and fine-tune.
    Good for:
        - Experimentation and debugging
        - Running on consumer GPUs (8-16GB)
        - Quick prototyping of downstream tasks

    Architecture: 12 layers × 512 dims × 16 heads
    """
    # Learning rate schedule
    start_lr = 1e-5
    max_lr = 1e-4
    min_lr = 5e-5

    # LoRA settings
    lora = 0
    alpha = 0

    # Sequence lengths
    nonz_len = 2048
    mask_len = 2048
    filt_len = 200

    # Model architecture — SMALLER
    dropout = 0.1
    enc_dims = 512        # Narrower layers (vs 1536)
    enc_nlayers = 12      # Fewer layers (vs 40)
    enc_num_heads = 16    # Fewer heads (vs 48)
    enc_dropout = 0.1

    # Training parameters
    temp = 0.2
    eps = 1e-2
    recompute = False     # No need for gradient checkpointing with smaller model
    sim = 0.8
    add_zero = False

    # Masking
    mask_ratio = 0.5
    ecs_threshold = 0.8
    ecs = True
    pad_zero = True

    # Batch size — can be larger with smaller model
    use_bs = 16

    # Fine-tuning specific
    num_cls = 1           # Number of classification classes (set during fine-tuning)
    dataset = ""          # Dataset name
    feature_col = ""      # Column to use as labels
    ckpt_path = ""        # Path to pretrained checkpoint
    device = "cpu"        # Compute device
    epoch = 5             # Number of training epochs
