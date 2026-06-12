"""
CellFM Command-Line Interface
================================

LAYMAN EXPLANATION:
    This file lets you run CellFM from the terminal (command line)
    without writing Python code. It's like having a remote control
    for the model.

    Examples:
        # Train a model
        python -m cellfm train --data my_cells.h5ad --epochs 10

        # Predict cell types
        python -m cellfm predict --checkpoint best.pt --data new_cells.h5ad

        # Extract embeddings for visualization
        python -m cellfm embed --checkpoint best.pt --data cells.h5ad --output embeddings.npy

        # Show model info
        python -m cellfm info --config 80M

TECHNICAL DETAILS:
    Uses Python's argparse for CLI parsing. Each subcommand maps to a
    function that orchestrates the corresponding pipeline.
"""

import argparse
import sys
import os
import time
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CellFM CLI."""
    parser = argparse.ArgumentParser(
        prog="cellfm",
        description="🧬 CellFM — A Single-Cell Foundation Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info --config 80M
  %(prog)s info --config 800M
  %(prog)s train --data cells.h5ad --label-col cell_type --epochs 10
  %(prog)s predict --checkpoint best.pt --data test.h5ad --label-col cell_type
  %(prog)s embed --checkpoint best.pt --data cells.h5ad --output embeddings.npy
        """,
    )

    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === info subcommand ===
    info_parser = subparsers.add_parser(
        "info",
        help="Show model configuration and parameter counts",
    )
    info_parser.add_argument(
        "--config", type=str, default="80M",
        choices=["80M", "800M"],
        help="Model configuration to display (default: 80M)",
    )

    # === train subcommand ===
    train_parser = subparsers.add_parser(
        "train",
        help="Train or fine-tune a CellFM model",
    )
    train_parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training data (.h5ad file)",
    )
    train_parser.add_argument(
        "--label-col", type=str, default=None,
        help="Column in adata.obs for labels (classification task)",
    )
    train_parser.add_argument(
        "--config", type=str, default="80M",
        choices=["80M", "800M"],
        help="Model configuration (default: 80M)",
    )
    train_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to pretrained checkpoint (for fine-tuning)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)",
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank (0 = full fine-tuning, >0 = LoRA, default: 0)",
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="./checkpoints",
        help="Output directory for checkpoints (default: ./checkpoints)",
    )
    train_parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )

    # === predict subcommand ===
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict cell types from gene expression data",
    )
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint",
    )
    predict_parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data file (.h5ad)",
    )
    predict_parser.add_argument(
        "--label-col", type=str, default=None,
        help="Column in adata.obs for true labels (for evaluation)",
    )
    predict_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for predictions (.npy or .csv)",
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32)",
    )
    predict_parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device (default: auto)",
    )

    # === embed subcommand ===
    embed_parser = subparsers.add_parser(
        "embed",
        help="Extract cell embeddings for downstream analysis",
    )
    embed_parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint",
    )
    embed_parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data file (.h5ad)",
    )
    embed_parser.add_argument(
        "--output", type=str, default="embeddings.npy",
        help="Output file for embeddings (.npy, default: embeddings.npy)",
    )
    embed_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    embed_parser.add_argument(
        "--device", type=str, default="auto",
        help="Compute device (default: auto)",
    )

    return parser


def get_device(device_str: str) -> str:
    """Resolve 'auto' device to the best available device."""
    if device_str != "auto":
        return device_str

    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def cmd_info(args):
    """Display model configuration and parameter counts."""
    from .config import Config80M, Config800M
    from .model import CellFM

    if args.config == "800M":
        config = Config800M()
        config_name = "800M (Full-Scale)"
    else:
        config = Config80M()
        config_name = "80M (Compact)"

    print("=" * 60)
    print(f"  🧬 CellFM — {config_name} Configuration")
    print("=" * 60)

    # Architecture
    print(f"\n  Architecture:")
    print(f"    Layers:         {config.enc_nlayers}")
    print(f"    Embed Dim:      {config.enc_dims}")
    print(f"    Heads:          {config.enc_num_heads}")
    print(f"    Head Dim:       {config.enc_dims // config.enc_num_heads}")
    print(f"    Max Genes:      {config.nonz_len}")

    # Training
    print(f"\n  Training Defaults:")
    print(f"    Start LR:       {config.start_lr}")
    print(f"    Peak LR:        {config.max_lr}")
    print(f"    Min LR:         {config.min_lr}")
    print(f"    Batch Size:     {config.use_bs}")
    print(f"    Dropout:        {config.dropout}")
    print(f"    Mask Ratio:     {config.mask_ratio}")
    print(f"    Grad Ckpt:      {config.recompute}")

    # Parameter count
    print(f"\n  Parameter Count:")
    model = CellFM(n_genes=20000, config=config)
    counts = model.count_parameters()
    for component, count in counts.items():
        print(f"    {component:<20} {count:>15,}")

    print()
    print("=" * 60)


def cmd_train(args):
    """Run the training pipeline."""
    import torch

    device = get_device(args.device)
    print(f"🧬 CellFM Training")
    print(f"   Data: {args.data}")
    print(f"   Device: {device}")
    print()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print(f"   Please provide a valid .h5ad file.")
        sys.exit(1)

    # Load data
    try:
        import scanpy as sc
    except ImportError:
        print("❌ scanpy is required for loading .h5ad files.")
        print("   Install with: pip install scanpy")
        sys.exit(1)

    print(f"📂 Loading data from {args.data}...")
    adata = sc.read_h5ad(args.data)
    print(f"   Loaded {adata.n_obs} cells × {adata.n_vars} genes")

    # Create config
    from .config import Config80M, Config800M
    config = Config800M() if args.config == "800M" else Config80M()

    # Determine number of classes
    num_classes = 0
    if args.label_col:
        unique_labels = adata.obs[args.label_col].nunique()
        num_classes = unique_labels
        print(f"   Classes: {num_classes} (from '{args.label_col}')")

    # Create model
    from .model import CellFM
    model = CellFM(n_genes=adata.n_vars, config=config, num_classes=num_classes)

    # Load checkpoint if provided
    if args.checkpoint:
        from .inference import load_pretrained
        model = load_pretrained(
            args.checkpoint, n_genes=adata.n_vars,
            config=config, num_classes=num_classes, device=device,
        )

    # Apply LoRA if requested
    if args.lora_rank > 0:
        from .lora import apply_lora_to_model
        model = apply_lora_to_model(model, rank=args.lora_rank)

    # Create dataloader
    from .data import create_dataloader
    train_loader = create_dataloader(
        adata, batch_size=args.batch_size,
        label_col=args.label_col, shuffle=True,
    )

    # Train
    from .trainer import CellFMTrainer
    trainer = CellFMTrainer(
        model=model, config=config,
        output_dir=args.output_dir, device=device,
    )

    task = "classification" if num_classes > 0 else "regression"
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        task=task,
    )

    print(f"\n✅ Training complete! Checkpoints saved to {args.output_dir}")


def cmd_predict(args):
    """Run prediction on new data."""
    import torch
    import numpy as np

    device = get_device(args.device)
    print(f"🧬 CellFM Prediction")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data: {args.data}")
    print(f"   Device: {device}")
    print()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    # Load data
    try:
        import scanpy as sc
    except ImportError:
        print("❌ scanpy required. Install with: pip install scanpy")
        sys.exit(1)

    adata = sc.read_h5ad(args.data)
    print(f"   Loaded {adata.n_obs} cells")

    # Load model
    from .inference import load_pretrained, CellFMPredictor
    from .data import create_dataloader, collate_cells

    model = load_pretrained(args.checkpoint, n_genes=adata.n_vars, device=device)

    # Create dataloader
    loader = create_dataloader(
        adata, batch_size=args.batch_size,
        label_col=args.label_col, shuffle=False,
    )

    # Predict
    predictor = CellFMPredictor(model, device=device)
    all_predictions = []

    for batch in loader:
        preds = predictor.predict_cell_types(
            batch['gene_ids'], batch['gene_values'],
            key_padding_mask=batch['padding_mask'],
        )
        all_predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    print(f"   Generated predictions for {len(predictions)} cells")

    # Evaluate if labels available
    if args.label_col and args.label_col in adata.obs.columns:
        from .metrics import accuracy
        true_labels = torch.tensor(
            [batch['label'] for batch in loader.dataset],
            dtype=torch.long,
        )
        pred_tensor = torch.tensor(predictions, dtype=torch.long)
        acc = accuracy(pred_tensor, true_labels[:len(pred_tensor)])
        print(f"   Accuracy: {acc:.4f}")

    # Save predictions
    if args.output:
        if args.output.endswith('.csv'):
            import pandas as pd
            pd.DataFrame({'prediction': predictions}).to_csv(args.output, index=False)
        else:
            np.save(args.output, predictions)
        print(f"   Saved predictions to {args.output}")

    print(f"\n✅ Prediction complete!")


def cmd_embed(args):
    """Extract cell embeddings."""
    import torch
    import numpy as np

    device = get_device(args.device)
    print(f"🧬 CellFM Embedding Extraction")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data: {args.data}")
    print(f"   Output: {args.output}")
    print(f"   Device: {device}")
    print()

    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    # Load data
    try:
        import scanpy as sc
    except ImportError:
        print("❌ scanpy required. Install with: pip install scanpy")
        sys.exit(1)

    adata = sc.read_h5ad(args.data)
    print(f"   Loaded {adata.n_obs} cells")

    # Load model
    from .inference import load_pretrained, extract_embeddings
    from .data import create_dataloader, collate_cells

    model = load_pretrained(args.checkpoint, n_genes=adata.n_vars, device=device)

    # Create dataloader (no shuffle for embeddings!)
    loader = create_dataloader(
        adata, batch_size=args.batch_size, shuffle=False,
    )

    # Extract
    print(f"   Extracting embeddings...")
    start = time.time()
    embeddings = extract_embeddings(model, loader, device=device)
    elapsed = time.time() - start

    print(f"   Shape: {embeddings.shape}")
    print(f"   Time: {elapsed:.1f}s")

    # Save
    np.save(args.output, embeddings)
    print(f"\n✅ Embeddings saved to {args.output}")


def main(argv=None):
    """Main entry point for the CellFM CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "info": cmd_info,
        "train": cmd_train,
        "predict": cmd_predict,
        "embed": cmd_embed,
    }

    cmd_func = commands.get(args.command)
    if cmd_func is None:
        parser.print_help()
        sys.exit(1)

    cmd_func(args)


if __name__ == "__main__":
    main()
