"""
CellFM Weight Conversion Utilities
=====================================

LAYMAN EXPLANATION:
    The original CellFM model was built with Huawei's MindSpore framework,
    which uses a different format for storing model weights than PyTorch.

    This module converts MindSpore checkpoints to PyTorch format, so you
    can use the official pre-trained weights with our PyTorch implementation.

    Think of it like converting a Word document to Google Docs format —
    the content is the same, but the file format is different.

TECHNICAL DETAILS:
    Key conversions:
    - MindSpore `.ckpt` files → PyTorch `.pt` checkpoint dicts
    - Parameter name mapping between MindSpore and PyTorch conventions
    - Shape validation to catch architecture mismatches
    - HuggingFace model download integration

    Usage:
        python -m cellfm.convert --input official.ckpt --output converted.pt
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List


# === Name Mapping: MindSpore → PyTorch ===
# MindSpore uses slightly different naming conventions.
# This mapping translates between the two.

MINDSPORE_TO_PYTORCH_MAP = {
    # Embedding layer
    "gene_encoder.embedding_table": "embedding.gene_encoder.weight",
    "value_encoder.weight": "embedding.value_encoder.weight",
    "value_encoder.bias": "embedding.value_encoder.bias",
    "cls_token": "embedding.cls_token",
    "pos_embed": "embedding.pos_embed",
    "emb_layer_norm.gamma": "embedding.layer_norm.weight",
    "emb_layer_norm.beta": "embedding.layer_norm.bias",
    # Final norm
    "final_norm.gamma": "final_norm.weight",
    "final_norm.beta": "final_norm.bias",
}


def _build_layer_mapping(n_layers: int) -> Dict[str, str]:
    """
    Build parameter name mapping for all encoder layers.

    MindSpore and PyTorch name layers differently:
        MindSpore: encoder.blocks.{i}.retention.q_proj.weight
        PyTorch:   layers.{i}.retention.q_proj.weight

    Args:
        n_layers: Number of encoder layers

    Returns:
        Dict mapping MindSpore names → PyTorch names
    """
    mapping = dict(MINDSPORE_TO_PYTORCH_MAP)

    for i in range(n_layers):
        ms_prefix = f"encoder.blocks.{i}"
        pt_prefix = f"layers.{i}"

        # Retention sub-layer
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mapping[f"{ms_prefix}.retention.{proj}.weight"] = (
                f"{pt_prefix}.retention.{proj}.weight"
            )

        # Group norm
        mapping[f"{ms_prefix}.retention.group_norm.gamma"] = (
            f"{pt_prefix}.retention.group_norm.weight"
        )
        mapping[f"{ms_prefix}.retention.group_norm.beta"] = (
            f"{pt_prefix}.retention.group_norm.bias"
        )

        # SGLU (SwiGLU) sub-layer
        for proj in ["w_gate", "w_up", "w_down"]:
            mapping[f"{ms_prefix}.sglu.{proj}.weight"] = (
                f"{pt_prefix}.sglu.{proj}.weight"
            )

        # Layer norms
        mapping[f"{ms_prefix}.norm1.gamma"] = f"{pt_prefix}.post_norm1.weight"
        mapping[f"{ms_prefix}.norm1.beta"] = f"{pt_prefix}.post_norm1.bias"
        mapping[f"{ms_prefix}.norm2.gamma"] = f"{pt_prefix}.post_norm2.weight"
        mapping[f"{ms_prefix}.norm2.beta"] = f"{pt_prefix}.post_norm2.bias"

    return mapping


def convert_mindspore_checkpoint(
    mindspore_path: str,
    output_path: str,
    n_genes: int = 20000,
    n_layers: int = 40,
    config_name: str = "800M",
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert a MindSpore CellFM checkpoint to PyTorch format.

    Args:
        mindspore_path: Path to the MindSpore .ckpt file
        output_path: Path to save the converted PyTorch .pt file
        n_genes: Number of genes in the vocabulary
        n_layers: Number of encoder layers (40 for 800M, 12 for 80M)
        config_name: Configuration name ('80M' or '800M')
        strict: If True, fail if any parameter cannot be mapped
        verbose: If True, print conversion details

    Returns:
        Dict of converted state_dict

    Example:
        >>> convert_mindspore_checkpoint(
        ...     "official_cellfm.ckpt",
        ...     "cellfm_pytorch.pt",
        ...     n_layers=40,
        ... )
    """
    if verbose:
        print("=" * 60)
        print("  🔄 CellFM Weight Conversion: MindSpore → PyTorch")
        print("=" * 60)
        print(f"  Input:   {mindspore_path}")
        print(f"  Output:  {output_path}")
        print(f"  Config:  {config_name} ({n_layers} layers)")
        print()

    # Load MindSpore checkpoint
    ms_params = _load_mindspore_params(mindspore_path)

    if verbose:
        print(f"  📂 Loaded {len(ms_params)} parameters from MindSpore checkpoint")

    # Build name mapping
    name_map = _build_layer_mapping(n_layers)

    # Convert parameters
    pytorch_state_dict = {}
    converted = 0
    skipped = 0
    unmapped = []

    for ms_name, ms_value in ms_params.items():
        if ms_name in name_map:
            pt_name = name_map[ms_name]
            pt_tensor = torch.from_numpy(ms_value)
            pytorch_state_dict[pt_name] = pt_tensor
            converted += 1
        else:
            unmapped.append(ms_name)
            skipped += 1

    if verbose:
        print(f"  ✅ Converted:  {converted} parameters")
        if skipped > 0:
            print(f"  ⚠️  Skipped:   {skipped} parameters (no mapping)")
            for name in unmapped[:10]:
                print(f"      - {name}")
            if len(unmapped) > 10:
                print(f"      ... and {len(unmapped) - 10} more")

    if strict and unmapped:
        raise ValueError(
            f"Strict mode: {len(unmapped)} parameters could not be mapped. "
            f"First: {unmapped[0]}"
        )

    # Save as PyTorch checkpoint
    checkpoint = {
        "model_state_dict": pytorch_state_dict,
        "global_step": 0,
        "config": {
            "enc_nlayers": n_layers,
            "converted_from": "mindspore",
            "source_file": os.path.basename(mindspore_path),
        },
    }

    torch.save(checkpoint, output_path)

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  💾 Saved PyTorch checkpoint: {output_path} ({size_mb:.1f} MB)")
        print("=" * 60)

    return pytorch_state_dict


def _load_mindspore_params(path: str) -> Dict[str, np.ndarray]:
    """
    Load parameters from a MindSpore checkpoint file.

    MindSpore checkpoints store parameters as serialized numpy arrays.
    This function handles the deserialization.

    Args:
        path: Path to the .ckpt file

    Returns:
        Dict mapping parameter names to numpy arrays
    """
    try:
        import mindspore
        from mindspore import load_checkpoint
        params = load_checkpoint(path)
        return {
            name: param.asnumpy()
            for name, param in params.items()
        }
    except ImportError:
        # Try loading as numpy npz (alternative format)
        try:
            data = np.load(path, allow_pickle=True)
            if hasattr(data, 'files'):
                return {name: data[name] for name in data.files}
            return dict(data.item()) if data.ndim == 0 else {}
        except Exception:
            raise ImportError(
                "Cannot load MindSpore checkpoint. Either:\n"
                "1. Install MindSpore: pip install mindspore\n"
                "2. Convert the checkpoint to .npz format first\n"
                "3. Use the HuggingFace conversion: "
                "python -m cellfm.convert --source huggingface"
            )


def validate_conversion(
    pytorch_state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Validate a converted state dict against a PyTorch model.

    Checks:
    - All model parameters have a match in the state dict
    - Shapes match between state dict and model

    Args:
        pytorch_state_dict: Converted state dict
        model: Target PyTorch CellFM model
        verbose: Print validation details

    Returns:
        (matched, unmatched): Lists of matched and unmatched parameter names
    """
    model_params = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    all_model_keys = set(model_params.keys()) | set(model_buffers.keys())

    matched = []
    unmatched = []
    shape_mismatches = []

    for key in all_model_keys:
        if key in pytorch_state_dict:
            # Check shape
            model_shape = (
                model_params[key].shape
                if key in model_params
                else model_buffers[key].shape
            )
            ckpt_shape = pytorch_state_dict[key].shape

            if model_shape == ckpt_shape:
                matched.append(key)
            else:
                shape_mismatches.append((key, model_shape, ckpt_shape))
        else:
            unmatched.append(key)

    if verbose:
        print(f"\n  Validation Results:")
        print(f"    ✅ Matched:         {len(matched)}/{len(all_model_keys)}")
        print(f"    ❌ Missing:         {len(unmatched)}")
        print(f"    ⚠️  Shape mismatch: {len(shape_mismatches)}")

        if shape_mismatches:
            for name, expected, got in shape_mismatches[:5]:
                print(f"       {name}: expected {expected}, got {got}")

        if unmatched:
            for name in unmatched[:5]:
                print(f"       Missing: {name}")

    return matched, unmatched


def download_and_convert_huggingface(
    repo_id: str = "ShangguanNingyuan/CellFM",
    output_path: str = "cellfm_pretrained.pt",
    verbose: bool = True,
) -> str:
    """
    Download CellFM weights from HuggingFace and convert to PyTorch.

    Args:
        repo_id: HuggingFace repository ID
        output_path: Where to save the converted checkpoint
        verbose: Print progress

    Returns:
        Path to the converted checkpoint
    """
    if verbose:
        print(f"📥 Downloading CellFM weights from HuggingFace: {repo_id}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading. "
            "Install with: pip install huggingface-hub"
        )

    # Download the checkpoint
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="CellFM.ckpt",
        cache_dir=None,
    )

    if verbose:
        print(f"   Downloaded to: {ckpt_path}")
        print(f"   Converting to PyTorch format...")

    # Convert
    convert_mindspore_checkpoint(
        mindspore_path=ckpt_path,
        output_path=output_path,
        n_layers=40,
        config_name="800M",
        verbose=verbose,
    )

    return output_path


def main():
    """CLI entry point for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert CellFM weights between frameworks",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to MindSpore checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output", type=str, default="cellfm_converted.pt",
        help="Output path for PyTorch checkpoint",
    )
    parser.add_argument(
        "--source", type=str, default="mindspore",
        choices=["mindspore", "huggingface"],
        help="Source of weights: 'mindspore' (local file) or 'huggingface' (download)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=40,
        help="Number of encoder layers (40 for 800M, 12 for 80M)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any parameter cannot be mapped",
    )

    args = parser.parse_args()

    if args.source == "huggingface":
        download_and_convert_huggingface(output_path=args.output)
    else:
        if args.input is None:
            parser.error("--input is required for mindspore source")
        convert_mindspore_checkpoint(
            mindspore_path=args.input,
            output_path=args.output,
            n_layers=args.n_layers,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
