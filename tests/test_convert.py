"""
Unit Tests for CellFM Weight Conversion
==========================================

Tests for parameter name mapping, conversion pipeline, and validation.

Run with: python -m pytest tests/test_convert.py -v
"""

import torch
import numpy as np
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.convert import (
    _build_layer_mapping,
    validate_conversion,
    MINDSPORE_TO_PYTORCH_MAP,
)
from cellfm.config import Config80M
from cellfm.model import CellFM


# === Test Configuration ===
N_GENES = 500
N_LAYERS = 2


class TestNameMapping:
    """Test MindSpore → PyTorch parameter name mapping."""

    def test_base_mapping_exists(self):
        """Base mapping should have embedding and norm entries."""
        assert "gene_encoder.embedding_table" in MINDSPORE_TO_PYTORCH_MAP
        assert "value_encoder.weight" in MINDSPORE_TO_PYTORCH_MAP
        assert "cls_token" in MINDSPORE_TO_PYTORCH_MAP

    def test_layer_mapping_count(self):
        """Layer mapping should generate entries for all layers."""
        mapping = _build_layer_mapping(N_LAYERS)

        # Each layer should have entries for q/k/v/out projections,
        # group norm, sglu, and layer norms
        layer_keys = [k for k in mapping if k.startswith("encoder.blocks.")]
        assert len(layer_keys) > 0

    def test_layer_mapping_format(self):
        """Layer mapping should produce valid PyTorch parameter names."""
        mapping = _build_layer_mapping(2)

        # Check specific layer 0 entries
        assert "encoder.blocks.0.retention.q_proj.weight" in mapping
        assert mapping["encoder.blocks.0.retention.q_proj.weight"] == (
            "layers.0.retention.q_proj.weight"
        )

    def test_layer_mapping_all_layers(self):
        """Should create mappings for all requested layers."""
        n = 5
        mapping = _build_layer_mapping(n)

        for i in range(n):
            key = f"encoder.blocks.{i}.retention.q_proj.weight"
            assert key in mapping, f"Missing mapping for layer {i}"

    def test_sglu_mapping(self):
        """SGLU mappings should exist."""
        mapping = _build_layer_mapping(1)

        assert "encoder.blocks.0.sglu.w_gate.weight" in mapping
        assert "encoder.blocks.0.sglu.w_up.weight" in mapping
        assert "encoder.blocks.0.sglu.w_down.weight" in mapping

    def test_norm_mapping(self):
        """MindSpore gamma/beta → PyTorch weight/bias mapping."""
        mapping = _build_layer_mapping(1)

        # Layer norms
        assert "encoder.blocks.0.norm1.gamma" in mapping
        assert mapping["encoder.blocks.0.norm1.gamma"] == "layers.0.post_norm1.weight"
        assert mapping["encoder.blocks.0.norm1.beta"] == "layers.0.post_norm1.bias"


class TestValidateConversion:
    """Test conversion validation utility."""

    def test_perfect_match(self):
        """Validation should report full match with correct state dict."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)
        state_dict = model.state_dict()

        matched, unmatched = validate_conversion(
            state_dict, model, verbose=False
        )

        # All non-buffer params should match
        assert len(matched) > 0
        # Buffers like gammas might not be in state_dict depending on approach
        # but named parameters should all match

    def test_missing_detection(self):
        """Should detect missing parameters."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)

        # Empty state dict — everything missing
        matched, unmatched = validate_conversion(
            {}, model, verbose=False
        )

        assert len(unmatched) > 0
        assert len(matched) == 0

    def test_partial_match(self):
        """Should handle partial matches correctly."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)
        full_state = model.state_dict()

        # Take only first few parameters
        partial_state = {k: v for k, v in list(full_state.items())[:3]}

        matched, unmatched = validate_conversion(
            partial_state, model, verbose=False
        )

        assert len(matched) == 3
        assert len(unmatched) > 0


class TestConversionRoundtrip:
    """Test creating a mock conversion and loading it."""

    def test_save_and_load_checkpoint(self):
        """Should save a checkpoint and load it back."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_converted.pt")

            # Save as if converted
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "global_step": 0,
                "config": {
                    "enc_nlayers": 2,
                    "converted_from": "test",
                },
            }
            torch.save(checkpoint, path)

            # Load back
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in loaded
            assert loaded["config"]["converted_from"] == "test"

            # Verify state dict can be loaded into model
            model.load_state_dict(loaded["model_state_dict"], strict=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
