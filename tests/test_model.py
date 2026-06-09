"""
Unit Tests for CellFM Model
==============================

These tests verify that:
1. All modules have correct output shapes
2. Forward pass works end-to-end
3. Parameter counts are reasonable
4. LoRA integration works correctly
5. Gradient flow is healthy

Run with: python -m pytest tests/test_model.py -v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellfm.config import Config80M, Config800M
from cellfm.embedding import GeneExpressionEmbedding
from cellfm.retention import MultiScaleRetention
from cellfm.sglu import SGLU
from cellfm.lora import LoRALinear, apply_lora_to_model, get_lora_params
from cellfm.model import CellFM, ERetNetBlock


# === Test Configuration ===
BATCH_SIZE = 2
SEQ_LEN = 128  # Small for fast testing (real: 2048)
N_GENES = 1000  # Small gene vocabulary for testing (real: ~20,000)


class TestGeneExpressionEmbedding:
    """Test the embedding module."""

    def test_output_shape(self):
        """Embedding should produce (batch, seq+1, dim) — +1 for CLS token."""
        embed = GeneExpressionEmbedding(n_genes=N_GENES, embed_dim=64)
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        output = embed(gene_ids, gene_values)

        assert output.shape == (BATCH_SIZE, SEQ_LEN + 1, 64), (
            f"Expected (2, 129, 64), got {output.shape}"
        )

    def test_cls_token_present(self):
        """CLS token should be at position 0."""
        embed = GeneExpressionEmbedding(n_genes=N_GENES, embed_dim=64)
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        output = embed(gene_ids, gene_values)

        # CLS token position exists
        cls_output = output[:, 0, :]  # Should not raise
        assert cls_output.shape == (BATCH_SIZE, 64)

    def test_padding_index_zero(self):
        """Gene index 0 should produce zero embedding (padding)."""
        embed = GeneExpressionEmbedding(n_genes=N_GENES, embed_dim=64)
        padding_embedding = embed.gene_encoder(torch.tensor([0]))
        assert torch.all(padding_embedding == 0), "Padding index should be zero vector"


class TestMultiScaleRetention:
    """Test the RetNet retention mechanism."""

    def test_output_shape(self):
        """Retention should preserve input shape."""
        retention = MultiScaleRetention(embed_dim=64, num_heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)

        output = retention(x)

        assert output.shape == x.shape, (
            f"Expected {x.shape}, got {output.shape}"
        )

    def test_with_padding_mask(self):
        """Retention should work with padding masks."""
        retention = MultiScaleRetention(embed_dim=64, num_heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)

        # Mask last 10 positions as padding
        mask = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        mask[:, -10:] = True

        output = retention(x, key_padding_mask=mask)

        assert output.shape == x.shape

    def test_decay_matrix_shape(self):
        """Decay matrix should be (num_heads, seq, seq)."""
        retention = MultiScaleRetention(embed_dim=64, num_heads=4)
        D = retention._build_decay_matrix(SEQ_LEN)

        assert D.shape == (4, SEQ_LEN, SEQ_LEN)

    def test_decay_matrix_diagonal(self):
        """Decay matrix diagonal should be 1 (self-interaction has no decay)."""
        retention = MultiScaleRetention(embed_dim=64, num_heads=4)
        D = retention._build_decay_matrix(16)

        # Diagonal elements should be close to 1 (gamma^0 = 1)
        for h in range(4):
            for i in range(16):
                assert abs(D[h, i, i].item() - 1.0) < 1e-5, (
                    f"D[{h},{i},{i}] should be 1.0, got {D[h,i,i].item()}"
                )

    def test_decay_rates_different(self):
        """Each head should have a different decay rate."""
        retention = MultiScaleRetention(embed_dim=64, num_heads=4)
        gammas = retention.gammas

        # All decay rates should be unique
        assert len(torch.unique(gammas)) == 4, "All heads should have different γ"


class TestSGLU:
    """Test the SwiGLU feed-forward module."""

    def test_output_shape(self):
        """SGLU should preserve input shape."""
        sglu = SGLU(embed_dim=64)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)

        output = sglu(x)

        assert output.shape == x.shape

    def test_hidden_dim_calculation(self):
        """Default hidden dim should be ~2.67× embed_dim, rounded to 64."""
        sglu = SGLU(embed_dim=512)
        # 8/3 * 512 ≈ 1365, rounded to nearest 64 = 1408
        expected = ((int(8 / 3 * 512) + 63) // 64) * 64
        assert sglu.hidden_dim == expected


class TestERetNetBlock:
    """Test a single ERetNet block."""

    def test_output_shape(self):
        """Block should preserve input shape."""
        block = ERetNetBlock(embed_dim=64, num_heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)

        output = block(x)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (residual + processing)."""
        block = ERetNetBlock(embed_dim=64, num_heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)

        output = block(x)

        # Output should not be identical to input
        assert not torch.allclose(output, x, atol=1e-6)


class TestCellFM:
    """Test the full CellFM model."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False
        return CellFM(n_genes=N_GENES, config=cfg)

    def test_encoder_output_shape(self, model):
        """Encoder mode should return (batch, seq+1, dim)."""
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        output = model(gene_ids, gene_values)

        assert output.shape == (BATCH_SIZE, SEQ_LEN + 1, 64)

    def test_cell_embedding(self, model):
        """Cell embedding should return (batch, dim)."""
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        cell_emb = model(gene_ids, gene_values, return_cell_embedding=True)

        assert cell_emb.shape == (BATCH_SIZE, 64)

    def test_classifier_mode(self):
        """With classifier, should return (batch, num_classes)."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=10)
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        logits = model(gene_ids, gene_values)

        assert logits.shape == (BATCH_SIZE, 10)

    def test_gene_embeddings(self, model):
        """Gene embeddings should return (batch, seq, dim) — no CLS."""
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        gene_embs = model.get_gene_embeddings(gene_ids, gene_values)

        assert gene_embs.shape == (BATCH_SIZE, SEQ_LEN, 64)

    def test_gradient_flow(self, model):
        """Gradients should flow through the entire model."""
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)

        output = model(gene_ids, gene_values, return_cell_embedding=True)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_count(self, model):
        """Verify parameter counting utility works."""
        counts = model.count_parameters()

        assert "total" in counts
        assert "embedding" in counts
        assert "retention_total" in counts
        assert "sglu_total" in counts
        assert counts["total"] > 0


class TestLoRA:
    """Test LoRA (Low-Rank Adaptation) functionality."""

    def test_lora_linear_shape(self):
        """LoRA linear should produce same output shape as original."""
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)
        output = lora(x)

        assert output.shape == (BATCH_SIZE, SEQ_LEN, 64)

    def test_lora_zero_init(self):
        """LoRA should start with zero update (B initialized to zero)."""
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)

        x = torch.randn(BATCH_SIZE, 64)

        # With B=0, LoRA output should equal original output
        with torch.no_grad():
            original_out = original(x)
            lora_out = lora(x)

        assert torch.allclose(original_out, lora_out, atol=1e-6), (
            "LoRA should produce same output as original at initialization"
        )

    def test_lora_freezes_original(self):
        """Original weights should be frozen after LoRA wrapping."""
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4)

        for param in lora.original_linear.parameters():
            assert not param.requires_grad, "Original weights should be frozen"

    def test_lora_params_trainable(self):
        """LoRA A and B matrices should be trainable."""
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4)

        assert lora.lora_A.requires_grad, "LoRA A should be trainable"
        assert lora.lora_B.requires_grad, "LoRA B should be trainable"

    def test_apply_lora_to_model(self):
        """LoRA should be applicable to the full CellFM model."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)

        # Count params before LoRA
        total_before = sum(p.numel() for p in model.parameters())

        # Apply LoRA
        model = apply_lora_to_model(model, rank=4, alpha=8)

        # Count trainable params after LoRA
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_after = sum(p.numel() for p in model.parameters())

        # LoRA should add parameters
        assert total_after > total_before

        # Trainable should be much less than total
        assert trainable < total_after

        # Model should still work
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)
        output = model(gene_ids, gene_values, return_cell_embedding=True)
        assert output.shape == (BATCH_SIZE, 64)

    def test_get_lora_params(self):
        """Should extract only LoRA parameters."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        model = CellFM(n_genes=N_GENES, config=cfg)
        model = apply_lora_to_model(model, rank=4)

        lora_params = get_lora_params(model)
        assert len(lora_params) > 0, "Should find LoRA parameters"

        # All should have 'lora_' in their name
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert any(p is param for p in lora_params)


# === Integration Test ===
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_80m(self):
        """Test complete forward pass with 80M config (small dims for speed)."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        # Create model
        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=5)

        # Create dummy data
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)
        labels = torch.randint(0, 5, (BATCH_SIZE,))

        # Forward pass
        logits = model(gene_ids, gene_values)
        assert logits.shape == (BATCH_SIZE, 5)

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        assert loss.item() > 0

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Should have gradients after backward"

    def test_full_pipeline_with_lora(self):
        """Test complete forward + backward with LoRA fine-tuning."""
        cfg = Config80M()
        cfg.enc_dims = 64
        cfg.enc_nlayers = 2
        cfg.enc_num_heads = 4
        cfg.recompute = False

        # Create and LoRA-ify model
        model = CellFM(n_genes=N_GENES, config=cfg, num_classes=5)
        model = apply_lora_to_model(model, rank=4, alpha=8)

        # Only LoRA params should be trainable
        optimizer = torch.optim.Adam(get_lora_params(model), lr=1e-3)

        # Training step
        gene_ids = torch.randint(1, N_GENES, (BATCH_SIZE, SEQ_LEN))
        gene_values = torch.randn(BATCH_SIZE, SEQ_LEN)
        labels = torch.randint(0, 5, (BATCH_SIZE,))

        logits = model(gene_ids, gene_values)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
