# 🏗️ Architecture Deep Dive: CellFM Internals

> *A technical reference for understanding every component of the CellFM model.*

---

## Overview

CellFM processes single-cell gene expression data through a pipeline of four major components:

```
Input: (gene_ids, gene_values) — per-cell gene expression
    ↓
┌─────────────────────────────────────────────┐
│  1. GeneExpressionEmbedding                 │
│     gene_encoder + value_encoder + CLS      │
│     Output: (batch, seq+1, embed_dim)       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  2. N × ERetNetBlock                        │
│     ├── LayerNorm                           │
│     ├── MultiScaleRetention  (+ residual)   │
│     ├── LayerNorm                           │
│     └── SGLU Feed-Forward    (+ residual)   │
│     Output: (batch, seq+1, embed_dim)       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  3. Final LayerNorm                         │
│     Output: (batch, seq+1, embed_dim)       │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  4. Task Head (Classification / Embedding)  │
│     CLS token → Linear → GELU → Linear     │
│     Output: (batch, num_classes)            │
└─────────────────────────────────────────────┘
```

---

## Component 1: Gene Expression Embedding

**File**: [`cellfm/embedding.py`](../cellfm/embedding.py)

### Purpose

Convert raw gene expression data (gene indices + expression values) into continuous vector representations that the neural network can process.

### Architecture

| Sub-component | Type | Shape | Description |
|---|---|---|---|
| `gene_encoder` | `nn.Embedding(n_genes+1, embed_dim)` | `(vocab, d)` | Maps gene index → vector (padding_idx=0) |
| `value_encoder` | `nn.Linear(1, embed_dim)` | `(1, d)` | Maps scalar expression → vector |
| `cls_token` | `nn.Parameter` | `(1, 1, d)` | Learnable cell-level summary token |
| `pos_embed` | `nn.Parameter` | `(1, 2049, d)` | Positional encoding for gene rank ordering |

### Forward Flow

```
gene_ids: (B, L)  →  gene_encoder  →  gene_emb: (B, L, d)
gene_values: (B, L)  →  unsqueeze(-1)  →  value_encoder  →  value_emb: (B, L, d)

combined = gene_emb + value_emb                          # (B, L, d)
combined = cat([cls_token, combined], dim=1)              # (B, L+1, d)
combined = combined + pos_embed[:, :L+1, :]               # + positional encoding
combined = LayerNorm(Dropout(combined))                   # (B, L+1, d)
```

### Design Decisions

- **Gene ID 0 = padding**: The embedding for index 0 is a zero vector, so padded positions contribute nothing.
- **Additive composition**: `gene_emb + value_emb` rather than concatenation, preserving dimensionality.
- **CLS token prepended**: Following BERT convention. After processing through all layers, the CLS token at position 0 serves as a cell-level representation.
- **Positional encoding**: Since genes are sorted by expression (highest first), position encodes expression rank.

---

## Component 2: Multi-Scale Retention

**File**: [`cellfm/retention.py`](../cellfm/retention.py)

### Purpose

Discover gene-gene relationships efficiently. This is the core "attention-like" mechanism that allows CellFM to understand which genes interact.

### Retention vs. Attention

| Property | Transformer Attention | RetNet Retention |
|---|---|---|
| **Formula** | `softmax(QK^T/√d) · V` | `(QK^T ⊙ D) · V` |
| **Normalization** | Softmax (sum-to-1) | Sum normalization |
| **Masking** | Causal mask (0/1) | Exponential decay matrix D |
| **Complexity** | O(n²) (parallelizable) | O(n²) parallel / O(n) recurrent |

### The Decay Matrix D

The key innovation of RetNet is the decay matrix `D`, where:

```
D[n, m] = γ^(n-m)  if n ≥ m, else 0
```

Each head has a different decay rate γ:

```python
γ_h = 1 - 2^(-5 - h * 8/num_heads)
```

This creates a spectrum:
- **Head 0**: γ ≈ 0.969 → slow decay → captures long-range interactions
- **Head N**: γ ≈ 0.853 → fast decay → captures local interactions

### Forward Flow

```
x: (B, L, d)
    ↓
Q = q_proj(x)        →  reshape  →  (B, H, L, d/H)
K = k_proj(x)        →  reshape  →  (B, H, L, d/H)
V = v_proj(x)        →  reshape  →  (B, H, L, d/H)
    ↓
retention = (Q @ K^T) / √(d/H)         # (B, H, L, L) — raw scores
retention = retention * D               # Apply per-head decay
retention = retention / sum(retention)  # Sum-normalize (not softmax!)
    ↓
output = retention @ V                  # (B, H, L, d/H)
output = reshape → GroupNorm → out_proj → dropout  # (B, L, d)
```

### Design Decisions

- **Fixed decay rates**: γ values are not learned — they're set by formula to ensure diverse scales.
- **GroupNorm**: Applied per-head (not LayerNorm) for better multi-head interaction.
- **No bias in projections**: Q, K, V, out projections are bias-free (common in modern architectures).

---

## Component 3: SGLU Feed-Forward

**File**: [`cellfm/sglu.py`](../cellfm/sglu.py)

### Purpose

Non-linear feature processing with gated activation. Selectively amplifies important features and suppresses irrelevant ones.

### Architecture (SwiGLU)

```
x: (B, L, d)
    ↓
gate = SiLU(w_gate(x))     # (B, L, h) — "what's important"
content = w_up(x)           # (B, L, h) — "what's the content"
hidden = gate * content     # (B, L, h) — gated output
output = w_down(hidden)     # (B, L, d) — project back
```

Where:
- `SiLU(x) = x · σ(x)` (smooth, non-linear activation)
- Hidden dim h = ⌈(8/3 × d) / 64⌉ × 64 (rounded for GPU efficiency)

### Design Decisions

- **SwiGLU over ReLU FFN**: Better performance empirically (used in LLaMA, PaLM, CellFM).
- **No bias**: All three projections are bias-free.
- **Hidden dim rounding**: Rounded to nearest multiple of 64 for GPU tensor core alignment.

---

## Component 4: LoRA Fine-Tuning

**File**: [`cellfm/lora.py`](../cellfm/lora.py)

### Purpose

Efficiently adapt the pre-trained model for downstream tasks by only training small adapter matrices (~1% of total parameters).

### How It Works

For each targeted linear layer W ∈ ℝ^{d×d}:

```
Original:   y = W·x
With LoRA:  y = W·x + (α/r) · (x @ A @ B)

Where:
  W: (d, d) — frozen original weights
  A: (d, r) — trainable down-projection (Kaiming init)
  B: (r, d) — trainable up-projection (zero init)
  r: rank (typically 4-16)
  α: scaling factor
```

### Target Modules

By default, LoRA is applied to the attention projections:
- `q_proj`, `k_proj`, `v_proj`, `out_proj`

### Parameter Savings

| Model | Full Params | LoRA Params (r=8) | Savings |
|---|---|---|---|
| **80M** | ~80M | ~786K | 99.0% |
| **800M** | ~800M | ~3.9M | 99.5% |

---

## Training Pipeline

**File**: [`cellfm/trainer.py`](../cellfm/trainer.py)

### Learning Rate Schedule

CellFM uses a cosine warmup schedule:
1. **Warmup** (10% of steps): Linear ramp from `start_lr` to `peak_lr`
2. **Cosine decay** (90% of steps): Smooth decrease to `min_lr`

### Pre-training Objective

**Masked Gene Prediction**: Randomly mask 50% of genes and predict their expression values.

### Fine-tuning Objective

**Cross-entropy**: Standard classification loss for cell type annotation tasks.

---

## Data Pipeline

**File**: [`cellfm/data.py`](../cellfm/data.py)

### Preprocessing Steps

1. **Load**: Read AnnData (.h5ad) file
2. **Normalize**: Library-size normalization (total = 10,000) + log1p
3. **Rank**: Sort genes by expression value (descending)
4. **Truncate**: Keep top-k genes (default: 2048)
5. **Pad**: Zero-pad shorter sequences

### Why Rank Ordering?

Unlike text tokens, genes have no inherent "order." CellFM creates an ordering by sorting genes from most-expressed to least-expressed. This:
- Ensures the most biologically important genes appear first
- Provides consistent input structure across different cells
- Enables meaningful positional encoding (position = expression rank)

---

## Model Configurations

| Parameter | 80M Config | 800M Config |
|---|---|---|
| `enc_dims` | 512 | 1536 |
| `enc_nlayers` | 12 | 40 |
| `enc_num_heads` | 16 | 48 |
| `head_dim` | 32 | 32 |
| `sglu_hidden` | 1408 | 4096 |
| `nonz_len` | 2048 | 2048 |
| `recompute` | False | True |
| `GPU Memory` | ~8 GB | ~32 GB |

---

## References

1. **CellFM Paper**: Zeng et al., "CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells", bioRxiv 2024.
2. **RetNet**: Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023.
3. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020.
4. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
