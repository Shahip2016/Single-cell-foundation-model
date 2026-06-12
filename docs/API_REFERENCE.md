# 📖 CellFM API Reference

> *Complete API documentation for the CellFM library.*

---

## Table of Contents

- [Configuration](#configuration)
- [Model](#model)
- [Embedding](#embedding)
- [Retention](#retention)
- [SGLU](#sglu)
- [LoRA](#lora)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Metrics](#metrics)
- [Visualization](#visualization)
- [Weight Conversion](#weight-conversion)
- [CLI](#cli)

---

## Configuration

**File**: [`cellfm/config.py`](../cellfm/config.py)

### `Config80M`

Compact configuration (80 million parameters) for fine-tuning and experimentation.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `enc_dims` | 512 | Embedding dimension |
| `enc_nlayers` | 12 | Number of ERetNet layers |
| `enc_num_heads` | 16 | Number of retention heads |
| `nonz_len` | 2048 | Max non-zero genes per cell |
| `dropout` | 0.1 | Dropout rate |
| `recompute` | False | Gradient checkpointing |

### `Config800M`

Full-scale configuration (800 million parameters) matching the published paper.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `enc_dims` | 1536 | Embedding dimension |
| `enc_nlayers` | 40 | Number of ERetNet layers |
| `enc_num_heads` | 48 | Number of retention heads |
| `nonz_len` | 2048 | Max non-zero genes per cell |
| `dropout` | 0.1 | Dropout rate |
| `recompute` | True | Gradient checkpointing (saves ~50% memory) |

---

## Model

**File**: [`cellfm/model.py`](../cellfm/model.py)

### `CellFM(n_genes, config, num_classes=0)`

The complete CellFM foundation model.

**Parameters**:
- `n_genes` (int): Number of genes in the vocabulary (~20,000 for human)
- `config`: Configuration object (`Config80M` or `Config800M`)
- `num_classes` (int): Output classes (0 = encoder-only mode)

**Methods**:

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `forward(gene_ids, gene_values)` | `(B, L)`, `(B, L)` | `(B, L+1, d)` or `(B, C)` | Full forward pass |
| `forward(..., return_cell_embedding=True)` | — | `(B, d)` | CLS token embedding |
| `get_gene_embeddings(gene_ids, gene_values)` | `(B, L)`, `(B, L)` | `(B, L, d)` | Per-gene representations |
| `count_parameters()` | — | `Dict[str, int]` | Parameter count by component |

**Example**:
```python
from cellfm import CellFM, Config80M

cfg = Config80M()
model = CellFM(n_genes=20000, config=cfg, num_classes=10)

gene_ids = torch.randint(0, 20000, (4, 2048))
gene_values = torch.randn(4, 2048)
logits = model(gene_ids, gene_values)  # (4, 10)
```

### `ERetNetBlock(embed_dim, num_heads, dropout=0.1)`

A single ERetNet block with retention + SGLU + residual connections.

---

## Embedding

**File**: [`cellfm/embedding.py`](../cellfm/embedding.py)

### `GeneExpressionEmbedding(n_genes, embed_dim, dropout=0.1)`

Converts gene IDs and expression values into vector representations.

**Forward**: `(gene_ids, gene_values)` → `(B, L+1, d)`

The +1 accounts for the prepended CLS token.

---

## Retention

**File**: [`cellfm/retention.py`](../cellfm/retention.py)

### `MultiScaleRetention(embed_dim, num_heads, dropout=0.1)`

Multi-scale retention mechanism (RetNet).

**Forward**: `(x, key_padding_mask=None)` → `(B, L, d)`

Each head uses a different exponential decay rate γ for multi-scale analysis.

---

## SGLU

**File**: [`cellfm/sglu.py`](../cellfm/sglu.py)

### `SGLU(embed_dim, hidden_dim=None, dropout=0.1)`

SwiGLU feed-forward network: `output = (SiLU(W_gate·x) ⊙ W_up·x) · W_down`

Default hidden dim: `⌈(8/3 × embed_dim) / 64⌉ × 64`

---

## LoRA

**File**: [`cellfm/lora.py`](../cellfm/lora.py)

### `LoRALinear(original_linear, rank=8, alpha=16.0, dropout=0.0)`

Wraps an `nn.Linear` with low-rank adaptation: `y = W·x + (α/r)·A·B·x`

### `apply_lora_to_model(model, rank=8, alpha=16.0, target_modules=None)`

Applies LoRA to all matching linear layers in the model.

**Default targets**: `["q_proj", "k_proj", "v_proj", "out_proj"]`

### `get_lora_params(model)` → `List[Parameter]`

Returns only the LoRA parameters for optimizer construction.

**Example**:
```python
model = apply_lora_to_model(model, rank=8, alpha=16)
optimizer = torch.optim.Adam(get_lora_params(model), lr=1e-4)
```

---

## Data Pipeline

**File**: [`cellfm/data.py`](../cellfm/data.py)

### `CellDataset(adata, max_genes=2048, label_col=None, normalize=True)`

PyTorch Dataset that converts AnnData objects to CellFM's input format.

**Returns per item**: `Dict` with `gene_ids`, `gene_values`, `padding_mask`, and optionally `label`.

### `SyntheticCellDataset(n_cells, n_genes, max_genes, n_classes=0, sparsity=0.9)`

Generates fake single-cell data for testing and benchmarking.

### `create_dataloader(adata, batch_size=16, max_genes=2048, ...)`

Creates a ready-to-use PyTorch DataLoader from an AnnData object.

### `collate_cells(batch)`

Custom collation function for batching cell data.

---

## Training

**File**: [`cellfm/trainer.py`](../cellfm/trainer.py)

### `CellFMTrainer(model, config, output_dir, device)`

Complete training engine.

**Main method**: `train(train_loader, val_loader=None, num_epochs=5, ...)`

Returns training history dict with `train_loss`, `val_loss`, `lr`.

### `CosineWarmupScheduler(optimizer, warmup_steps, total_steps, ...)`

Linear warmup + cosine decay learning rate schedule.

### `MaskedMSELoss(reduction='mean')`

MSE loss that ignores padded positions (for pre-training).

---

## Inference

**File**: [`cellfm/inference.py`](../cellfm/inference.py)

### `load_pretrained(checkpoint_path, n_genes=20000, config=None, ...)`

Load a pre-trained CellFM model from a checkpoint file.

### `CellFMPredictor(model, device='cpu', max_genes=2048)`

High-level inference wrapper.

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_cell_types(gene_ids, gene_values)` | `(B,)` int tensor | Predicted class indices |
| `predict_probabilities(gene_ids, gene_values)` | `(B, C)` float tensor | Softmax probabilities |
| `get_cell_embeddings(gene_ids, gene_values)` | `(B, d)` float tensor | CLS token embeddings |

### `extract_embeddings(model, dataloader, device='cpu')`

Extract cell embeddings from an entire dataset. Returns `(N, d)` numpy array.

### `export_model(model, output_path, ...)`

Export model to TorchScript format for deployment.

---

## Metrics

**File**: [`cellfm/metrics.py`](../cellfm/metrics.py)

### `accuracy(predictions, targets, top_k=1)` → `float`

Compute top-k classification accuracy.

### `precision_recall_f1(predictions, targets, num_classes=None, average='macro')` → `Dict`

Returns dict with `precision`, `recall`, `f1`, and `per_class_*` breakdowns.

### `confusion_matrix(predictions, targets, num_classes=None)` → `np.ndarray`

Returns `(C, C)` confusion matrix.

### `CellFMEvaluator(model, device, label_names=None)`

High-level evaluator with `evaluate(dataloader)` and `print_report(results)`.

---

## Visualization

**File**: [`cellfm/visualization.py`](../cellfm/visualization.py)

### `plot_embeddings(embeddings, labels=None, method='umap', ...)`

Plot cell embeddings in 2D using UMAP, t-SNE, or PCA.

### `plot_training_history(history, ...)`

Plot training loss, validation loss, and learning rate curves.

### `plot_confusion_matrix(cm, label_names=None, normalize=False, ...)`

Plot confusion matrix as a color-coded heatmap.

### `plot_gene_importance(gene_ids, importance_scores, top_k=20, ...)`

Plot horizontal bar chart of most important genes.

### `plot_model_summary(param_counts, ...)`

Plot parameter distribution as a donut chart.

### `reduce_dimensions(embeddings, method='pca', n_components=2)`

Reduce high-dimensional embeddings to 2D/3D.

---

## Weight Conversion

**File**: [`cellfm/convert.py`](../cellfm/convert.py)

### `convert_mindspore_checkpoint(mindspore_path, output_path, ...)`

Convert MindSpore `.ckpt` files to PyTorch `.pt` format.

### `validate_conversion(pytorch_state_dict, model)`

Validate converted weights against a PyTorch model architecture.

### `download_and_convert_huggingface(repo_id, output_path)`

Download from HuggingFace and convert automatically.

---

## CLI

**File**: [`cellfm/cli.py`](../cellfm/cli.py)

Run CellFM from the command line:

```bash
# Show model configuration
python -m cellfm info --config 80M

# Train a model
python -m cellfm train --data cells.h5ad --label-col cell_type --epochs 10

# Predict cell types
python -m cellfm predict --checkpoint best.pt --data test.h5ad

# Extract embeddings
python -m cellfm embed --checkpoint best.pt --data cells.h5ad --output emb.npy
```

---

*For more details, see the source code docstrings and the [Architecture Deep-Dive](ARCHITECTURE.md).*
