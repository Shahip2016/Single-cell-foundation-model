# 🧬 CellFM — A Single-Cell Foundation Model

[![Paper](https://img.shields.io/badge/bioRxiv-2024.06.04.597369-blue)](https://www.biorxiv.org/content/10.1101/2024.06.04.597369v1)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)

> **A clean, well-documented PyTorch reimplementation of "CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells"**
>
> *Original paper by Yuansong Zeng, Jiancong Xie, Zhuoyi Wei, et al.*

---

## 🤔 What Is This? (The Plain English Version)

Imagine every cell in your body is a tiny city with ~20,000 workers (genes). Each worker works at different intensities depending on the cell type — a brain cell has different "busy workers" than a blood cell.

**CellFM** is an AI that has "read" the activity reports of **100 million human cells** and learned the patterns. Think of it like ChatGPT, but instead of learning language from text, it learns the "language of cells" from gene expression data.

After training, CellFM can:

| Task | What It Does | Analogy |
|------|-------------|---------|
| 🏷️ **Cell Annotation** | Identify what type a cell is | "This is a T-cell" |
| 🧪 **Perturbation Prediction** | Predict what happens when you disable a gene | "If you silence gene X, here's what changes" |
| 🔬 **Gene Function Prediction** | Predict what unknown genes do | "This mystery gene behaves like known gene Y" |
| 🔗 **Batch Integration** | Remove technical noise from experiments | "These datasets were run differently but describe the same biology" |

---

## 🏗️ Architecture Overview

CellFM uses a **RetNet** (Retentive Network) backbone — a Transformer variant with linear complexity, making it efficient for processing thousands of genes per cell.

```
Input: Gene Expression Vector (20,000 genes)
    ↓
┌─────────────────────────────────┐
│  📦 Embedding Module            │  Convert expression values → vectors
│     • Gene ID Embedding         │
│     • Expression Value Encoding │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  🧠 ERetNet Backbone            │  Learn gene-gene relationships
│     40 layers × 48 heads        │
│     • Multi-Scale Retention     │  (like attention, but faster)
│     • SGLU Feed-Forward         │  (smart signal amplification)
│     • Layer Normalization       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  🎯 Task-Specific Head          │  Adapted via LoRA fine-tuning
│     • Classification            │
│     • Regression                │
│     • Embedding extraction      │
└─────────────────────────────────┘
    ↓
Output: Predictions / Cell Embeddings
```

### Model Configurations

| Config | Parameters | Layers | Heads | Embed Dim | Use Case |
|--------|-----------|--------|-------|-----------|----------|
| **80M** | ~80 million | 12 | 16 | 512 | Fine-tuning, experimentation |
| **800M** | ~800 million | 40 | 48 | 1536 | Full-scale inference |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:Shahip2016/Single-cell-foundation-model.git
cd Single-cell-foundation-model

# Create a conda environment
conda create -n cellfm python=3.9
conda activate cellfm

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from cellfm.config import Config80M
from cellfm.model import CellFM

# Initialize model with 80M config
cfg = Config80M()
model = CellFM(n_genes=20000, config=cfg)

# Forward pass with dummy data
# gene_ids: which genes are expressed (batch, seq_len)
# gene_values: how much they're expressed (batch, seq_len)
gene_ids = torch.randint(0, 20000, (4, 2048))
gene_values = torch.randn(4, 2048)

output = model(gene_ids, gene_values)
print(f"Output shape: {output.shape}")  # (4, 2048, 512)
```

---

## 📚 Learning Resources

- **[Layman's Guide](docs/LAYMAN_GUIDE.md)** — If you're new to single-cell biology or foundation models
- **[Architecture Deep-Dive](docs/ARCHITECTURE.md)** — Technical details of every component
- **[API Reference](docs/API_REFERENCE.md)** — Complete API documentation
- **[Contributing Guide](docs/CONTRIBUTING.md)** — How to contribute to the project

### Available Modules

| Module | File | Purpose |
|--------|------|---------|
| Config | `cellfm/config.py` | Model hyperparameters (80M, 800M) |
| Embedding | `cellfm/embedding.py` | Gene expression → vector embeddings |
| Retention | `cellfm/retention.py` | Multi-scale retention (RetNet) |
| SGLU | `cellfm/sglu.py` | SwiGLU feed-forward layer |
| Model | `cellfm/model.py` | Full CellFM assembly |
| LoRA | `cellfm/lora.py` | Low-rank adaptation for fine-tuning |
| Data | `cellfm/data.py` | Data loading & preprocessing pipeline |
| Trainer | `cellfm/trainer.py` | Training loop, LR scheduling, checkpointing |
| Inference | `cellfm/inference.py` | Pretrained loading, prediction, embedding extraction |
| **Metrics** | `cellfm/metrics.py` | Accuracy, F1, confusion matrix, evaluator |
| **Visualization** | `cellfm/visualization.py` | UMAP/t-SNE plots, training curves, heatmaps |
| **CLI** | `cellfm/cli.py` | Command-line interface for train/predict/embed |
| **Convert** | `cellfm/convert.py` | MindSpore → PyTorch weight conversion |

### Command-Line Usage

```bash
# Show model info
python -m cellfm info --config 80M

# Train a model
python -m cellfm train --data cells.h5ad --label-col cell_type --epochs 10

# Predict cell types
python -m cellfm predict --checkpoint best.pt --data test.h5ad

# Extract embeddings
python -m cellfm embed --checkpoint best.pt --data cells.h5ad --output emb.npy
```

---

## 📄 Citation

If you use this work, please cite the original paper:

```bibtex
@article{CellFM,
    title={CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells},
    author={Yuansong Zeng, Jiancong Xie, Zhuoyi Wei, Yun Su, Ningyuan Shangguan, Shuangyu Yang,
            Chengyang Zhang, Wenbing Li, Jinbo Zhang, Nan Fang, Hongyu Zhang, Huiying Zhao,
            Yutong Lu, Jue Fan, Weijiang Yu, and Yuedong Yang},
    journal={bioRxiv},
    year={2024},
    doi={10.1101/2024.06.04.597369}
}
```

---

## 📜 License

This project is licensed under [CC BY-NC-ND 4.0](LICENSE), matching the original CellFM license.

## 🙏 Acknowledgments

- [Original CellFM (MindSpore)](https://github.com/biomed-AI/CellFM)
- [CellFM-torch](https://github.com/biomed-AI/CellFM-torch) — Official PyTorch port
- [HuggingFace Model Card](https://huggingface.co/ShangguanNingyuan/CellFM)
