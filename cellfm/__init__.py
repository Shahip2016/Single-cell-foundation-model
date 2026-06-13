"""
CellFM — A Single-Cell Foundation Model

A PyTorch reimplementation of "CellFM: a large-scale foundation model
pre-trained on transcriptomics of 100 million human cells"

Paper: https://www.biorxiv.org/content/10.1101/2024.06.04.597369v1
Original code: https://github.com/biomed-AI/CellFM

Modules:
    - config:          Model configurations (Config80M, Config800M)
    - embedding:       Gene expression embedding layer
    - retention:       Multi-scale retention mechanism (RetNet)
    - sglu:            SwiGLU feed-forward layer
    - model:           Full CellFM model assembly
    - lora:            LoRA fine-tuning adapters
    - data:            Data loading and preprocessing pipeline
    - trainer:         Training engine with LR scheduling
    - inference:       Inference utilities and pretrained loading
    - metrics:         Evaluation metrics (accuracy, F1, confusion matrix)
    - visualization:   Plotting utilities (UMAP, training curves, heatmaps)
    - cli:             Command-line interface
    - convert:         MindSpore → PyTorch weight conversion
"""

__version__ = "0.2.0"
__author__ = "Pratik Shah"
__paper__ = "CellFM (Zeng et al., 2024)"

from .config import Config80M, Config800M
from .model import CellFM, ERetNetBlock
from .lora import LoRALinear, apply_lora_to_model, get_lora_params
from .metrics import accuracy, precision_recall_f1, confusion_matrix, CellFMEvaluator
from .export import export_to_onnx

