"""
CellFM Evaluation Metrics
===========================

LAYMAN EXPLANATION:
    After training an AI, you need to measure HOW GOOD it actually is.
    This file provides tools to evaluate model performance:

    1. ACCURACY: What fraction of cells did we classify correctly?
    2. F1 SCORE: A balanced measure that handles imbalanced datasets
       (e.g., when 90% of cells are one type and 10% are another)
    3. CONFUSION MATRIX: A table showing what the model predicted vs. reality
    4. PER-CLASS METRICS: How well does the model do for each cell type?

    Think of it like grading an exam:
        Accuracy = "How many questions did you get right overall?"
        F1 Score = "How well did you do on EACH topic?"
        Confusion Matrix = "Here's exactly where you made mistakes"

TECHNICAL DETAILS:
    Implements pure-PyTorch metrics (no sklearn dependency required):
    - accuracy(): Simple top-1 accuracy
    - precision_recall_f1(): Per-class and macro-averaged metrics
    - confusion_matrix(): N×N confusion matrix
    - CellFMEvaluator: High-level evaluator class with batched evaluation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Any


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1,
) -> float:
    """
    Compute top-k classification accuracy.

    Args:
        predictions: (N, C) logits or (N,) predicted class indices
        targets: (N,) true class indices
        top_k: Consider prediction correct if true label is in top-k predictions

    Returns:
        Accuracy as a float in [0, 1]

    Example:
        >>> logits = torch.randn(100, 10)
        >>> labels = torch.randint(0, 10, (100,))
        >>> acc = accuracy(logits, labels)
        >>> print(f"Accuracy: {acc:.2%}")
    """
    if predictions.dim() == 1:
        # Already class indices
        correct = (predictions == targets).float().sum()
        return (correct / targets.shape[0]).item()

    # predictions are logits: (N, C)
    if top_k == 1:
        pred_classes = predictions.argmax(dim=-1)
        correct = (pred_classes == targets).float().sum()
    else:
        _, top_indices = predictions.topk(top_k, dim=-1)
        correct = (top_indices == targets.unsqueeze(-1)).any(dim=-1).float().sum()

    return (correct / targets.shape[0]).item()


def precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Precision: Of all cells predicted as type X, how many actually ARE type X?
    Recall: Of all actual type X cells, how many did we FIND?
    F1: The harmonic mean of precision and recall (balanced metric).

    Args:
        predictions: (N, C) logits or (N,) predicted class indices
        targets: (N,) true class indices
        num_classes: Number of classes (auto-detected if None)
        average: 'macro' (unweighted mean across classes) or
                 'weighted' (weighted by class frequency)

    Returns:
        Dict with 'precision', 'recall', 'f1', and per-class breakdowns

    Example:
        >>> logits = torch.randn(100, 5)
        >>> labels = torch.randint(0, 5, (100,))
        >>> metrics = precision_recall_f1(logits, labels)
        >>> print(f"F1: {metrics['f1']:.4f}")
    """
    # Convert logits to class indices if needed
    if predictions.dim() > 1:
        pred_classes = predictions.argmax(dim=-1)
    else:
        pred_classes = predictions

    if num_classes is None:
        num_classes = max(pred_classes.max().item(), targets.max().item()) + 1

    # Compute per-class TP, FP, FN
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_support = []

    for c in range(num_classes):
        tp = ((pred_classes == c) & (targets == c)).float().sum()
        fp = ((pred_classes == c) & (targets != c)).float().sum()
        fn = ((pred_classes != c) & (targets == c)).float().sum()

        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        # Handle case where class has no support
        support = (targets == c).float().sum().item()
        if support == 0:
            precision = torch.tensor(0.0)
            recall = torch.tensor(0.0)
            f1 = torch.tensor(0.0)

        per_class_precision.append(precision.item())
        per_class_recall.append(recall.item())
        per_class_f1.append(f1.item())
        per_class_support.append(support)

    # Compute averaged metrics
    total_support = sum(per_class_support)

    if average == "macro":
        # Unweighted mean across classes with support > 0
        active_classes = [i for i, s in enumerate(per_class_support) if s > 0]
        if active_classes:
            avg_precision = np.mean([per_class_precision[i] for i in active_classes])
            avg_recall = np.mean([per_class_recall[i] for i in active_classes])
            avg_f1 = np.mean([per_class_f1[i] for i in active_classes])
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
    elif average == "weighted":
        # Weighted by class frequency
        if total_support > 0:
            weights = [s / total_support for s in per_class_support]
            avg_precision = sum(w * p for w, p in zip(weights, per_class_precision))
            avg_recall = sum(w * r for w, r in zip(weights, per_class_recall))
            avg_f1 = sum(w * f for w, f in zip(weights, per_class_f1))
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
    else:
        raise ValueError(f"Unknown average mode: {average}. Use 'macro' or 'weighted'.")

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
    }


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    The confusion matrix C is such that C[i, j] equals the number of
    observations known to be in class i but predicted to be in class j.

    Args:
        predictions: (N, C) logits or (N,) predicted class indices
        targets: (N,) true class indices
        num_classes: Number of classes (auto-detected if None)

    Returns:
        C: (num_classes, num_classes) numpy array

    Example:
        >>> logits = torch.randn(100, 5)
        >>> labels = torch.randint(0, 5, (100,))
        >>> cm = confusion_matrix(logits, labels)
        >>> print(cm.shape)  # (5, 5)
    """
    if predictions.dim() > 1:
        pred_classes = predictions.argmax(dim=-1)
    else:
        pred_classes = predictions

    if num_classes is None:
        num_classes = max(pred_classes.max().item(), targets.max().item()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets.cpu().numpy(), pred_classes.cpu().numpy()):
        cm[t, p] += 1

    return cm


class CellFMEvaluator:
    """
    High-level evaluation utility for CellFM models.

    Runs the model on a validation/test DataLoader and computes
    comprehensive evaluation metrics.

    Args:
        model (nn.Module): Trained CellFM model with a classification head
        device (str): Compute device
        label_names (list): Optional human-readable class names

    Example:
        >>> evaluator = CellFMEvaluator(model, device="cuda")
        >>> results = evaluator.evaluate(test_loader)
        >>> print(f"Test Accuracy: {results['accuracy']:.2%}")
        >>> print(f"Test F1: {results['f1']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        label_names: Optional[List[str]] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.label_names = label_names

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        average: str = "macro",
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader with cell data (must include 'label')
            average: Averaging strategy for multi-class metrics

        Returns:
            Dict with:
                - accuracy: Overall classification accuracy
                - precision, recall, f1: Averaged metrics
                - confusion_matrix: NxN confusion matrix
                - per_class_*: Per-class breakdowns
                - n_samples: Total number of samples evaluated
        """
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            gene_ids = batch['gene_ids'].to(self.device)
            gene_values = batch['gene_values'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(
                gene_ids, gene_values,
                key_padding_mask=padding_mask,
            )

            all_predictions.append(logits.cpu())
            all_targets.append(labels.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        acc = accuracy(all_predictions, all_targets)
        prf = precision_recall_f1(
            all_predictions, all_targets, average=average,
        )
        cm = confusion_matrix(all_predictions, all_targets)

        results = {
            "accuracy": acc,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "confusion_matrix": cm,
            "per_class_precision": prf["per_class_precision"],
            "per_class_recall": prf["per_class_recall"],
            "per_class_f1": prf["per_class_f1"],
            "per_class_support": prf["per_class_support"],
            "n_samples": all_targets.shape[0],
        }

        # Add label names if provided
        if self.label_names is not None:
            results["label_names"] = self.label_names

        return results

    def print_report(self, results: Dict[str, Any]):
        """
        Print a formatted classification report.

        Args:
            results: Output from self.evaluate()
        """
        print("=" * 60)
        print("  CellFM Evaluation Report")
        print("=" * 60)
        print(f"  Samples:    {results['n_samples']}")
        print(f"  Accuracy:   {results['accuracy']:.4f}")
        print(f"  Precision:  {results['precision']:.4f}")
        print(f"  Recall:     {results['recall']:.4f}")
        print(f"  F1 Score:   {results['f1']:.4f}")
        print("-" * 60)

        num_classes = len(results['per_class_f1'])
        label_names = results.get('label_names', [f"Class {i}" for i in range(num_classes)])

        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for i in range(num_classes):
            name = label_names[i] if i < len(label_names) else f"Class {i}"
            print(
                f"  {name:<20} "
                f"{results['per_class_precision'][i]:>10.4f} "
                f"{results['per_class_recall'][i]:>10.4f} "
                f"{results['per_class_f1'][i]:>10.4f} "
                f"{int(results['per_class_support'][i]):>10}"
            )

        print("=" * 60)
