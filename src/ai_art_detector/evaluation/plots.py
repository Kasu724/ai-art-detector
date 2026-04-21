"""Plotting utilities for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn import metrics as sk_metrics


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save evaluation plots. Install the `ml` extras."
        ) from exc
    return plt


def save_confusion_matrix(confusion_matrix: list[list[int]], output_path: Path) -> Path:
    plt = _load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], labels=["Human", "AI"])
    axis.set_yticks([0, 1], labels=["Human", "AI"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    for row_idx, row in enumerate(confusion_matrix):
        for col_idx, value in enumerate(row):
            axis.text(col_idx, row_idx, str(value), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_roc_curve(targets: np.ndarray, probabilities: np.ndarray, output_path: Path) -> Path:
    plt = _load_matplotlib()
    fpr, tpr, _ = sk_metrics.roc_curve(targets, probabilities)
    auc_value = sk_metrics.auc(fpr, tpr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_probability_histogram(probabilities: np.ndarray, targets: np.ndarray, output_path: Path) -> Path:
    plt = _load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(probabilities[targets == 0], bins=20, alpha=0.7, label="Human", color="#1f77b4")
    axis.hist(probabilities[targets == 1], bins=20, alpha=0.7, label="AI", color="#ff7f0e")
    axis.set_xlabel("Predicted P(AI)")
    axis.set_ylabel("Count")
    axis.set_title("Probability Histogram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_reliability_diagram(
    probabilities: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    bins: int = 10,
) -> Path:
    plt = _load_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    prob_true, prob_pred = calibration_curve(targets, probabilities, n_bins=bins)
    axis.plot(prob_pred, prob_true, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Frequency")
    axis.set_title("Reliability Diagram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
