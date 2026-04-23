"""Metrics, threshold tuning, and calibration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn import metrics as sk_metrics


@dataclass(slots=True)
class ThresholdSearchResult:
    metric_name: str
    threshold: float
    score: float


@dataclass(slots=True)
class TemperatureScalingResult:
    temperature: float
    pre_log_loss: float | None
    post_log_loss: float | None


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]] | None]:
    predictions = (probabilities >= threshold).astype(int)
    confusion = sk_metrics.confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()

    output: dict[str, float | int | list[list[int]] | None] = {
        "threshold": float(threshold),
        "accuracy": float(sk_metrics.accuracy_score(targets, predictions)),
        "precision": float(sk_metrics.precision_score(targets, predictions, zero_division=0)),
        "recall": float(sk_metrics.recall_score(targets, predictions, zero_division=0)),
        "f1": float(sk_metrics.f1_score(targets, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "balanced_accuracy": float(sk_metrics.balanced_accuracy_score(targets, predictions)),
        "brier_score": float(sk_metrics.brier_score_loss(targets, probabilities)),
        "confusion_matrix": confusion.astype(int).tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        output["roc_auc"] = float(sk_metrics.roc_auc_score(targets, probabilities))
    except ValueError:
        output["roc_auc"] = None

    try:
        output["average_precision"] = float(
            sk_metrics.average_precision_score(targets, probabilities)
        )
    except ValueError:
        output["average_precision"] = None

    try:
        output["log_loss"] = float(sk_metrics.log_loss(targets, probabilities, labels=[0, 1]))
    except ValueError:
        output["log_loss"] = None

    return output


def threshold_grid(
    minimum: float,
    maximum: float,
    steps: int,
) -> np.ndarray:
    if steps < 2:
        return np.array([minimum], dtype=float)
    return np.linspace(minimum, maximum, steps)


def tune_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    metric_name: str = "f1",
    minimum: float = 0.05,
    maximum: float = 0.95,
    steps: int = 37,
) -> ThresholdSearchResult:
    scorer_map = {
        "accuracy": lambda y_true, y_pred: sk_metrics.accuracy_score(y_true, y_pred),
        "precision": lambda y_true, y_pred: sk_metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda y_true, y_pred: sk_metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": lambda y_true, y_pred: sk_metrics.f1_score(y_true, y_pred, zero_division=0),
        "fbeta_2": lambda y_true, y_pred: sk_metrics.fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "balanced_accuracy": lambda y_true, y_pred: sk_metrics.balanced_accuracy_score(y_true, y_pred),
    }
    if metric_name not in scorer_map:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")

    best_threshold = 0.5
    best_score = -math.inf
    scorer = scorer_map[metric_name]
    for threshold in threshold_grid(minimum, maximum, steps):
        predictions = (probabilities >= threshold).astype(int)
        score = float(scorer(targets, predictions))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return ThresholdSearchResult(metric_name=metric_name, threshold=best_threshold, score=best_score)


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bucket_probs = probabilities[mask]
        bucket_targets = targets[mask]
        accuracy = bucket_targets.mean()
        confidence = bucket_probs.mean()
        ece += abs(confidence - accuracy) * (mask.sum() / len(probabilities))
    return float(ece)


def fit_temperature_scaling(
    targets: np.ndarray,
    logits: np.ndarray,
    minimum: float = 0.5,
    maximum: float = 5.0,
    steps: int = 46,
) -> TemperatureScalingResult:
    pre_probs = sigmoid(logits)
    try:
        pre_log_loss = float(sk_metrics.log_loss(targets, pre_probs, labels=[0, 1]))
    except ValueError:
        pre_log_loss = None

    best_temperature = 1.0
    best_log_loss = math.inf
    for temperature in threshold_grid(minimum, maximum, steps):
        scaled_probs = sigmoid(logits / temperature)
        try:
            current_log_loss = float(sk_metrics.log_loss(targets, scaled_probs, labels=[0, 1]))
        except ValueError:
            continue
        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_temperature = float(temperature)

    post_log_loss = None if best_log_loss is math.inf else float(best_log_loss)
    return TemperatureScalingResult(
        temperature=best_temperature,
        pre_log_loss=pre_log_loss,
        post_log_loss=post_log_loss,
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return sigmoid(logits / temperature)
