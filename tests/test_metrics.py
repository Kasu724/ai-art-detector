from __future__ import annotations

import numpy as np

from ai_art_detector.evaluation.comparison import compare_evaluation_runs
from ai_art_detector.evaluation.metrics import (
    apply_temperature,
    compute_binary_metrics,
    expected_calibration_error,
    fit_temperature_scaling,
    tune_threshold,
)
from ai_art_detector.utils.io import read_json, write_json


def test_binary_metrics_include_confusion_values() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(targets, probabilities, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_tuning_finds_reasonable_boundary() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.4, 0.6, 0.9])
    result = tune_threshold(targets, probabilities, metric_name="f1", minimum=0.1, maximum=0.9, steps=9)

    assert 0.4 <= result.threshold <= 0.6
    assert result.score >= 0.99


def test_temperature_scaling_runs_and_returns_positive_temperature() -> None:
    targets = np.array([0, 0, 1, 1])
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    result = fit_temperature_scaling(targets, logits)
    calibrated = apply_temperature(logits, result.temperature)

    assert result.temperature > 0
    assert calibrated.shape == logits.shape
    assert 0.0 <= expected_calibration_error(targets, calibrated) <= 1.0


def test_compare_runs_writes_summary(tmp_path) -> None:
    first = tmp_path / "metrics_a.json"
    second = tmp_path / "metrics_b.json"
    write_json(
        {
            "checkpoint_path": "run_a.pt",
            "metrics": {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.9,
                "f1": 0.82,
                "roc_auc": 0.88,
                "ece": 0.07,
                "threshold": 0.5,
                "calibrated": True,
            },
        },
        first,
    )
    write_json(
        {
            "checkpoint_path": "run_b.pt",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.8,
                "recall": 0.88,
                "f1": 0.84,
                "roc_auc": 0.9,
                "ece": 0.05,
                "threshold": 0.45,
                "calibrated": True,
            },
        },
        second,
    )

    output_path = compare_evaluation_runs([first, second], tmp_path / "comparison.json")
    payload = read_json(output_path)
    assert len(payload["experiments"]) == 2
    assert output_path.with_suffix(".md").exists()
