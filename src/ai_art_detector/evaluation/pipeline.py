"""Evaluation pipeline, prediction collection, and artifact writing."""

from __future__ import annotations

import logging
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_art_detector.config import ExperimentConfig, experiment_config_from_dict
from ai_art_detector.data.datasets import build_dataloaders
from ai_art_detector.evaluation.metrics import (
    apply_temperature,
    compute_binary_metrics,
    expected_calibration_error,
    fit_temperature_scaling,
    sigmoid,
    tune_threshold,
)
from ai_art_detector.evaluation.plots import (
    save_confusion_matrix,
    save_probability_histogram,
    save_reliability_diagram,
    save_roc_curve,
)
from ai_art_detector.models.factory import create_model
from ai_art_detector.tracking.experiment import create_run_context, record_stage_metadata
from ai_art_detector.utils.device import autocast_context, resolve_device
from ai_art_detector.utils.io import write_json, write_rows
from ai_art_detector.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictionOutputs:
    targets: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]
    paths: list[str]
    label_names: list[str]
    sources: list[str]
    split: str
    loss: float | None = None


@dataclass(slots=True)
class EvaluationResult:
    run_dir: str
    metrics_path: str
    predictions_path: str | None
    summary_path: str
    top_errors_path: str | None


def _get_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for evaluation. Install the `ml` extras."
        ) from exc
    return torch, nn


def predict_dataloader(
    model,
    dataloader,
    device: str,
    loss_fn=None,
    mixed_precision: bool = False,
    split_name: str = "unknown",
) -> PredictionOutputs:
    torch, _ = _get_torch()
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    paths: list[str] = []
    label_names: list[str] = []
    sources: list[str] = []
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True).unsqueeze(1)
            with autocast_context(device, enabled=mixed_precision):
                logits = model(images)
                loss = loss_fn(logits, targets) if loss_fn is not None else None

            all_logits.append(logits.detach().cpu().numpy().reshape(-1))
            all_targets.append(targets.detach().cpu().numpy().reshape(-1))
            sample_ids.extend(batch["sample_ids"])
            paths.extend(batch["paths"])
            label_names.extend(batch["label_names"])
            sources.extend(batch["sources"])
            if loss is not None:
                batch_size = images.shape[0]
                running_loss += float(loss.item()) * batch_size
                total_samples += batch_size

    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.array([])
    targets_array = np.concatenate(all_targets, axis=0).astype(int) if all_targets else np.array([], dtype=int)
    probabilities = sigmoid(logits_array)
    average_loss = (running_loss / total_samples) if total_samples else None
    return PredictionOutputs(
        targets=targets_array,
        logits=logits_array,
        probabilities=probabilities,
        sample_ids=sample_ids,
        paths=paths,
        label_names=label_names,
        sources=sources,
        split=split_name,
        loss=average_loss,
    )


def _load_checkpoint(checkpoint_path: str | Path):
    torch, _ = _get_torch()
    return torch.load(checkpoint_path, map_location="cpu")


def _resolve_eval_config(config: ExperimentConfig, checkpoint: dict[str, Any]) -> ExperimentConfig:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config:
        return experiment_config_from_dict(checkpoint_config)
    return config


def _load_model_from_checkpoint(config: ExperimentConfig, checkpoint_path: str | Path, device: str):
    torch, nn = _get_torch()
    checkpoint = _load_checkpoint(checkpoint_path)
    effective_config = _resolve_eval_config(config, checkpoint)
    checkpoint_model_config = replace(effective_config.model, pretrained=False, weights=None)
    model = create_model(checkpoint_model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    return model, criterion, checkpoint, effective_config


def _prediction_rows(outputs: PredictionOutputs, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, probability in enumerate(outputs.probabilities.tolist()):
        predicted_label = "ai" if probability >= threshold else "human"
        rows.append(
            {
                "sample_id": outputs.sample_ids[index],
                "path": outputs.paths[index],
                "source": outputs.sources[index],
                "true_label": outputs.label_names[index],
                "predicted_label": predicted_label,
                "probability_ai": probability,
                "confidence": max(probability, 1.0 - probability),
                "is_correct": outputs.label_names[index] == predicted_label,
            }
        )
    return rows


def _top_error_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    mistakes = [row for row in rows if not row["is_correct"]]
    mistakes.sort(key=lambda row: row["confidence"], reverse=True)
    return mistakes[:limit]


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split: str = "test",
) -> EvaluationResult:
    set_global_seed(
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        num_threads=config.runtime.num_threads,
    )
    device = resolve_device(config.runtime.device)
    model, criterion, checkpoint, effective_config = _load_model_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    dataloaders, _ = build_dataloaders(effective_config)
    if split not in dataloaders:
        raise ValueError(f"Split `{split}` not found in prepared manifest.")

    run_context = create_run_context(config=effective_config, stage="evaluation")
    outputs = predict_dataloader(
        model=model,
        dataloader=dataloaders[split],
        device=device,
        loss_fn=criterion,
        mixed_precision=effective_config.train.mixed_precision,
        split_name=split,
    )

    threshold = effective_config.eval.decision_threshold
    threshold_summary: dict[str, Any] = {"selected_threshold": threshold, "source": "config"}
    tuning_split_name = effective_config.eval.tune_threshold_on
    calibrated_probabilities = outputs.probabilities
    calibration_summary: dict[str, Any] | None = None

    if tuning_split_name in dataloaders and tuning_split_name != split:
        tuning_outputs = predict_dataloader(
            model=model,
            dataloader=dataloaders[tuning_split_name],
            device=device,
            loss_fn=criterion,
            mixed_precision=effective_config.train.mixed_precision,
            split_name=tuning_split_name,
        )

        if effective_config.eval.calibration_enabled and effective_config.eval.calibration_method == "temperature_scaling":
            temperature_result = fit_temperature_scaling(
                targets=tuning_outputs.targets,
                logits=tuning_outputs.logits,
                minimum=effective_config.eval.calibration_temperature_min,
                maximum=effective_config.eval.calibration_temperature_max,
                steps=effective_config.eval.calibration_temperature_steps,
            )
            calibrated_probabilities = apply_temperature(outputs.logits, temperature_result.temperature)
            calibration_summary = {
                "method": effective_config.eval.calibration_method,
                "temperature": temperature_result.temperature,
                "pre_log_loss": temperature_result.pre_log_loss,
                "post_log_loss": temperature_result.post_log_loss,
                "ece_before": expected_calibration_error(
                    tuning_outputs.targets,
                    tuning_outputs.probabilities,
                    bins=effective_config.eval.reliability_bins,
                ),
                "ece_after": expected_calibration_error(
                    tuning_outputs.targets,
                    apply_temperature(tuning_outputs.logits, temperature_result.temperature),
                    bins=effective_config.eval.reliability_bins,
                ),
            }
            tuning_probabilities = apply_temperature(tuning_outputs.logits, temperature_result.temperature)
        else:
            tuning_probabilities = tuning_outputs.probabilities

        threshold_result = tune_threshold(
            targets=tuning_outputs.targets,
            probabilities=tuning_probabilities,
            metric_name=effective_config.eval.threshold_metric,
            minimum=effective_config.eval.threshold_search_min,
            maximum=effective_config.eval.threshold_search_max,
            steps=effective_config.eval.threshold_search_steps,
        )
        threshold = threshold_result.threshold
        threshold_summary = {
            "selected_threshold": threshold_result.threshold,
            "metric_name": threshold_result.metric_name,
            "metric_value": threshold_result.score,
            "source": tuning_split_name,
        }

    metrics = compute_binary_metrics(outputs.targets, calibrated_probabilities, threshold=threshold)
    metrics["split"] = split
    metrics["calibrated"] = calibration_summary is not None
    metrics["ece"] = expected_calibration_error(
        outputs.targets,
        calibrated_probabilities,
        bins=effective_config.eval.reliability_bins,
    )

    metrics_path = run_context.run_dir / "metrics.json"
    predictions_path = run_context.run_dir / "predictions.csv" if effective_config.eval.save_predictions_csv else None
    top_errors_path = run_context.run_dir / "top_errors.csv" if effective_config.eval.save_error_analysis else None
    summary_path = run_context.run_dir / "evaluation_summary.json"

    prediction_rows = _prediction_rows(
        PredictionOutputs(
            targets=outputs.targets,
            logits=outputs.logits,
            probabilities=calibrated_probabilities,
            sample_ids=outputs.sample_ids,
            paths=outputs.paths,
            label_names=outputs.label_names,
            sources=outputs.sources,
            split=outputs.split,
            loss=outputs.loss,
        ),
        threshold=threshold,
    )
    if predictions_path is not None:
        write_rows(prediction_rows, predictions_path)
    if top_errors_path is not None:
        write_rows(_top_error_rows(prediction_rows, effective_config.eval.top_k_errors), top_errors_path)

    if effective_config.eval.save_curves and metrics.get("roc_auc") is not None:
        save_roc_curve(outputs.targets, calibrated_probabilities, run_context.run_dir / "roc_curve.png")
        save_confusion_matrix(metrics["confusion_matrix"], run_context.run_dir / "confusion_matrix.png")
        save_probability_histogram(
            calibrated_probabilities,
            outputs.targets,
            run_context.run_dir / "probability_histogram.png",
        )
        save_reliability_diagram(
            calibrated_probabilities,
            outputs.targets,
            run_context.run_dir / "reliability_diagram.png",
            bins=effective_config.eval.reliability_bins,
        )

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "threshold_summary": threshold_summary,
        "calibration_summary": calibration_summary,
        "best_training_metric": checkpoint.get("best_metric"),
    }
    write_json(payload, metrics_path)
    record_stage_metadata(run_context, payload, summary_filename="evaluation_summary.json")

    return EvaluationResult(
        run_dir=str(run_context.run_dir),
        metrics_path=str(metrics_path),
        predictions_path=str(predictions_path) if predictions_path is not None else None,
        summary_path=str(summary_path),
        top_errors_path=str(top_errors_path) if top_errors_path is not None else None,
    )
