"""Training entry points and training loop implementation."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ai_art_detector.config import ExperimentConfig, resolve_path
from ai_art_detector.data.datasets import build_dataloaders
from ai_art_detector.evaluation.metrics import compute_binary_metrics
from ai_art_detector.evaluation.pipeline import predict_dataloader
from ai_art_detector.models.factory import count_parameters, create_model
from ai_art_detector.tracking.experiment import create_run_context, record_stage_metadata
from ai_art_detector.utils.device import autocast_context, resolve_device
from ai_art_detector.utils.io import write_json
from ai_art_detector.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingResult:
    run_dir: str
    best_checkpoint: str
    final_checkpoint: str
    history_path: str
    summary_path: str


def _get_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for training. Install the `ml` extras."
        ) from exc
    return torch, nn, optim


def _compute_pos_weight(train_samples: list[Any]) -> float:
    positives = sum(sample.label for sample in train_samples)
    negatives = max(len(train_samples) - positives, 1)
    positives = max(positives, 1)
    return float(negatives / positives)


def _build_optimizer(optim, model, config: ExperimentConfig):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_name = config.train.optimizer.lower()
    if optimizer_name == "adamw":
        return optim.AdamW(
            params,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    if optimizer_name == "sgd":
        return optim.SGD(
            params,
            lr=config.train.learning_rate,
            momentum=0.9,
            weight_decay=config.train.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.train.optimizer}")


def _build_scheduler(optim, optimizer, config: ExperimentConfig):
    scheduler_name = config.train.scheduler.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.train.epochs, 1),
            eta_min=config.train.min_learning_rate,
        )
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(config.train.epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {config.train.scheduler}")


def _write_history_csv(history: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in history for key in record.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return output_path


def _save_checkpoint(
    path: Path,
    torch,
    model,
    optimizer,
    scheduler,
    epoch: int,
    config: ExperimentConfig,
    best_metric: float,
    training_history: list[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config.to_dict(),
            "best_metric": best_metric,
            "history": training_history,
        },
        path,
    )
    return path


def _save_training_curve(history: list[dict[str, Any]], output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    epochs = [record["epoch"] for record in history]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [record["train_loss"] for record in history], label="Train Loss")
    axis.plot(epochs, [record["val_loss"] for record in history], label="Val Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training Curve")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _maybe_load_initial_checkpoint(torch, model, config: ExperimentConfig) -> dict[str, Any] | None:
    checkpoint_path = config.model.checkpoint_path
    if not checkpoint_path:
        return None

    resolved_path = resolve_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint `{resolved_path}` does not contain model weights.")

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = sorted(load_result.missing_keys)
    unexpected_keys = sorted(load_result.unexpected_keys)
    LOGGER.info(
        "Warm-started model from %s | missing_keys=%s unexpected_keys=%s",
        resolved_path,
        len(missing_keys),
        len(unexpected_keys),
    )
    return {
        "checkpoint_path": str(resolved_path),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def train_model(config: ExperimentConfig) -> TrainingResult:
    torch, nn, optim = _get_torch()
    set_global_seed(
        seed=config.runtime.seed,
        deterministic=config.runtime.deterministic,
        num_threads=config.runtime.num_threads,
    )
    device = resolve_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)

    dataloaders, split_samples = build_dataloaders(config)
    if not dataloaders.get("train") or not dataloaders.get("val"):
        raise ValueError("Training requires both `train` and `val` splits in the manifest.")

    run_context = create_run_context(config=config, stage="training")
    checkpoints_dir = run_context.run_dir / "checkpoints"
    history_path = run_context.run_dir / "history.json"
    history_csv_path = run_context.run_dir / "history.csv"
    summary_path = run_context.run_dir / "training_summary.json"

    model = create_model(config.model).to(device)
    initialization = _maybe_load_initial_checkpoint(torch, model, config)
    parameter_counts = count_parameters(model)
    optimizer = _build_optimizer(optim, model, config)
    scheduler = _build_scheduler(optim, optimizer, config)

    pos_weight = None
    if config.train.use_pos_weight:
        pos_weight_value = _compute_pos_weight(split_samples["train"])
        pos_weight = torch.tensor([pos_weight_value], device=device)
        LOGGER.info("Using positive class weight %.4f", pos_weight_value)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.train.mixed_precision and device.startswith("cuda"),
    )

    history: list[dict[str, Any]] = []
    best_metric_name = config.train.checkpoint_metric
    best_metric = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for step, batch in enumerate(dataloaders["train"], start=1):
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=config.train.mixed_precision):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            if config.train.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            if step % config.train.log_interval == 0:
                LOGGER.info(
                    "Epoch %s Step %s | train_loss=%.4f",
                    epoch,
                    step,
                    running_loss / max(total_samples, 1),
                )

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(total_samples, 1)
        val_predictions = predict_dataloader(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            loss_fn=criterion,
            mixed_precision=config.train.mixed_precision,
        )
        val_metrics = compute_binary_metrics(
            targets=val_predictions.targets,
            probabilities=val_predictions.probabilities,
            threshold=config.eval.decision_threshold,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_predictions.loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "confusion_matrix"},
        }
        history.append(epoch_record)
        LOGGER.info(
            "Epoch %s complete | train_loss=%.4f val_loss=%.4f %s=%.4f",
            epoch,
            train_loss,
            val_predictions.loss or float("nan"),
            best_metric_name,
            float(val_metrics.get(best_metric_name) or 0.0),
        )

        current_metric = float(val_metrics.get(best_metric_name) or float("-inf"))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoints_dir / "best.pt",
                torch,
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                best_metric,
                history,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.train.early_stopping_patience:
            LOGGER.info("Early stopping triggered at epoch %s", epoch)
            break

    final_checkpoint = _save_checkpoint(
        checkpoints_dir / "final.pt",
        torch,
        model,
        optimizer,
        scheduler,
        epoch=history[-1]["epoch"],
        config=config,
        best_metric=best_metric,
        training_history=history,
    )

    write_json({"history": history}, history_path)
    _write_history_csv(history, history_csv_path)
    _save_training_curve(history, run_context.run_dir / "training_curve.png")

    summary = {
        "best_checkpoint": str(checkpoints_dir / "best.pt"),
        "final_checkpoint": str(final_checkpoint),
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric,
        "parameter_counts": parameter_counts,
        "initialization": initialization,
        "num_train_samples": len(split_samples["train"]),
        "num_val_samples": len(split_samples["val"]),
        "history_length": len(history),
    }
    record_stage_metadata(run_context, summary, summary_filename="training_summary.json")
    return TrainingResult(
        run_dir=str(run_context.run_dir),
        best_checkpoint=str(checkpoints_dir / "best.pt"),
        final_checkpoint=str(final_checkpoint),
        history_path=str(history_path),
        summary_path=str(summary_path),
    )
