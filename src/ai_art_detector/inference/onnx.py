"""ONNX export helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

from ai_art_detector.config import ExperimentConfig, resolve_path
from ai_art_detector.models.factory import create_model
from ai_art_detector.utils.io import write_json


def export_checkpoint_to_onnx(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 18,
) -> Path:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for ONNX export. Install the `ml` extras."
        ) from exc

    checkpoint_path = resolve_path(checkpoint_path)
    output_path = resolve_path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stored_config = checkpoint.get("config")
    if stored_config:
        from ai_art_detector.config import experiment_config_from_dict

        config = experiment_config_from_dict(stored_config)

    checkpoint_model_config = replace(config.model, pretrained=False, weights=None)
    model = create_model(checkpoint_model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset_version,
    )

    write_json(
        {
            "checkpoint_path": str(checkpoint_path),
            "onnx_path": str(output_path),
            "opset_version": opset_version,
            "input_shape": [1, 3, config.data.image_size, config.data.image_size],
            "model_name": config.model.name,
        },
        output_path.with_suffix(".metadata.json"),
    )
    return output_path
