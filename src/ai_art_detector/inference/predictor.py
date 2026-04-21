"""Inference helpers shared by the CLI, API, and demo surfaces."""

from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai_art_detector.config import ExperimentConfig, experiment_config_from_dict, resolve_path
from ai_art_detector.data.transforms import build_transforms
from ai_art_detector.evaluation.metrics import apply_temperature, sigmoid
from ai_art_detector.models.factory import create_model
from ai_art_detector.utils.device import resolve_device
from ai_art_detector.utils.io import read_json


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


@dataclass(slots=True)
class PredictionResult:
    predicted_label: str
    decision: str
    probability_ai: float
    probabilities: dict[str, float]
    confidence: float
    threshold: float
    model_name: str
    backend: str
    calibrated: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


class BasePredictor:
    backend: str = "base"

    def predict_bytes(self, payload: bytes) -> PredictionResult:
        raise NotImplementedError

    def predict_file(self, image_path: str | Path) -> PredictionResult:
        return self.predict_bytes(Path(image_path).read_bytes())


def _load_image_from_bytes(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload))
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Input is not a valid image file.") from exc


def _load_threshold_and_temperature(
    metrics_path: str | Path | None,
    threshold: float | None,
) -> tuple[float, float | None]:
    if metrics_path is None:
        return threshold if threshold is not None else 0.5, None

    payload = read_json(resolve_path(metrics_path))
    threshold_payload = payload.get("threshold_summary", {})
    selected_threshold = threshold if threshold is not None else threshold_payload.get("selected_threshold", 0.5)
    calibration_payload = payload.get("calibration_summary")
    temperature = calibration_payload.get("temperature") if calibration_payload else None
    return float(selected_threshold), (float(temperature) if temperature is not None else None)


class TorchImagePredictor(BasePredictor):
    backend = "torch"

    def __init__(
        self,
        model,
        transform,
        model_name: str,
        device: str,
        threshold: float = 0.5,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.temperature = temperature

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ExperimentConfig | None = None,
        metrics_path: str | Path | None = None,
        threshold: float | None = None,
        device: str = "auto",
    ) -> "TorchImagePredictor":
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch is required for checkpoint-based inference. Install the `ml` extras."
            ) from exc

        checkpoint_path = resolve_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if config is None:
            stored_config = checkpoint.get("config")
            if stored_config is None:
                raise ValueError("Checkpoint does not include config metadata.")
            config = experiment_config_from_dict(stored_config)

        checkpoint_model_config = replace(config.model, pretrained=False, weights=None)
        model = create_model(checkpoint_model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        resolved_device = resolve_device(device)
        model.to(resolved_device)
        model.eval()

        selected_threshold, temperature = _load_threshold_and_temperature(metrics_path, threshold)
        transform = build_transforms(config.data)["predict"]
        return cls(
            model=model,
            transform=transform,
            model_name=config.model.name,
            device=resolved_device,
            threshold=selected_threshold,
            temperature=temperature,
        )

    def predict_bytes(self, payload: bytes) -> PredictionResult:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch is required for checkpoint-based inference. Install the `ml` extras."
            ) from exc

        image = _load_image_from_bytes(payload)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).detach().cpu().numpy().reshape(-1)
        probability_ai = sigmoid(logits)[0]
        if self.temperature is not None:
            probability_ai = apply_temperature(logits, self.temperature)[0]
        predicted_label = "ai" if probability_ai >= self.threshold else "human"
        return PredictionResult(
            predicted_label=predicted_label,
            decision=predicted_label,
            probability_ai=float(probability_ai),
            probabilities={"human": float(1.0 - probability_ai), "ai": float(probability_ai)},
            confidence=float(max(probability_ai, 1.0 - probability_ai)),
            threshold=self.threshold,
            model_name=self.model_name,
            backend=self.backend,
            calibrated=self.temperature is not None,
        )


class OnnxImagePredictor(BasePredictor):
    backend = "onnx"

    def __init__(
        self,
        session,
        input_name: str,
        transform,
        model_name: str,
        threshold: float = 0.5,
        temperature: float | None = None,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.transform = transform
        self.model_name = model_name
        self.threshold = threshold
        self.temperature = temperature

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        config: ExperimentConfig,
        metrics_path: str | Path | None = None,
        threshold: float | None = None,
    ) -> "OnnxImagePredictor":
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required for ONNX inference. Install the `onnx` extras."
            ) from exc

        selected_threshold, temperature = _load_threshold_and_temperature(metrics_path, threshold)
        session = ort.InferenceSession(str(resolve_path(onnx_path)))
        input_name = session.get_inputs()[0].name
        transform = build_transforms(config.data)["predict"]
        return cls(
            session=session,
            input_name=input_name,
            transform=transform,
            model_name=config.model.name,
            threshold=selected_threshold,
            temperature=temperature,
        )

    def predict_bytes(self, payload: bytes) -> PredictionResult:
        image = _load_image_from_bytes(payload)
        tensor = self.transform(image).unsqueeze(0).numpy().astype(np.float32)
        logits = self.session.run(None, {self.input_name: tensor})[0].reshape(-1)
        probability_ai = sigmoid(logits)[0]
        if self.temperature is not None:
            probability_ai = apply_temperature(logits, self.temperature)[0]
        predicted_label = "ai" if probability_ai >= self.threshold else "human"
        return PredictionResult(
            predicted_label=predicted_label,
            decision=predicted_label,
            probability_ai=float(probability_ai),
            probabilities={"human": float(1.0 - probability_ai), "ai": float(probability_ai)},
            confidence=float(max(probability_ai, 1.0 - probability_ai)),
            threshold=self.threshold,
            model_name=self.model_name,
            backend=self.backend,
            calibrated=self.temperature is not None,
        )


def load_predictor(
    config: ExperimentConfig | None,
    checkpoint_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    threshold: float | None = None,
    device: str | None = None,
) -> BasePredictor:
    if onnx_path:
        if config is None:
            raise ValueError("A config is required for ONNX inference.")
        return OnnxImagePredictor.from_onnx(
            onnx_path=onnx_path,
            config=config,
            metrics_path=metrics_path,
            threshold=threshold,
        )

    if checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path if config is not None else None
    if checkpoint_path is None:
        raise ValueError("A checkpoint path is required for inference.")
    return TorchImagePredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        metrics_path=metrics_path,
        threshold=threshold,
        device=device or (config.runtime.device if config is not None else "auto"),
    )


def load_predictor_from_environment(config: ExperimentConfig | None = None) -> BasePredictor | None:
    checkpoint_path = os.getenv("AIAD_MODEL_PATH")
    onnx_path = os.getenv("AIAD_ONNX_PATH")
    metrics_path = os.getenv("AIAD_METRICS_PATH")
    threshold = os.getenv("AIAD_THRESHOLD")
    config_path = os.getenv("AIAD_CONFIG_PATH")

    if config is None and config_path:
        from ai_art_detector.config import load_experiment_config

        config = load_experiment_config(config_path)
    if checkpoint_path is None and onnx_path is None:
        return None

    return load_predictor(
        config=config,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        metrics_path=metrics_path,
        threshold=float(threshold) if threshold else None,
        device=os.getenv("AIAD_DEVICE", config.runtime.device),
    )
