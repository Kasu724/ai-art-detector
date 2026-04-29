"""Folder benchmark helpers for externally held-out sample images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ai_art_detector.config import ExperimentConfig, dump_config, resolve_path
from ai_art_detector.inference.predictor import load_predictor
from ai_art_detector.utils.io import write_json, write_rows

LABEL_ALIASES = {
    "ai": "ai",
    "fake": "ai",
    "generated": "ai",
    "human": "human",
    "real": "human",
}


@dataclass(slots=True)
class SampleBenchmarkResult:
    run_dir: str
    summary_path: str
    predictions_path: str
    accuracy: float
    correct: int
    total: int


def expected_label_from_path(path: Path) -> str:
    """Infer the benchmark label from a parent directory or filename prefix."""
    candidates = [path.parent.name.lower(), path.stem.lower()]
    for candidate in candidates:
        for prefix, label in LABEL_ALIASES.items():
            if candidate == prefix or candidate.startswith(prefix):
                return label
    raise ValueError(
        f"Cannot infer expected label for `{path}`. Use names beginning with "
        "`ai`, `fake`, `generated`, `human`, or `real`."
    )


def iter_benchmark_images(sample_dir: str | Path, allowed_extensions: Iterable[str]) -> list[Path]:
    sample_dir = resolve_path(sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {sample_dir}")
    if not sample_dir.is_dir():
        raise NotADirectoryError(f"Benchmark path is not a folder: {sample_dir}")

    extensions = {extension.lower() for extension in allowed_extensions}
    paths = [
        path
        for path in sample_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(paths)


def benchmark_sample_folder(
    *,
    config: ExperimentConfig,
    sample_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
    threshold: float | None = None,
) -> SampleBenchmarkResult:
    """Run inference on a held-out folder and write benchmark artifacts."""
    images = iter_benchmark_images(sample_dir, config.data.allowed_extensions or [])
    if not images:
        raise ValueError(f"No benchmark images found in `{sample_dir}`.")

    predictor = load_predictor(
        config=config,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        metrics_path=metrics_path,
        threshold=threshold,
    )

    rows = []
    correct = 0
    for image_path in images:
        expected_label = expected_label_from_path(image_path)
        prediction = predictor.predict_file(image_path)
        is_correct = prediction.predicted_label == expected_label
        correct += int(is_correct)
        rows.append(
            {
                "path": str(image_path),
                "filename": image_path.name,
                "expected_label": expected_label,
                "predicted_label": prediction.predicted_label,
                "probability_ai": prediction.probability_ai,
                "confidence": prediction.confidence,
                "threshold": prediction.threshold,
                "is_correct": is_correct,
            }
        )

    total = len(rows)
    accuracy = correct / total
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(
        Path(config.runtime.outputs_dir)
        / "benchmarks"
        / f"{timestamp}_{config.project.experiment_name}_sample"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = dump_config(config, run_dir / "resolved_config.yaml")

    predictions_path = write_rows(rows, run_dir / "predictions.csv")
    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sample_dir": str(resolve_path(sample_dir)),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "onnx_path": str(onnx_path) if onnx_path is not None else None,
        "threshold_override": threshold,
        "resolved_config_path": str(resolved_config_path),
        "predictions": rows,
    }
    summary_path = write_json(summary, run_dir / "benchmark_summary.json")

    return SampleBenchmarkResult(
        run_dir=str(run_dir),
        summary_path=str(summary_path),
        predictions_path=str(predictions_path),
        accuracy=accuracy,
        correct=correct,
        total=total,
    )

