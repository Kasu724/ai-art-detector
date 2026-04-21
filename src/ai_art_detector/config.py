"""Configuration loading and composition utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


def _detect_project_root() -> Path:
    search_roots = [Path.cwd(), *Path.cwd().parents]
    for base in search_roots:
        if (base / "pyproject.toml").exists() and (base / "src").exists():
            return base
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _detect_project_root()


@dataclass(slots=True)
class ProjectConfig:
    name: str = "ai-art-detector"
    experiment_name: str = "baseline_transfer_learning"
    description: str = ""


@dataclass(slots=True)
class DataConfig:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    manifest_path: str = "data/processed/dataset_manifest.csv"
    summary_path: str = "data/processed/preparation_summary.json"
    allowed_extensions: list[str] | None = None
    label_map: dict[str, int] | None = None
    positive_label: str = "ai"
    split_ratios: dict[str, float] | None = None
    stratify_fields: list[str] | None = None
    compute_sha256: bool = True
    skip_invalid_images: bool = True
    follow_symlinks: bool = False
    max_files: int | None = None
    image_size: int = 224
    resize_size: int = 256
    normalization_mean: list[float] | None = None
    normalization_std: list[float] | None = None
    train_random_resized_crop_scale: list[float] | None = None
    train_horizontal_flip_prob: float = 0.5
    train_color_jitter: dict[str, float] | None = None
    train_random_erasing_prob: float = 0.0
    num_channels: int = 3
    pin_memory: bool = True

    def __post_init__(self) -> None:
        if self.allowed_extensions is None:
            self.allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        if self.label_map is None:
            self.label_map = {"human": 0, "ai": 1}
        if self.split_ratios is None:
            self.split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        if self.stratify_fields is None:
            self.stratify_fields = ["label_name", "source"]
        if self.normalization_mean is None:
            self.normalization_mean = [0.485, 0.456, 0.406]
        if self.normalization_std is None:
            self.normalization_std = [0.229, 0.224, 0.225]
        if self.train_random_resized_crop_scale is None:
            self.train_random_resized_crop_scale = [0.8, 1.0]
        if self.train_color_jitter is None:
            self.train_color_jitter = {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.02,
            }


@dataclass(slots=True)
class ModelConfig:
    name: str = "resnet18"
    family: str = "torchvision"
    pretrained: bool = True
    weights: str = "IMAGENET1K_V1"
    num_classes: int = 1
    input_size: int = 224
    dropout: float = 0.2
    freeze_backbone: bool = False
    checkpoint_path: str | None = None
    onnx_path: str | None = None


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    mixed_precision: bool = True
    checkpoint_metric: str = "roc_auc"
    early_stopping_patience: int = 3
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = 1.0
    use_pos_weight: bool = True
    log_interval: int = 10


@dataclass(slots=True)
class EvalConfig:
    decision_threshold: float = 0.5
    threshold_metric: str = "f1"
    save_curves: bool = True
    save_error_analysis: bool = True
    calibration_method: str = "temperature_scaling"
    calibration_enabled: bool = True
    tune_threshold_on: str = "val"
    threshold_search_min: float = 0.05
    threshold_search_max: float = 0.95
    threshold_search_steps: int = 37
    save_predictions_csv: bool = True
    top_k_errors: int = 24
    calibration_temperature_min: float = 0.5
    calibration_temperature_max: float = 5.0
    calibration_temperature_steps: int = 46
    reliability_bins: int = 10


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    outputs_dir: str = "artifacts"
    log_level: str = "INFO"
    save_resolved_config: bool = True
    deterministic: bool = True
    num_threads: int | None = None
    run_name_suffix: str = ""


@dataclass(slots=True)
class ExperimentConfig:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    runtime: RuntimeConfig
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}, got {type(payload)!r}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


T = TypeVar("T")


def _build_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - field_names)
    if unknown:
        raise ValueError(f"Unknown fields for {cls.__name__}: {unknown}")
    return cls(**values)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = resolve_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    root_config = load_yaml(config_path)
    defaults = root_config.pop("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("The `defaults` section must be a mapping of config groups to names.")

    config_dir = config_path.parent
    sections: dict[str, dict[str, Any]] = {}
    for section_name in ["data", "model", "train", "eval", "runtime"]:
        base_section: dict[str, Any] = {}
        default_name = defaults.get(section_name)
        if default_name:
            section_path = config_dir / section_name / f"{default_name}.yaml"
            if not section_path.exists():
                raise FileNotFoundError(f"Missing config section: {section_path}")
            base_section = load_yaml(section_path)

        inline_overrides = root_config.get(section_name, {})
        if inline_overrides and not isinstance(inline_overrides, dict):
            raise ValueError(f"Inline config for `{section_name}` must be a mapping.")
        sections[section_name] = deep_merge(base_section, inline_overrides or {})

    project_config = _build_dataclass(ProjectConfig, root_config.get("project", {}))
    notes = root_config.get("notes", "")
    return ExperimentConfig(
        project=project_config,
        data=_build_dataclass(DataConfig, sections["data"]),
        model=_build_dataclass(ModelConfig, sections["model"]),
        train=_build_dataclass(TrainConfig, sections["train"]),
        eval=_build_dataclass(EvalConfig, sections["eval"]),
        runtime=_build_dataclass(RuntimeConfig, sections["runtime"]),
        notes=notes,
    )


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        project=_build_dataclass(ProjectConfig, payload.get("project", {})),
        data=_build_dataclass(DataConfig, payload.get("data", {})),
        model=_build_dataclass(ModelConfig, payload.get("model", {})),
        train=_build_dataclass(TrainConfig, payload.get("train", {})),
        eval=_build_dataclass(EvalConfig, payload.get("eval", {})),
        runtime=_build_dataclass(RuntimeConfig, payload.get("runtime", {})),
        notes=payload.get("notes", ""),
    )


def dump_config(config: ExperimentConfig, output_path: str | Path) -> Path:
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return output_path
