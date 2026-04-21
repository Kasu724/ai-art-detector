"""Lightweight local experiment tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_art_detector.config import ExperimentConfig, dump_config, resolve_path
from ai_art_detector.data.manifests import write_json


@dataclass(slots=True)
class RunContext:
    stage: str
    run_name: str
    run_dir: Path
    resolved_config_path: Path


def create_run_context(config: ExperimentConfig, stage: str) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{config.runtime.run_name_suffix}" if config.runtime.run_name_suffix else ""
    run_name = f"{timestamp}_{config.project.experiment_name}{suffix}"
    run_dir = resolve_path(Path(config.runtime.outputs_dir) / stage / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "resolved_config.yaml"
    if config.runtime.save_resolved_config:
        dump_config(config, resolved_config_path)

    return RunContext(
        stage=stage,
        run_name=run_name,
        run_dir=run_dir,
        resolved_config_path=resolved_config_path,
    )


def record_stage_metadata(
    run_context: RunContext,
    payload: dict,
    summary_filename: str = "summary.json",
) -> Path:
    return write_json(payload, run_context.run_dir / summary_filename)
