from __future__ import annotations

from pathlib import Path

from ai_art_detector.config import load_experiment_config
from ai_art_detector.evaluation.pipeline import _resolve_eval_config


def test_resolve_eval_config_keeps_requested_dataset_with_checkpoint_model() -> None:
    anime_config = load_experiment_config(Path("configs/experiment_anime_social_transfer.yaml"))
    checkpoint_config = load_experiment_config(Path("configs/experiment_real_art_hf_improved.yaml"))

    resolved = _resolve_eval_config(anime_config, {"config": checkpoint_config.to_dict()})

    assert resolved.data.raw_dir == anime_config.data.raw_dir
    assert resolved.data.manifest_path == anime_config.data.manifest_path
    assert resolved.model.name == checkpoint_config.model.name
    assert resolved.eval.threshold_metric == anime_config.eval.threshold_metric
