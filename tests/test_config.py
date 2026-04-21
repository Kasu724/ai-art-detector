from __future__ import annotations

from ai_art_detector.config import PROJECT_ROOT, load_experiment_config


def test_load_experiment_config_uses_section_defaults() -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs/experiment.yaml")

    assert config.project.name == "ai-art-detector"
    assert config.project.experiment_name == "baseline_transfer_learning"
    assert config.data.label_map == {"human": 0, "ai": 1}
    assert config.model.name == "resnet18"
    assert config.train.checkpoint_metric == "roc_auc"
    assert config.runtime.outputs_dir == "artifacts"
