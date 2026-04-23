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


def test_load_real_dataset_experiment_configs() -> None:
    baseline = load_experiment_config(PROJECT_ROOT / "configs/experiment_real_art_hf.yaml")
    improved = load_experiment_config(PROJECT_ROOT / "configs/experiment_real_art_hf_improved.yaml")
    anime = load_experiment_config(PROJECT_ROOT / "configs/experiment_anime_social_transfer.yaml")
    fanart = load_experiment_config(PROJECT_ROOT / "configs/experiment_anime_fanart_v2_transfer.yaml")
    fanart_v3 = load_experiment_config(PROJECT_ROOT / "configs/experiment_anime_fanart_v3_transfer.yaml")
    fanart_v4 = load_experiment_config(PROJECT_ROOT / "configs/experiment_anime_fanart_v4_transfer.yaml")

    assert baseline.project.experiment_name == "real_art_hf_resnet18"
    assert baseline.data.raw_dir == "data/raw/hf_art_images_ai_and_real"
    assert baseline.data.manifest_path == "data/processed/real_art_hf_manifest.csv"

    assert improved.project.experiment_name == "real_art_hf_efficientnet_b0"
    assert improved.model.name == "efficientnet_b0"
    assert improved.data.manifest_path == "data/processed/real_art_hf_manifest.csv"

    assert anime.project.experiment_name == "anime_social_transfer_efficientnet_b0"
    assert anime.data.raw_dir == "data/raw/anime_social_filter"
    assert anime.model.checkpoint_path is not None

    assert fanart.project.experiment_name == "anime_fanart_v2_transfer_efficientnet_b0"
    assert fanart.data.raw_dir == "data/raw/anime_fanart_filter_v2"
    assert fanart.model.checkpoint_path is not None

    assert fanart_v3.project.experiment_name == "anime_fanart_v3_transfer_efficientnet_b0"
    assert fanart_v3.data.raw_dir == "data/raw/anime_fanart_filter_v3"
    assert fanart_v3.model.checkpoint_path is not None

    assert fanart_v4.project.experiment_name == "anime_fanart_v4_recall_efficientnet_b0"
    assert fanart_v4.data.raw_dir == "data/raw/anime_fanart_filter_v4"
    assert fanart_v4.eval.threshold_metric == "fbeta_2"
    assert fanart_v4.model.checkpoint_path is not None
