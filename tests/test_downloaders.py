from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ai_art_detector.data.downloaders import download_anime_social_dataset, download_real_art_dataset


class _DummyLabelFeature:
    names = ["fake", "real"]


class _DummySplit:
    features = {"label": _DummyLabelFeature()}

    def __iter__(self):
        yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0)), "label": 0}
        yield {"image": Image.new("RGB", (8, 8), color=(0, 0, 255)), "label": 1}


class _DummyImageSplit:
    def __iter__(self):
        yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0))}
        yield {"image": Image.new("RGB", (8, 8), color=(0, 0, 255))}
        yield {"image": Image.new("RGB", (8, 8), color=(0, 255, 0))}


def test_download_real_art_dataset_writes_images_and_summary(monkeypatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_id: str, streaming: bool = False):
        assert dataset_id == "DataScienceProject/Art_Images_Ai_And_Real_"
        assert streaming is True
        return {"train": _DummySplit(), "test": _DummySplit()}

    monkeypatch.setattr(
        "ai_art_detector.data.downloaders._require_datasets",
        lambda: fake_load_dataset,
    )

    output_dir = tmp_path / "raw" / "hf_art_images_ai_and_real"
    result = download_real_art_dataset(output_dir=output_dir)

    assert result["num_downloaded"] == 4
    assert result["split_counts"] == {"train": 2, "test": 2}
    assert result["label_counts"] == {"ai": 2, "human": 2}
    assert (output_dir / "ai" / "train_000000.png").exists()
    assert (output_dir / "human" / "train_000001.png").exists()
    assert Path(result["summary_path"]).exists()

    with Path(result["summary_path"]).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["dataset_id"] == "DataScienceProject/Art_Images_Ai_And_Real_"


def test_download_anime_social_dataset_writes_balanced_sources(monkeypatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_id: str, streaming: bool = False):
        assert streaming is True
        return {"train": _DummyImageSplit()}

    monkeypatch.setattr(
        "ai_art_detector.data.downloaders._require_datasets",
        lambda: fake_load_dataset,
    )

    output_dir = tmp_path / "raw" / "anime_social_filter"
    result = download_anime_social_dataset(output_dir=output_dir, human_limit=4, ai_limit=4)

    assert result["num_downloaded"] == 8
    assert result["label_counts"] == {"human": 4, "ai": 4}
    assert result["source_counts"]["sayurio_anime_art"] == 2
    assert result["source_counts"]["dhiraj45_animes"] == 2
    assert result["source_counts"]["open_niji_0_32237"] == 2
    assert result["source_counts"]["open_niji_65001_100000"] == 2
    assert (output_dir / "sayurio_anime_art" / "human" / "sayurio_anime_art_000000.png").exists()
    assert (output_dir / "open_niji_0_32237" / "ai" / "open_niji_0_32237_000000.png").exists()
