from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ai_art_detector.data.downloaders import (
    download_anime_fanart_dataset,
    download_anime_fanart_v3_dataset,
    download_anime_fanart_v4_dataset,
    download_anime_social_dataset,
    download_real_art_dataset,
)


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


def test_download_anime_fanart_dataset_uses_curated_human_source(monkeypatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_id: str, streaming: bool = False):
        assert streaming is True
        if dataset_id == "ShinoharaHare/Danbooru-2024-Filtered-1M":
            class _DanbooruSplit:
                def __iter__(self):
                    base = {
                        "rating": "s",
                        "artist_tags": ["artist_name"],
                        "character_tags": ["miku"],
                        "copyright_tags": ["vocaloid"],
                        "tags": ["1girl"],
                        "general_tags": ["solo"],
                        "meta_tags": ["highres"],
                        "safe_check_score": {"label": ["polluted", "safe"], "score": [0.05, 0.95]},
                        "completeness_score": {"label": ["polished", "rough", "monochrome"], "score": [0.97, 0.02, 0.01]},
                        "aesthetic_score": 6.5,
                    }
                    for _ in range(4):
                        yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0)), **base}

            return {"train": _DanbooruSplit()}
        return {"train": _DummyImageSplit()}

    monkeypatch.setattr(
        "ai_art_detector.data.downloaders._require_datasets",
        lambda: fake_load_dataset,
    )

    output_dir = tmp_path / "raw" / "anime_fanart_filter_v2"
    result = download_anime_fanart_dataset(output_dir=output_dir, human_limit=2, ai_limit=4)

    assert result["num_downloaded"] == 6
    assert result["label_counts"] == {"human": 2, "ai": 4}
    assert result["source_counts"]["danbooru_fanart"] == 2
    assert (output_dir / "danbooru_fanart" / "human" / "danbooru_fanart_000000.png").exists()


def test_download_anime_fanart_v3_dataset_includes_broader_ai_source_mix(monkeypatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_id: str, streaming: bool = False):
        assert streaming is True
        if dataset_id == "ShinoharaHare/Danbooru-2024-Filtered-1M":
            class _DanbooruSplit:
                def __iter__(self):
                    human = {
                        "rating": "s",
                        "artist_tags": ["artist_name"],
                        "character_tags": ["miku"],
                        "copyright_tags": ["vocaloid"],
                        "tags": ["1girl"],
                        "general_tags": ["solo"],
                        "meta_tags": ["highres"],
                        "safe_check_score": {"label": ["polluted", "safe"], "score": [0.05, 0.95]},
                        "completeness_score": {"label": ["polished", "rough", "monochrome"], "score": [0.97, 0.02, 0.01]},
                        "aesthetic_score": 6.5,
                    }
                    for _ in range(4):
                        yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0)), **human}

            return {"train": _DanbooruSplit()}
        return {"train": _DummyImageSplit()}

    monkeypatch.setattr(
        "ai_art_detector.data.downloaders._require_datasets",
        lambda: fake_load_dataset,
    )

    output_dir = tmp_path / "raw" / "anime_fanart_filter_v3"
    result = download_anime_fanart_v3_dataset(output_dir=output_dir, human_limit=2, ai_limit=4)

    assert result["num_downloaded"] == 6
    assert result["label_counts"] == {"human": 2, "ai": 4}
    assert result["source_counts"]["danbooru_fanart"] == 2
    assert result["source_counts"]["open_niji_0_32237"] >= 1
    assert result["source_counts"]["open_niji_32238_65000"] >= 1
    assert (output_dir / "open_niji_32238_65000" / "ai").exists()


def test_download_anime_fanart_v4_dataset_adds_auxiliary_ghibli_sources(monkeypatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_id: str, streaming: bool = False):
        assert streaming is True
        if dataset_id == "ShinoharaHare/Danbooru-2024-Filtered-1M":
            class _DanbooruSplit:
                def __iter__(self):
                    human = {
                        "rating": "s",
                        "artist_tags": ["artist_name"],
                        "character_tags": ["miku"],
                        "copyright_tags": ["vocaloid"],
                        "tags": ["1girl"],
                        "general_tags": ["solo"],
                        "meta_tags": ["highres"],
                        "safe_check_score": {"label": ["polluted", "safe"], "score": [0.05, 0.95]},
                        "completeness_score": {"label": ["polished", "rough", "monochrome"], "score": [0.97, 0.02, 0.01]},
                        "aesthetic_score": 6.5,
                    }
                    for _ in range(8):
                        yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0)), **human}

            return {"train": _DanbooruSplit()}
        if dataset_id == "pulnip/ghibli-dataset":
            class _GhibliSplit:
                def __iter__(self):
                    yield {"image": Image.new("RGB", (8, 8), color=(255, 0, 0)), "label": "real"}
                    yield {"image": Image.new("RGB", (8, 8), color=(0, 255, 0)), "label": "KappaNeuro"}
                    yield {"image": Image.new("RGB", (8, 8), color=(0, 0, 255)), "label": "nitrosocke"}

            return {"train": _GhibliSplit()}
        return {"train": _DummyImageSplit()}

    monkeypatch.setattr(
        "ai_art_detector.data.downloaders._require_datasets",
        lambda: fake_load_dataset,
    )

    output_dir = tmp_path / "raw" / "anime_fanart_filter_v4"
    result = download_anime_fanart_v4_dataset(output_dir=output_dir, human_limit=5, ai_limit=8)

    assert result["label_counts"]["human"] >= 5
    assert result["label_counts"]["ai"] >= 6
    assert result["source_counts"]["pulnip_ghibli_real"] == 1
    assert result["source_counts"]["pulnip_ghibli_ai"] >= 1
    assert (output_dir / "pulnip_ghibli_ai" / "ai").exists()
