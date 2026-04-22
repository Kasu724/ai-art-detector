"""Download supported real datasets into the repo's raw-data layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_art_detector.config import resolve_path
from ai_art_detector.utils.io import write_json

SUPPORTED_DATASETS = {
    "real_art_hf": {
        "dataset_id": "DataScienceProject/Art_Images_Ai_And_Real_",
        "label_map": {
            "fake": "ai",
            "real": "human",
        },
    }
}

ANIME_SOCIAL_SOURCES = {
    "human": [
        {
            "dataset_id": "sayurio/anime-art-image",
            "source_name": "sayurio_anime_art",
            "split": "train",
            "image_key": "image",
        },
        {
            "dataset_id": "Dhiraj45/Animes",
            "source_name": "dhiraj45_animes",
            "split": "train",
            "image_key": "image",
        },
    ],
    "ai": [
        {
            "dataset_id": "ShoukanLabs/OpenNiji-0_32237",
            "source_name": "open_niji_0_32237",
            "split": "train",
            "image_key": "image",
        },
        {
            "dataset_id": "ShoukanLabs/OpenNiji-65001_100000",
            "source_name": "open_niji_65001_100000",
            "split": "train",
            "image_key": "image",
        },
    ],
}

AI_TAG_EXCLUSIONS = {"ai-generated", "ai-assisted"}


@dataclass(slots=True)
class DownloadResult:
    dataset_id: str
    output_dir: str
    summary_path: str
    num_downloaded: int
    split_counts: dict[str, int]
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "output_dir": self.output_dir,
            "summary_path": self.summary_path,
            "num_downloaded": self.num_downloaded,
            "split_counts": self.split_counts,
            "label_counts": self.label_counts,
        }


def _require_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `datasets` package is required for real dataset download. "
            "Reinstall the project after pulling the updated dependencies."
        ) from exc
    return load_dataset


def _iter_dataset_splits(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        return list(dataset.items())
    raise TypeError(f"Unsupported dataset container: {type(dataset)!r}")


def _split_quota(total: int, num_sources: int, index: int) -> int:
    base = total // num_sources
    remainder = total % num_sources
    return base + (1 if index < remainder else 0)


def _save_rgb_image(image: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)


def _classification_score(payload: dict[str, Any] | None, label: str) -> float:
    if not payload:
        return 0.0
    labels = payload.get("label") or []
    scores = payload.get("score") or []
    for candidate, score in zip(labels, scores):
        if candidate == label:
            return float(score)
    return 0.0


def _is_danbooru_fanart_row(row: dict[str, Any]) -> bool:
    if str(row.get("rating", "")).lower() not in {"g", "s"}:
        return False
    if not row.get("artist_tags"):
        return False
    if not row.get("copyright_tags") and not row.get("character_tags"):
        return False
    all_tags = set(row.get("tags") or []) | set(row.get("general_tags") or []) | set(row.get("meta_tags") or [])
    if AI_TAG_EXCLUSIONS & all_tags:
        return False
    if _classification_score(row.get("safe_check_score"), "safe") < 0.8:
        return False
    if _classification_score(row.get("completeness_score"), "polished") < 0.85:
        return False
    if float(row.get("aesthetic_score") or 0.0) < 5.0:
        return False
    return True


def _is_danbooru_ai_row(row: dict[str, Any]) -> bool:
    if str(row.get("rating", "")).lower() not in {"g", "s"}:
        return False
    all_tags = set(row.get("tags") or []) | set(row.get("general_tags") or []) | set(row.get("meta_tags") or [])
    if not (AI_TAG_EXCLUSIONS & all_tags):
        return False
    if _classification_score(row.get("safe_check_score"), "safe") < 0.8:
        return False
    if _classification_score(row.get("completeness_score"), "polished") < 0.75:
        return False
    if float(row.get("aesthetic_score") or 0.0) < 4.5:
        return False
    return True


def _download_streaming_source(
    *,
    load_dataset,
    dataset_id: str,
    split: str,
    image_key: str,
    output_dir: Path,
    source_name: str,
    label_name: str,
    quota: int,
    row_filter=None,
) -> int:
    if quota <= 0:
        return 0

    dataset = load_dataset(dataset_id, streaming=True)
    if split not in dataset:
        raise ValueError(f"Split `{split}` not found in dataset `{dataset_id}`")

    label_dir = output_dir / source_name / label_name
    written = 0
    for row in dataset[split]:
        if row_filter is not None and not row_filter(row):
            continue
        image = row.get(image_key)
        if image is None:
            continue
        output_path = label_dir / f"{source_name}_{written:06d}.png"
        _save_rgb_image(image, output_path)
        written += 1
        if written >= quota:
            break
    return written


def download_real_art_dataset(
    output_dir: str | Path,
    max_per_split: int | None = None,
    dataset_key: str = "real_art_hf",
) -> dict[str, Any]:
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset key: {dataset_key}")

    load_dataset = _require_datasets()
    spec = SUPPORTED_DATASETS[dataset_key]
    dataset_id = spec["dataset_id"]
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / f"{output_dir.name}_download_summary.json"

    dataset = load_dataset(dataset_id, streaming=True)
    label_mapping = spec["label_map"]

    split_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {"ai": 0, "human": 0}
    total_downloaded = 0
    for split_name, split_dataset in _iter_dataset_splits(dataset):
        label_feature = split_dataset.features.get("label")
        label_names = getattr(label_feature, "names", None)
        split_total = 0
        limit = max_per_split
        for index, row in enumerate(split_dataset):
            if limit is not None and index >= limit:
                break

            raw_label = row["label"]
            label_name = label_names[raw_label] if label_names is not None else str(raw_label)
            mapped_label = label_mapping.get(str(label_name).lower())
            if mapped_label is None:
                raise ValueError(f"Unsupported label `{label_name}` in {dataset_id}")

            target_dir = output_dir / mapped_label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{split_name}_{index:06d}.png"
            if not target_path.exists():
                row["image"].convert("RGB").save(target_path)

            split_total += 1
            label_counts[mapped_label] += 1
            total_downloaded += 1

        split_counts[split_name] = split_total

    result = DownloadResult(
        dataset_id=dataset_id,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        num_downloaded=total_downloaded,
        split_counts=split_counts,
        label_counts=label_counts,
    )
    write_json(result.to_dict(), summary_path)
    return result.to_dict()


def download_anime_social_dataset(
    output_dir: str | Path,
    human_limit: int = 3000,
    ai_limit: int = 3000,
) -> dict[str, Any]:
    load_dataset = _require_datasets()
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / f"{output_dir.name}_download_summary.json"

    source_counts: dict[str, int] = {}
    label_counts = {"human": 0, "ai": 0}

    for label_name, source_specs in ANIME_SOCIAL_SOURCES.items():
        total_target = human_limit if label_name == "human" else ai_limit
        for index, spec in enumerate(source_specs):
            quota = _split_quota(total_target, len(source_specs), index)
            written = _download_streaming_source(
                load_dataset=load_dataset,
                dataset_id=spec["dataset_id"],
                split=spec["split"],
                image_key=spec["image_key"],
                output_dir=output_dir,
                source_name=spec["source_name"],
                label_name=label_name,
                quota=quota,
            )
            source_counts[spec["source_name"]] = written
            label_counts[label_name] += written

    result = {
        "dataset_family": "anime_social_filter",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "num_downloaded": label_counts["human"] + label_counts["ai"],
        "label_counts": label_counts,
        "source_counts": source_counts,
    }
    write_json(result, summary_path)
    return result


def download_anime_fanart_dataset(
    output_dir: str | Path,
    human_limit: int = 3000,
    ai_limit: int = 3000,
) -> dict[str, Any]:
    load_dataset = _require_datasets()
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / f"{output_dir.name}_download_summary.json"

    source_counts: dict[str, int] = {}
    label_counts = {"human": 0, "ai": 0}

    human_sources = [
        {
            "dataset_id": "ShinoharaHare/Danbooru-2024-Filtered-1M",
            "source_name": "danbooru_fanart",
            "split": "train",
            "image_key": "image",
            "row_filter": _is_danbooru_fanart_row,
        }
    ]
    ai_sources = ANIME_SOCIAL_SOURCES["ai"]

    for index, spec in enumerate(human_sources):
        quota = _split_quota(human_limit, len(human_sources), index)
        written = _download_streaming_source(
            load_dataset=load_dataset,
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            image_key=spec["image_key"],
            output_dir=output_dir,
            source_name=spec["source_name"],
            label_name="human",
            quota=quota,
            row_filter=spec.get("row_filter"),
        )
        source_counts[spec["source_name"]] = written
        label_counts["human"] += written

    for index, spec in enumerate(ai_sources):
        quota = _split_quota(ai_limit, len(ai_sources), index)
        written = _download_streaming_source(
            load_dataset=load_dataset,
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            image_key=spec["image_key"],
            output_dir=output_dir,
            source_name=spec["source_name"],
            label_name="ai",
            quota=quota,
        )
        source_counts[spec["source_name"]] = written
        label_counts["ai"] += written

    result = {
        "dataset_family": "anime_fanart_filter_v2",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "num_downloaded": label_counts["human"] + label_counts["ai"],
        "label_counts": label_counts,
        "source_counts": source_counts,
        "filters": {
            "human_rating": ["g", "s"],
            "exclude_tags": sorted(AI_TAG_EXCLUSIONS),
            "minimum_safe_score": 0.8,
            "minimum_polished_score": 0.85,
            "minimum_aesthetic_score": 5.0,
            "require_artist_tags": True,
            "require_character_or_copyright_tags": True,
        },
    }
    write_json(result, summary_path)
    return result


def download_anime_fanart_v3_dataset(
    output_dir: str | Path,
    human_limit: int = 3000,
    ai_limit: int = 3000,
) -> dict[str, Any]:
    load_dataset = _require_datasets()
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir.parent / f"{output_dir.name}_download_summary.json"

    source_counts: dict[str, int] = {}
    label_counts = {"human": 0, "ai": 0}

    human_sources = [
        {
            "dataset_id": "ShinoharaHare/Danbooru-2024-Filtered-1M",
            "source_name": "danbooru_fanart",
            "split": "train",
            "image_key": "image",
            "row_filter": _is_danbooru_fanart_row,
            "quota": human_limit,
        }
    ]
    ai_sources = [
        {
            "dataset_id": "ShoukanLabs/OpenNiji-0_32237",
            "source_name": "open_niji_0_32237",
            "split": "train",
            "image_key": "image",
            "quota": ai_limit // 3,
        },
        {
            "dataset_id": "ShoukanLabs/OpenNiji-32238_65000",
            "source_name": "open_niji_32238_65000",
            "split": "train",
            "image_key": "image",
            "quota": ai_limit // 3,
        },
        {
            "dataset_id": "ShoukanLabs/OpenNiji-65001_100000",
            "source_name": "open_niji_65001_100000",
            "split": "train",
            "image_key": "image",
            "quota": ai_limit - 2 * (ai_limit // 3),
        },
    ]

    for spec in human_sources:
        written = _download_streaming_source(
            load_dataset=load_dataset,
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            image_key=spec["image_key"],
            output_dir=output_dir,
            source_name=spec["source_name"],
            label_name="human",
            quota=spec["quota"],
            row_filter=spec.get("row_filter"),
        )
        source_counts[spec["source_name"]] = written
        label_counts["human"] += written

    for spec in ai_sources:
        written = _download_streaming_source(
            load_dataset=load_dataset,
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            image_key=spec["image_key"],
            output_dir=output_dir,
            source_name=spec["source_name"],
            label_name="ai",
            quota=spec["quota"],
            row_filter=spec.get("row_filter"),
        )
        source_counts[spec["source_name"]] = written
        label_counts["ai"] += written

    result = {
        "dataset_family": "anime_fanart_filter_v3",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "num_downloaded": label_counts["human"] + label_counts["ai"],
        "label_counts": label_counts,
        "source_counts": source_counts,
        "filters": {
            "human_filters": {
                "rating": ["g", "s"],
                "exclude_tags": sorted(AI_TAG_EXCLUSIONS),
                "minimum_safe_score": 0.8,
                "minimum_polished_score": 0.85,
                "minimum_aesthetic_score": 5.0,
                "require_artist_tags": True,
                "require_character_or_copyright_tags": True,
            },
            "ai_filters": {
                "source_mix": [
                    "ShoukanLabs/OpenNiji-0_32237",
                    "ShoukanLabs/OpenNiji-32238_65000",
                    "ShoukanLabs/OpenNiji-65001_100000",
                ],
            },
        },
    }
    write_json(result, summary_path)
    return result
