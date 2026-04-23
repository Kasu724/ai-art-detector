"""Manifest-driven dataset preparation utilities."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ai_art_detector.config import ExperimentConfig, PROJECT_ROOT, resolve_path
from ai_art_detector.data.adapters import FolderDatasetAdapter
from ai_art_detector.data.manifests import write_json, write_manifest
from ai_art_detector.data.schemas import DataPreparationResult, ManifestRecord
from ai_art_detector.tracking.experiment import create_run_context, record_stage_metadata
from ai_art_detector.utils.filesystem import project_relative_path


def _compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_images(config: ExperimentConfig, raw_dir: Path) -> tuple[list[ManifestRecord], list[str]]:
    adapter = FolderDatasetAdapter(
        raw_dir=raw_dir,
        label_names=set((config.data.label_map or {}).keys()),
        allowed_extensions={extension.lower() for extension in config.data.allowed_extensions or []},
        follow_symlinks=config.data.follow_symlinks,
        max_files=config.data.max_files,
    )
    files = adapter.iter_files()

    records: list[ManifestRecord] = []
    invalid_files: list[str] = []
    for file_path in files:
        relative_to_raw = file_path.relative_to(raw_dir)
        inferred = adapter.infer_source_and_label(relative_to_raw)
        if inferred is None:
            continue

        source, label_name = inferred
        try:
            with Image.open(file_path) as image:
                image.load()
                width, height = image.size
        except (UnidentifiedImageError, OSError) as exc:
            if config.data.skip_invalid_images:
                invalid_files.append(f"{file_path}: {exc}")
                continue
            raise

        sha256 = _compute_sha256(file_path) if config.data.compute_sha256 else ""
        sample_id = sha256[:16] if sha256 else hashlib.md5(str(file_path).encode("utf-8")).hexdigest()[:16]
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                path=project_relative_path(file_path, PROJECT_ROOT),
                relative_path=relative_to_raw.as_posix(),
                label=config.data.label_map[label_name],
                label_name=label_name,
                source=source,
                split="",
                file_size_bytes=file_path.stat().st_size,
                sha256=sha256,
                width=width,
                height=height,
                extension=file_path.suffix.lower(),
            )
        )

    return records, invalid_files


def _largest_remainder_counts(split_ratios: dict[str, float], total_count: int) -> dict[str, int]:
    exact_counts = {split: ratio * total_count for split, ratio in split_ratios.items()}
    floor_counts = {split: math.floor(count) for split, count in exact_counts.items()}
    remainder = total_count - sum(floor_counts.values())
    ranked = sorted(
        split_ratios.keys(),
        key=lambda split: (exact_counts[split] - floor_counts[split], split),
        reverse=True,
    )
    for split in ranked[:remainder]:
        floor_counts[split] += 1
    return floor_counts


def assign_splits(
    records: list[ManifestRecord],
    split_ratios: dict[str, float],
    stratify_fields: list[str],
    seed: int,
) -> list[str]:
    total_ratio = sum(split_ratios.values())
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    active_fields = list(stratify_fields)
    ordered_splits = list(split_ratios.keys())
    while True:
        grouped_records: dict[tuple[str, ...], list[ManifestRecord]] = defaultdict(list)
        for record in records:
            group_key = (
                tuple(str(getattr(record, field)) for field in active_fields)
                if active_fields
                else ("all",)
            )
            record.split = ""
            grouped_records[group_key].append(record)

        rng = random.Random(seed)
        for group in grouped_records.values():
            rng.shuffle(group)
            split_counts = _largest_remainder_counts(split_ratios, len(group))

            cursor = 0
            for split in ordered_splits:
                next_cursor = cursor + split_counts[split]
                for record in group[cursor:next_cursor]:
                    record.split = split
                cursor = next_cursor

        assigned_counts = Counter(record.split for record in records)
        missing_splits = [split for split in ordered_splits if assigned_counts.get(split, 0) == 0]
        if not missing_splits or not active_fields:
            break
        active_fields = active_fields[:-1]
    return active_fields


def build_summary(
    records: list[ManifestRecord],
    invalid_files: list[str],
    config: ExperimentConfig,
    used_stratify_fields: list[str],
) -> dict:
    label_counts = Counter(record.label_name for record in records)
    source_counts = Counter(record.source for record in records)
    split_counts = Counter(record.split for record in records)
    split_label_counts: dict[str, dict[str, int]] = defaultdict(dict)

    for split in sorted(split_counts):
        split_records = [record for record in records if record.split == split]
        split_label_counts[split] = dict(Counter(record.label_name for record in split_records))

    return {
        "project_name": config.project.name,
        "experiment_name": config.project.experiment_name,
        "num_records": len(records),
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "split_label_counts": dict(split_label_counts),
        "invalid_files": invalid_files,
        "stratify_fields": used_stratify_fields,
        "positive_label": config.data.positive_label,
    }


def prepare_dataset(
    config: ExperimentConfig,
    raw_dir: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
) -> DataPreparationResult:
    raw_dir = resolve_path(raw_dir or config.data.raw_dir)
    manifest_path = resolve_path(manifest_path or config.data.manifest_path)
    summary_path = resolve_path(summary_path or config.data.summary_path)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    run_context = create_run_context(config=config, stage="data_preparation")
    records, invalid_files = _scan_images(config=config, raw_dir=raw_dir)
    if not records:
        raise ValueError(
            "No labeled images were found. Expected `data/raw/<label>/...` or "
            "`data/raw/<source>/<label>/...`."
        )

    used_stratify_fields = assign_splits(
        records=records,
        split_ratios=config.data.split_ratios or {},
        stratify_fields=config.data.stratify_fields or [],
        seed=config.runtime.seed,
    )

    write_manifest(records, manifest_path)
    summary = build_summary(
        records=records,
        invalid_files=invalid_files,
        config=config,
        used_stratify_fields=used_stratify_fields,
    )
    write_json(summary, summary_path)
    record_stage_metadata(run_context, summary, summary_filename="data_preparation_summary.json")

    return DataPreparationResult(
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        run_dir=str(run_context.run_dir),
        num_records=len(records),
        invalid_files=invalid_files,
    )
