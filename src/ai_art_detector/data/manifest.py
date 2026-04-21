"""Helpers for loading prepared dataset manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ai_art_detector.config import resolve_path


@dataclass(slots=True)
class ManifestSample:
    sample_id: str
    path: str
    relative_path: str
    label: int
    label_name: str
    source: str
    split: str
    file_size_bytes: int
    sha256: str
    width: int
    height: int
    extension: str

    @property
    def label_float(self) -> float:
        return float(self.label)


def load_manifest(manifest_path: str | Path) -> list[ManifestSample]:
    manifest_path = resolve_path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    samples: list[ManifestSample] = []
    for row in rows:
        samples.append(
            ManifestSample(
                sample_id=row["sample_id"],
                path=row["path"],
                relative_path=row["relative_path"],
                label=int(row["label"]),
                label_name=row["label_name"],
                source=row["source"],
                split=row["split"],
                file_size_bytes=int(row["file_size_bytes"]),
                sha256=row["sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                extension=row["extension"],
            )
        )
    return samples


def split_manifest(samples: list[ManifestSample]) -> dict[str, list[ManifestSample]]:
    split_map: dict[str, list[ManifestSample]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_map.setdefault(sample.split, []).append(sample)
    return split_map
