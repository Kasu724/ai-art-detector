"""CSV and JSON helpers for dataset manifest artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_art_detector.data.schemas import ManifestRecord


def write_manifest(records: list[ManifestRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(ManifestRecord.__dataclass_fields__.keys())

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    return output_path


def write_json(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
