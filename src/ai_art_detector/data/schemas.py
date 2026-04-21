"""Typed records shared across dataset preparation modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ManifestRecord:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DataPreparationResult:
    manifest_path: str
    summary_path: str
    run_dir: str
    num_records: int
    invalid_files: list[str]
