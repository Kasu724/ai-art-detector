"""Dataset adapter layer for manifest preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FolderDatasetAdapter:
    """Interpret a local folder tree as labeled image samples."""

    raw_dir: Path
    label_names: set[str]
    allowed_extensions: set[str]
    follow_symlinks: bool = False
    max_files: int | None = None

    def iter_files(self) -> list[Path]:
        files = sorted(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.allowed_extensions
            and (self.follow_symlinks or not path.is_symlink())
        )
        if self.max_files is not None:
            files = files[: self.max_files]
        return files

    def infer_source_and_label(self, relative_path: Path) -> tuple[str, str] | None:
        parts = relative_path.parts
        if len(parts) < 2:
            return None

        if parts[0] in self.label_names:
            return ("default", parts[0])

        if len(parts) >= 3 and parts[1] in self.label_names:
            return (parts[0], parts[1])

        return None
