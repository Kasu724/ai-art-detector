"""Filesystem helpers shared across project modules."""

from __future__ import annotations

from pathlib import Path


def project_relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path.resolve())
