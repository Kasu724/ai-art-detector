"""Minimal `.env` loader for local development entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from ai_art_detector.config import PROJECT_ROOT


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()

    key, separator, value = stripped.partition("=")
    if not separator:
        return None

    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_project_env(env_path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    path = Path(env_path) if env_path is not None else (PROJECT_ROOT / ".env")
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue

        key, value = parsed
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded
