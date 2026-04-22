from __future__ import annotations

import os
from pathlib import Path

from ai_art_detector.utils.env import load_project_env


def test_load_project_env_sets_values_without_overriding_existing(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AIAD_MODEL_PATH=models/best.pt\nAIAD_DEVICE=cpu\n# comment\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AIAD_DEVICE", "cuda")
    monkeypatch.delenv("AIAD_MODEL_PATH", raising=False)

    loaded = load_project_env(env_path=env_path)

    assert loaded["AIAD_MODEL_PATH"] == "models/best.pt"
    assert loaded["AIAD_DEVICE"] == "cpu"
    assert os.environ["AIAD_MODEL_PATH"] == "models/best.pt"
    assert os.environ["AIAD_DEVICE"] == "cuda"
