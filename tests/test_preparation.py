from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from ai_art_detector.config import load_experiment_config
from ai_art_detector.data.preparation import prepare_dataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24), color=color)
    image.save(path)


def test_prepare_dataset_builds_manifest_and_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_image(raw_dir / "source_alpha" / "human" / "human_1.png", (255, 0, 0))
    _write_image(raw_dir / "source_alpha" / "human" / "human_2.png", (255, 50, 50))
    _write_image(raw_dir / "source_alpha" / "ai" / "ai_1.png", (0, 255, 0))
    _write_image(raw_dir / "source_alpha" / "ai" / "ai_2.png", (0, 200, 0))
    _write_image(raw_dir / "source_beta" / "human" / "human_3.png", (0, 0, 255))
    _write_image(raw_dir / "source_beta" / "human" / "human_4.png", (30, 30, 255))
    _write_image(raw_dir / "source_beta" / "ai" / "ai_3.png", (255, 255, 0))
    _write_image(raw_dir / "source_beta" / "ai" / "ai_4.png", (200, 200, 0))

    config = load_experiment_config(Path("configs/experiment.yaml"))
    config.data.raw_dir = str(raw_dir)
    config.data.manifest_path = str(tmp_path / "processed" / "dataset_manifest.csv")
    config.data.summary_path = str(tmp_path / "processed" / "preparation_summary.json")
    config.runtime.outputs_dir = str(tmp_path / "artifacts")

    result = prepare_dataset(config)

    manifest_path = Path(result.manifest_path)
    summary_path = Path(result.summary_path)
    assert manifest_path.exists()
    assert summary_path.exists()
    assert Path(result.run_dir).exists()
    assert result.num_records == 8
    assert result.invalid_files == []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["label_name"] for row in rows} == {"human", "ai"}
    assert {row["source"] for row in rows} == {"source_alpha", "source_beta"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["num_records"] == 8
    assert summary["label_counts"] == {"ai": 4, "human": 4}
    assert summary["split_counts"]["train"] >= 1
