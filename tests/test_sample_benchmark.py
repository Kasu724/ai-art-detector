from __future__ import annotations

from pathlib import Path

import pytest

from ai_art_detector.evaluation.sample_benchmark import expected_label_from_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path("sample/ai1.jpg"), "ai"),
        (Path("sample/fake_example.png"), "ai"),
        (Path("sample/generated_001.webp"), "ai"),
        (Path("sample/real1.jpg"), "human"),
        (Path("sample/human_example.png"), "human"),
        (Path("sample/ai/image.jpg"), "ai"),
        (Path("sample/human/image.jpg"), "human"),
    ],
)
def test_expected_label_from_path(path: Path, expected: str) -> None:
    assert expected_label_from_path(path) == expected


def test_expected_label_from_path_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Cannot infer expected label"):
        expected_label_from_path(Path("sample/unknown.jpg"))
