from __future__ import annotations

from pathlib import Path

import torch

from ai_art_detector.config import load_experiment_config
from ai_art_detector.models.factory import create_model
from ai_art_detector.training.pipeline import _maybe_load_initial_checkpoint


def test_maybe_load_initial_checkpoint_restores_weights(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/experiment_smoke.yaml"))
    model = create_model(config.model)
    for parameter in model.parameters():
        parameter.data.fill_(0.25)

    checkpoint_path = tmp_path / "tiny_cnn_init.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    reloaded_model = create_model(config.model)
    for parameter in reloaded_model.parameters():
        parameter.data.zero_()

    config.model.checkpoint_path = str(checkpoint_path)
    initialization = _maybe_load_initial_checkpoint(torch, reloaded_model, config)

    assert initialization is not None
    assert initialization["checkpoint_path"] == str(checkpoint_path.resolve())
    first_parameter = next(reloaded_model.parameters())
    assert torch.allclose(first_parameter, torch.full_like(first_parameter, 0.25))
