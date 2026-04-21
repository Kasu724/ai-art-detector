"""Model factory for supported classification backbones."""

from __future__ import annotations

from typing import Any

from ai_art_detector.config import ModelConfig


def _build_tiny_cnn(config: ModelConfig):
    try:
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to build models. Install the `ml` extras."
        ) from exc

    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(config.dropout),
        nn.Linear(64, config.num_classes),
    )


def create_model(config: ModelConfig):
    try:
        import torch.nn as nn
        from torchvision import models
        from torchvision.models import (
            EfficientNet_B0_Weights,
            ResNet18_Weights,
        )
    except ModuleNotFoundError:
        if config.name == "tiny_cnn":
            return _build_tiny_cnn(config)
        raise ModuleNotFoundError(
            "torch and torchvision are required to build training models. "
            "Install the `ml` extras."
        )

    if config.name == "tiny_cnn":
        return _build_tiny_cnn(config)

    if config.name == "resnet18":
        weights = None
        if config.pretrained:
            weights = getattr(ResNet18_Weights, str(config.weights)) if config.weights else ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, config.num_classes),
        )
    elif config.name == "efficientnet_b0":
        weights = None
        if config.pretrained:
            weights = (
                getattr(EfficientNet_B0_Weights, str(config.weights))
                if config.weights
                else EfficientNet_B0_Weights.DEFAULT
            )
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, config.num_classes),
        )
    else:
        raise ValueError(f"Unsupported model: {config.name}")

    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        classifier = getattr(model, "fc", None) or getattr(model, "classifier", None)
        if classifier is not None:
            for parameter in classifier.parameters():
                parameter.requires_grad = True

    return model


def count_parameters(model: Any) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
