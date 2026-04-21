"""Image transform builders for training and evaluation."""

from __future__ import annotations

from ai_art_detector.config import DataConfig


def build_transforms(data_config: DataConfig):
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for image transforms. Install the `ml` extras."
        ) from exc

    normalize = transforms.Normalize(
        mean=data_config.normalization_mean or [0.485, 0.456, 0.406],
        std=data_config.normalization_std or [0.229, 0.224, 0.225],
    )

    train_ops = [
        transforms.RandomResizedCrop(
            data_config.image_size,
            scale=tuple(data_config.train_random_resized_crop_scale or [0.8, 1.0]),
        ),
        transforms.RandomHorizontalFlip(p=data_config.train_horizontal_flip_prob),
    ]

    color_jitter = data_config.train_color_jitter or {}
    if any(value > 0 for value in color_jitter.values()):
        train_ops.append(
            transforms.ColorJitter(
                brightness=color_jitter.get("brightness", 0.0),
                contrast=color_jitter.get("contrast", 0.0),
                saturation=color_jitter.get("saturation", 0.0),
                hue=color_jitter.get("hue", 0.0),
            )
        )

    train_ops.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if data_config.train_random_erasing_prob > 0:
        train_ops.append(transforms.RandomErasing(p=data_config.train_random_erasing_prob))

    eval_ops = [
        transforms.Resize(data_config.resize_size),
        transforms.CenterCrop(data_config.image_size),
        transforms.ToTensor(),
        normalize,
    ]
    return {
        "train": transforms.Compose(train_ops),
        "val": transforms.Compose(eval_ops),
        "test": transforms.Compose(eval_ops),
        "predict": transforms.Compose(eval_ops),
    }
