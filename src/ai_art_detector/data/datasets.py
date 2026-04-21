"""Torch datasets and dataloaders backed by the prepared manifest."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from ai_art_detector.config import ExperimentConfig, PROJECT_ROOT
from ai_art_detector.data.manifest import ManifestSample, load_manifest, split_manifest
from ai_art_detector.data.transforms import build_transforms

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset


class ManifestImageDataset:
    """Load images from a prepared manifest split."""

    def __init__(self, samples: list[ManifestSample], transform=None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = Path(sample.path)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            transformed = self.transform(image) if self.transform else image

        return {
            "image": transformed,
            "target": sample.label_float,
            "sample_id": sample.sample_id,
            "path": sample.path,
            "label_name": sample.label_name,
            "source": sample.source,
            "split": sample.split,
        }


def _collate_metadata(batch: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for dataloading. Install the `ml` extras."
        ) from exc

    images = torch.stack([item["image"] for item in batch])
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
    return {
        "images": images,
        "targets": targets,
        "sample_ids": [item["sample_id"] for item in batch],
        "paths": [item["path"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
        "sources": [item["source"] for item in batch],
        "splits": [item["split"] for item in batch],
    }


def build_dataloaders(config: ExperimentConfig):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for training and evaluation. Install the `ml` extras."
        ) from exc

    samples = load_manifest(config.data.manifest_path)
    splits = split_manifest(samples)
    transforms = build_transforms(config.data)

    datasets = {
        split: ManifestImageDataset(split_samples, transform=transforms[split])
        for split, split_samples in splits.items()
        if split in transforms
    }

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=split == "train",
            num_workers=config.train.num_workers,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
            collate_fn=_collate_metadata,
        )
        for split, dataset in datasets.items()
    }
    return dataloaders, splits
