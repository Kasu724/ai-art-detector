"""Synthetic dataset generation for pipeline smoke tests."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _human_like_image(size: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 242, 236))
    draw = ImageDraw.Draw(image)
    for _ in range(8):
        x0, y0 = rng.randint(0, size - 40), rng.randint(0, size - 40)
        x1, y1 = x0 + rng.randint(15, 60), y0 + rng.randint(15, 60)
        color = tuple(rng.randint(20, 220) for _ in range(3))
        if rng.random() < 0.5:
            draw.ellipse((x0, y0, x1, y1), fill=color, outline=None)
        else:
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=None)
    return image.filter(ImageFilter.GaussianBlur(radius=0.4))


def _ai_like_image(size: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for x in range(size):
        for y in range(size):
            base = int((x / max(size - 1, 1)) * 255)
            alt = int((y / max(size - 1, 1)) * 255)
            noise = rng.randint(-15, 15)
            pixels[x, y] = (
                max(0, min(255, base + noise)),
                max(0, min(255, 180 - base + noise)),
                max(0, min(255, alt + noise)),
            )
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x0, y0 = rng.randint(0, size - 30), rng.randint(0, size - 30)
        x1, y1 = x0 + rng.randint(10, 50), y0 + rng.randint(10, 50)
        outline = tuple(rng.randint(0, 255) for _ in range(3))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=rng.randint(2, 10), outline=outline, width=2)
    return image.filter(ImageFilter.DETAIL)


def generate_smoke_dataset(
    output_dir: str | Path,
    samples_per_class: int = 12,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    for label_name, generator in [("human", _human_like_image), ("ai", _ai_like_image)]:
        label_dir = output_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            image = generator(image_size, rng)
            image.save(label_dir / f"{label_name}_{index:03d}.png")
    return output_dir
