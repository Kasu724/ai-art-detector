"""Device selection utilities."""

from __future__ import annotations

from typing import Any


def resolve_device(requested_device: str = "auto") -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def autocast_context(device: str, enabled: bool) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        from contextlib import nullcontext

        return nullcontext()

    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    from contextlib import nullcontext

    return nullcontext()
