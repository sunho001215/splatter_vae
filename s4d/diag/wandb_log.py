"""W&B initialisation and media wrappers (all optional; everything also lands on disk)."""

from __future__ import annotations

import os

import numpy as np

_ENABLED = False


def init_wandb(cfg: dict, run_name: str, project: str, enabled: bool, run_dir):
    global _ENABLED
    _ENABLED = enabled
    if not enabled:
        return None
    import wandb  # noqa: PLC0415

    from s4d.gpu_guard import HOST_NAME  # noqa: PLC0415

    mode = os.environ.get("WANDB_MODE", "online")
    return wandb.init(
        project=project,
        name=run_name,
        config=cfg,
        tags=[f"host={HOST_NAME}"],
        dir=str(run_dir),
        mode=mode,
        settings=wandb.Settings(start_method="thread"),
    )


def image(array: np.ndarray, caption: str = ""):
    if not _ENABLED:
        return None
    import wandb  # noqa: PLC0415

    return wandb.Image(array, caption=caption)


def video(frames: np.ndarray, fps: int = 8, caption: str = ""):
    """frames (T,H,W,3) uint8."""
    if not _ENABLED:
        return None
    import wandb  # noqa: PLC0415

    return wandb.Video(np.ascontiguousarray(frames.transpose(0, 3, 1, 2)), fps=fps, format="gif", caption=caption)


def object3d(points: np.ndarray, vectors: np.ndarray | None = None):
    """points (N,6) xyz+rgb; vectors (M,2,3) start/end."""
    if not _ENABLED:
        return None
    import wandb  # noqa: PLC0415

    payload = {"type": "lidar/beta", "points": points.astype(np.float32)}
    if vectors is not None and len(vectors):
        payload["vectors"] = vectors.astype(np.float32)
    return wandb.Object3D(payload)


def table(columns: list[str], rows: list[list]):
    if not _ENABLED:
        return None
    import wandb  # noqa: PLC0415

    return wandb.Table(columns=columns, data=rows)
