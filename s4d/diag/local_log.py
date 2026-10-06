"""Run logger: metrics.jsonl + log.txt + PNG/GIF/PLY mirror of every W&B panel."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


class RunLogger:
    def __init__(self, run_dir: str | Path, wandb_run=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb_run
        self._metrics = open(self.run_dir / "metrics.jsonl", "a")
        self._log = open(self.run_dir / "log.txt", "a")

    def text(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def scalars(self, step: int, values: dict[str, float]) -> None:
        record = {"step": int(step), **{k: float(v) for k, v in values.items()}}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def media(self, step: int, values: dict) -> None:
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def eval_dir(self, step: int) -> Path:
        d = self.run_dir / "eval" / f"step_{step:07d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def save_png(path: Path, image: np.ndarray) -> None:
        Image.fromarray(np.ascontiguousarray(image)).save(path)

    @staticmethod
    def save_gif(path: Path, frames: np.ndarray, fps: int = 8) -> None:
        """frames (T,H,W,3) uint8."""
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)

    @staticmethod
    def save_ply(path: Path, points: np.ndarray) -> None:
        """points (N,6) xyz + rgb(0-255)."""
        header = (
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(p[3])} {int(p[4])} {int(p[5])}\n")

    def close(self) -> None:
        self._metrics.close()
        self._log.close()
