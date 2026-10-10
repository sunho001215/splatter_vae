"""Fail closed before any test imports torch or another GPU-aware package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from s4d.gpu_guard import enforce_allowed_gpus, validate_runtime_path  # noqa: E402

GPU_MAPPING = enforce_allowed_gpus()

import pytest  # noqa: E402
import torch  # noqa: E402


def pytest_configure(config):
    temp = os.environ.get("S4D_PYTEST_TMP") or config.option.basetemp or str(REPO / ".pytest_tmp")
    config.option.basetemp = str(validate_runtime_path(Path(temp)))


@pytest.fixture
def batch():
    """A small, fully valid training batch with independently translated cameras."""
    B, T, V, H, W = 2, 3, 2, 16, 16
    K = torch.tensor([[24.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]])
    K = K.expand(B, V, 3, 3).clone()
    c2w = torch.eye(4).expand(B, V, 4, 4).clone()
    c2w[:, 1, 0, 3] = 0.1
    w2c = torch.linalg.inv(c2w)
    return {
        "images": torch.zeros(B, T, V, 3, H, W, dtype=torch.uint8),
        "K": K,
        "w2c": w2c,
        "c2w": c2w,
        "depth": torch.ones(B, T, V, 1, H, W),
        "motion3d": torch.zeros(B, 3, V, 3, H, W),
        "motion_weight": torch.ones(B, 3, V, 1, H, W),
        "motion_score": torch.zeros(B, T, V, 1, H, W),
        "probe_state": torch.zeros(B, T, 11),
        "meta": {
            "task": ["hammer", "hammer"],
            "episode": ["ep000", "ep001"],
            "t_indices": [(0, 2, 4), (0, 4, 8)],
            "stride": [2, 4],
        },
    }


@pytest.fixture
def heldout_batch(batch):
    """Optional held-out fields appear together and retain the camera/time axes."""
    batch.update(
        {
            "eval_images": batch["images"][:, :, :1].clone(),
            "eval_K": batch["K"][:, :1].clone(),
            "eval_w2c": batch["w2c"][:, :1].clone(),
            "eval_depth": batch["depth"][:, :, :1].clone(),
        }
    )
    return batch
