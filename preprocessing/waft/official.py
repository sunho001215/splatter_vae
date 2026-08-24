from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

WAFT_PREPROCESSING_VERSION = "droid-rlds-forward-flow-uncertainty-v2"


def load_official_waft(
    repository: str,
    config_path: str,
    checkpoint: str,
    depth_checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    root = Path(repository).expanduser().resolve()
    if not (root / "model" / "waft_a1.py").is_file():
        raise FileNotFoundError(
            f"Official WAFT was not found at {root}; clone https://github.com/princeton-vl/WAFT."
        )
    config_file = Path(config_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint).expanduser().resolve()
    depth_file = Path(depth_checkpoint).expanduser().resolve()
    for path in (config_file, checkpoint_file, depth_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model import fetch_model  # type: ignore

    values = json.loads(config_file.read_text(encoding="utf-8"))
    arguments = argparse.Namespace(**values)
    if getattr(arguments, "algorithm", None) != "waft-a1":
        raise ValueError(
            "The official WAFT configuration must select algorithm='waft-a1'."
        )
    previous = Path.cwd()
    os.chdir(depth_file.parent.parent)
    try:
        model = fetch_model(arguments)
    finally:
        os.chdir(previous)
    state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"WAFT checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    return model.requires_grad_(False).eval().to(device)


def make_waft_predictor(
    model: torch.nn.Module,
    device: torch.device,
    *,
    iterations: int | None = None,
) -> Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    @torch.inference_mode()
    def predict(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if first.shape != second.shape or first.ndim != 4 or first.shape[-1] != 3:
            raise ValueError("WAFT batches must align as (B,H,W,3).")
        image1 = torch.from_numpy(np.ascontiguousarray(first)).permute(0, 3, 1, 2)
        image2 = torch.from_numpy(np.ascontiguousarray(second)).permute(0, 3, 1, 2)
        output = model(
            image1.to(device=device, dtype=torch.float32),
            image2.to(device=device, dtype=torch.float32),
            iters=iterations,
        )
        predictions = output.get("flow")
        if not isinstance(predictions, (list, tuple)) or not predictions:
            raise RuntimeError("Official WAFT did not return a nonempty flow pyramid.")
        information = output.get("info")
        if not isinstance(information, (list, tuple)) or not information:
            raise RuntimeError(
                "Official WAFT did not return its predictive flow distribution."
            )
        info = information[-1].float()
        raw_log_scale = info[:, 2:]
        mixture_weight = info[:, :2].softmax(dim=1)
        minimum = float(model.args.var_min)
        maximum = float(model.args.var_max)
        log_scale = torch.stack(
            (
                raw_log_scale[:, 0].clamp(min=0.0, max=maximum),
                raw_log_scale[:, 1].clamp(min=minimum, max=0.0),
            ),
            dim=1,
        )
        uncertainty = (log_scale * mixture_weight).sum(dim=1)
        confidence = torch.exp(-uncertainty).clamp(0.0, 1.0)
        return {
            "forward_flow": predictions[-1].permute(0, 2, 3, 1).float().cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "uncertainty": uncertainty.cpu().numpy(),
        }

    return predict
