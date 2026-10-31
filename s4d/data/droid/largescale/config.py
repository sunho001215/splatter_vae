"""Build the large-scale DROID datasets from a training config (``data.regime: droid_largescale``)."""

from __future__ import annotations

from s4d.config import get
from s4d.data.droid.largescale.dataset import DroidLargeScaleDataset


def dataset_kwargs(cfg: dict) -> dict:
    data = cfg["data"]
    missing = [k for k in ("manifest", "pointworld_root", "rlds_index_dir", "rlds_data_root") if not data.get(k)]
    if missing:
        raise ValueError(f"droid_largescale needs data.{', data.'.join(missing)}")
    return {
        "pointworld_root": data["pointworld_root"],
        "rlds_index_dir": data["rlds_index_dir"],
        "rlds_data_root": data["rlds_data_root"],
        "validation_percent": float(data.get("validation_percent", 2.0)),
        "stride": int(data.get("stride", 3)),
        "image_height": int(get(cfg, "model.encoder.image_height", 144)),
        "image_width": int(get(cfg, "model.encoder.image_width", 256)),
        "max_tracks": int(data.get("max_tracks", 4096)),
        "with_wrist": bool(data.get("with_wrist", True)),
        "gripper_offset_m": float(data.get("gripper_offset_m", -0.019)),
        "seed": int(get(cfg, "train.seed", 0)),
        "max_open_files": int(data.get("max_open_files", 8)),
    }


def build_datasets(cfg: dict) -> tuple[DroidLargeScaleDataset, DroidLargeScaleDataset]:
    kwargs = dataset_kwargs(cfg)
    manifest = cfg["data"]["manifest"]
    return (
        DroidLargeScaleDataset(manifest, split="train", **kwargs),
        DroidLargeScaleDataset(manifest, split="validation", with_eval=True, **kwargs),
    )
