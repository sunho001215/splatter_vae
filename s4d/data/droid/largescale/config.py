"""Build the large-scale DROID datasets from a training config (``data.regime: droid_largescale``)."""

from __future__ import annotations

from s4d.config import get
from s4d.data.droid.largescale.dataset import DroidLargeScaleDataset
from s4d.model.encoder import visible_count


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
        **crop_kwargs(cfg),
    }


def crop_kwargs(cfg: dict) -> dict:
    """``data.crop`` (square-crop profiles only; absent: the aspect-keeping resize, no extra keys): the crop settings,
    the encoder's patch size and its kept temporal-token count, which every crop must leave unpadded."""
    crop = get(cfg, "data.crop")
    if not crop:
        return {}
    if str(get(cfg, "model.encoder.mask_mode", "tube")) != "random":
        raise ValueError("data.crop is defined with model.encoder.mask_mode random (padded tokens are forced masked)")
    frames, patch = int(get(cfg, "model.encoder.num_frames", 3)), int(get(cfg, "model.encoder.patch_size", 16))
    size = int(get(cfg, "model.encoder.image_height", 224))
    tokens = frames * (size // patch) * (int(get(cfg, "model.encoder.image_width", 224)) // patch)
    return {
        "crop": dict(crop),
        "patch_size": patch,
        "min_eligible_tokens": visible_count(tokens, float(get(cfg, "model.encoder.mask_ratio", 0.5))),
    }


def build_datasets(cfg: dict) -> tuple[DroidLargeScaleDataset, DroidLargeScaleDataset]:
    """Training split: the training crop policy ("motion"); validation split: ``data.crop.eval_policy`` (default
    "center", the padding-free inference crop)."""
    kwargs = dataset_kwargs(cfg)
    manifest = cfg["data"]["manifest"]
    val = dict(kwargs)
    if "crop" in kwargs:
        kwargs["crop_policy"] = "motion"
        val["crop_policy"] = str(kwargs["crop"].get("eval_policy", "center"))
    return (
        DroidLargeScaleDataset(manifest, split="train", **kwargs),
        DroidLargeScaleDataset(manifest, split="validation", with_eval=True, **val),
    )
