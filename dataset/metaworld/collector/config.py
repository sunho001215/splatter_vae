from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class VisualizeCfg:
    enabled: bool = False
    show_window: bool = True
    every_n_steps: int = 1
    ncols: int = 3
    pad: int = 4
    window_name: str = "metaworld demos (top=RGB, bottom=SEG)"
    stop_key: str = "q"
    save_video_dir: Optional[str] = None
    video_fps: float = 10.0


@dataclass
class CameraCfg:
    name: str
    r: float
    theta: float
    phi: float
    fovy: float


@dataclass
class OutputCfg:
    path: str
    mode: str = "overwrite"
    compression: Optional[str] = "lzf"


@dataclass
class MetaWorldCfg:
    benchmark_id: str
    env_name: str
    seed: int = 0
    max_steps: int = 500
    terminate_on_success: bool = True


@dataclass
class RenderCfg:
    height: int = 224
    width: int = 224
    lookat: Optional[List[float]] = None
    up: List[float] = None
    cameras: List[CameraCfg] = None
    save_depth: bool = False


@dataclass
class SegCfg:
    enabled: bool = True
    save_objtype: bool = False


@dataclass
class NoiseScheduleEntryCfg:
    strength: float
    num_demos: int


@dataclass
class ExpertGuidedCfg:
    num_demos: int
    noise_schedule: List[NoiseScheduleEntryCfg]
    noise_smoothing: float = 0.85
    noise_target_hold_steps: tuple[int, int] = (8, 16)


@dataclass
class PerturbRecoverCfg:
    num_demos: int
    start_fraction: tuple[float, float] = (0.25, 0.55)
    duration_steps: tuple[int, int] = (8, 24)
    strength_range: tuple[float, float] = (0.30, 0.70)
    noise_smoothing: float = 0.85
    noise_target_hold_steps: tuple[int, int] = (4, 10)
    minimum_recovery_steps: int = 10


@dataclass
class SmoothRandomCfg:
    num_demos: int
    xyz_smoothing: float = 0.80
    gripper_smoothing: float = 0.95
    xyz_target_hold_steps: tuple[int, int] = (6, 14)
    gripper_target_hold_steps: tuple[int, int] = (30, 60)


@dataclass
class CollectionCfg:
    scripted_policy_class: Optional[str]
    expert_guided: ExpertGuidedCfg
    perturb_recover: PerturbRecoverCfg
    smooth_random: SmoothRandomCfg


@dataclass
class DinoCfg:
    enabled: bool = False
    model: str = "dinov2_vitb14"
    image_size: int = 224
    batch_size: int = 16
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class Config:
    output: OutputCfg
    metaworld: MetaWorldCfg
    render: RenderCfg
    segmentation: SegCfg
    collection: CollectionCfg
    dino: DinoCfg
    visualize: VisualizeCfg


def _integer_range(values: list[int], name: str) -> tuple[int, int]:
    if len(values) != 2 or int(values[0]) <= 0 or int(values[1]) < int(values[0]):
        raise ValueError(f"{name} must be [positive_min, max>=min].")
    return int(values[0]), int(values[1])


def _float_range(values: list[float], name: str) -> tuple[float, float]:
    if len(values) != 2 or float(values[1]) < float(values[0]):
        raise ValueError(f"{name} must be [min, max>=min].")
    return float(values[0]), float(values[1])


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    out = OutputCfg(**raw["output"])
    mw = MetaWorldCfg(**raw["metaworld"])
    cams = [CameraCfg(**camera) for camera in raw["render"]["cameras"]]
    render = RenderCfg(
        height=int(raw["render"].get("height", 224)),
        width=int(raw["render"].get("width", 224)),
        lookat=raw["render"].get("lookat"),
        up=raw["render"].get("up", [0.0, 0.0, 1.0]),
        cameras=cams,
        save_depth=bool(raw["render"].get("save_depth", False)),
    )
    seg = SegCfg(**raw.get("segmentation", {"enabled": True}))

    collection_raw = raw["collection"]
    expert_raw = collection_raw["expert_guided"]
    perturb_raw = collection_raw["perturb_recover"]
    random_raw = collection_raw["smooth_random"]
    collection = CollectionCfg(
        scripted_policy_class=collection_raw.get("scripted_policy_class"),
        expert_guided=ExpertGuidedCfg(
            num_demos=int(expert_raw["num_demos"]),
            noise_schedule=[
                NoiseScheduleEntryCfg(
                    strength=float(entry["strength"]),
                    num_demos=int(entry["num_demos"]),
                )
                for entry in expert_raw["noise_schedule"]
            ],
            noise_smoothing=float(expert_raw.get("noise_smoothing", 0.85)),
            noise_target_hold_steps=_integer_range(
                expert_raw.get("noise_target_hold_steps", [8, 16]),
                "expert_guided.noise_target_hold_steps",
            ),
        ),
        perturb_recover=PerturbRecoverCfg(
            num_demos=int(perturb_raw["num_demos"]),
            start_fraction=_float_range(
                perturb_raw.get("start_fraction", [0.25, 0.55]),
                "perturb_recover.start_fraction",
            ),
            duration_steps=_integer_range(
                perturb_raw.get("duration_steps", [8, 24]),
                "perturb_recover.duration_steps",
            ),
            strength_range=_float_range(
                perturb_raw.get("strength_range", [0.30, 0.70]),
                "perturb_recover.strength_range",
            ),
            noise_smoothing=float(perturb_raw.get("noise_smoothing", 0.85)),
            noise_target_hold_steps=_integer_range(
                perturb_raw.get("noise_target_hold_steps", [4, 10]),
                "perturb_recover.noise_target_hold_steps",
            ),
            minimum_recovery_steps=int(
                perturb_raw.get("minimum_recovery_steps", 10)
            ),
        ),
        smooth_random=SmoothRandomCfg(
            num_demos=int(random_raw["num_demos"]),
            xyz_smoothing=float(random_raw.get("xyz_smoothing", 0.80)),
            gripper_smoothing=float(random_raw.get("gripper_smoothing", 0.95)),
            xyz_target_hold_steps=_integer_range(
                random_raw.get("xyz_target_hold_steps", [6, 14]),
                "smooth_random.xyz_target_hold_steps",
            ),
            gripper_target_hold_steps=_integer_range(
                random_raw.get("gripper_target_hold_steps", [30, 60]),
                "smooth_random.gripper_target_hold_steps",
            ),
        ),
    )

    expected_noise_schedule = {
        0.00: 30,
        0.05: 30,
        0.15: 30,
        0.30: 30,
        0.50: 30,
    }
    actual_noise_schedule = {
        float(entry.strength): int(entry.num_demos)
        for entry in collection.expert_guided.noise_schedule
    }
    if actual_noise_schedule != expected_noise_schedule:
        raise ValueError(
            "expert_guided.noise_schedule must contain exactly 30 demos at "
            "strengths 0.00, 0.05, 0.15, 0.30, and 0.50."
        )
    if collection.expert_guided.num_demos != 150:
        raise ValueError("expert_guided.num_demos must be exactly 150.")
    if collection.perturb_recover.num_demos != 60:
        raise ValueError("perturb_recover.num_demos must be exactly 60.")
    if collection.smooth_random.num_demos != 40:
        raise ValueError("smooth_random.num_demos must be exactly 40.")
    if not 0.0 <= collection.perturb_recover.start_fraction[0] <= collection.perturb_recover.start_fraction[1] <= 1.0:
        raise ValueError("perturb_recover.start_fraction must lie within [0, 1].")
    if not 0.0 <= collection.perturb_recover.strength_range[0] <= collection.perturb_recover.strength_range[1] <= 1.0:
        raise ValueError("perturb_recover.strength_range must lie within [0, 1].")
    if collection.perturb_recover.minimum_recovery_steps < 1:
        raise ValueError("perturb_recover.minimum_recovery_steps must be positive.")
    for name, value in (
        ("expert_guided.noise_smoothing", collection.expert_guided.noise_smoothing),
        ("perturb_recover.noise_smoothing", collection.perturb_recover.noise_smoothing),
        ("smooth_random.xyz_smoothing", collection.smooth_random.xyz_smoothing),
        ("smooth_random.gripper_smoothing", collection.smooth_random.gripper_smoothing),
    ):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1).")

    dino = DinoCfg(**raw.get("dino", {"enabled": False}))
    viz = VisualizeCfg(**raw.get("visualize", {"enabled": False}))
    return Config(
        output=out,
        metaworld=mw,
        render=render,
        segmentation=seg,
        collection=collection,
        dino=dino,
        visualize=viz,
    )
