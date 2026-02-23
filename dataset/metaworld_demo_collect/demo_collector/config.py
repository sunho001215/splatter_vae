from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class CameraCfg:
    name: str
    r: float
    theta: float  # degrees
    phi: float    # degrees
    fovy: float   # degrees


@dataclass
class OutputCfg:
    path: str
    mode: str = "overwrite"         # overwrite | resume
    compression: Optional[str] = "gzip"


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


@dataclass
class SegCfg:
    enabled: bool = True
    save_objtype: bool = False


@dataclass
class PolicyBlockCfg:
    enabled: bool = True
    num_demos: int = 0
    policy_class: Optional[str] = None  # scripted only


@dataclass
class PoliciesCfg:
    scripted: PolicyBlockCfg
    random: PolicyBlockCfg


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
    policies: PoliciesCfg
    dino: DinoCfg


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    out = OutputCfg(**raw["output"])
    mw = MetaWorldCfg(**raw["metaworld"])

    cams = [CameraCfg(**c) for c in raw["render"]["cameras"]]
    render = RenderCfg(
        height=int(raw["render"].get("height", 224)),
        width=int(raw["render"].get("width", 224)),
        lookat=raw["render"].get("lookat", None),
        up=raw["render"].get("up", [0.0, 0.0, 1.0]),
        cameras=cams,
    )

    seg = SegCfg(**raw.get("segmentation", {"enabled": True}))
    pol_raw = raw["policies"]
    policies = PoliciesCfg(
        scripted=PolicyBlockCfg(**pol_raw.get("scripted", {})),
        random=PolicyBlockCfg(**pol_raw.get("random", {})),
    )

    dino = DinoCfg(**raw.get("dino", {"enabled": False}))

    return Config(
        output=out,
        metaworld=mw,
        render=render,
        segmentation=seg,
        policies=policies,
        dino=dino,
    )