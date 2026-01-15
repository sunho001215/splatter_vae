# policy_utils/config_loader.py
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.normalize import NormalizationMode


def _to_policy_feature(spec: Any) -> PolicyFeature:
    if isinstance(spec, PolicyFeature):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Feature spec must be dict or PolicyFeature, got: {type(spec)}")

    ftype = spec.get("type")
    shape = spec.get("shape")
    if not isinstance(ftype, str):
        raise ValueError(f"Feature spec missing string 'type': {spec}")
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"Feature spec missing list 'shape': {spec}")

    # "VISUAL"/"STATE"/"ACTION" -> FeatureType enum
    if ftype in FeatureType.__members__:
        ftype_enum = FeatureType[ftype]
    else:
        ftype_enum = FeatureType(ftype)

    return PolicyFeature(type=ftype_enum, shape=tuple(int(x) for x in shape))


def _convert_feature_dict(d: Dict[str, Any]) -> Dict[str, PolicyFeature]:
    return {k: _to_policy_feature(v) for k, v in d.items()}


def _to_feature_type(k: Any) -> FeatureType:
    if isinstance(k, FeatureType):
        return k
    if not isinstance(k, str):
        raise TypeError(f"normalization_mapping key must be str/FeatureType, got {type(k)}")
    if k in FeatureType.__members__:
        return FeatureType[k]
    return FeatureType(k)


def _to_norm_mode(v: Any) -> NormalizationMode:
    if isinstance(v, NormalizationMode):
        return v
    if not isinstance(v, str):
        raise TypeError(f"normalization_mapping value must be str/NormalizationMode, got {type(v)}")
    # "MEAN_STD"/"MIN_MAX"/"IDENTITY" -> NormalizationMode enum
    if v in NormalizationMode.__members__:
        return NormalizationMode[v]
    return NormalizationMode(v)


def _convert_normalization_mapping(m: Any) -> Any:
    """
    JSON: {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    ->    {FeatureType.VISUAL: NormalizationMode.MEAN_STD, ...}
    """
    if m is None:
        return None
    if not isinstance(m, dict):
        raise TypeError(f"normalization_mapping must be dict, got {type(m)}")
    return {_to_feature_type(k): _to_norm_mode(v) for k, v in m.items()}


def load_diffusion_config_json(
    json_path: str | Path,
) -> Tuple[DiffusionConfig, Dict[str, Any]]:
    json_path = Path(json_path)
    raw: Dict[str, Any] = json.loads(json_path.read_text())

    raw.pop("type", None)

    # features: dict -> PolicyFeature
    if "input_features" in raw and isinstance(raw["input_features"], dict):
        raw["input_features"] = _convert_feature_dict(raw["input_features"])
    if "output_features" in raw and isinstance(raw["output_features"], dict):
        raw["output_features"] = _convert_feature_dict(raw["output_features"])

    # IMPORTANT FIX: normalization_mapping string -> enum
    if "normalization_mapping" in raw:
        raw["normalization_mapping"] = _convert_normalization_mapping(raw["normalization_mapping"])

    # common list->tuple conversions
    if "crop_shape" in raw and raw["crop_shape"] is not None:
        raw["crop_shape"] = tuple(raw["crop_shape"])
    if "optimizer_betas" in raw and raw["optimizer_betas"] is not None:
        raw["optimizer_betas"] = tuple(raw["optimizer_betas"])

    # DiffusionConfig init args filtering
    sig = inspect.signature(DiffusionConfig)
    filtered = {k: v for k, v in raw.items() if k in sig.parameters}

    cfg = DiffusionConfig(**filtered)
    return cfg, raw
