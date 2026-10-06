from .camera import (
    LAGERNVS_IMAGE_SIZE,
    canonicalize_droid_views,
    lager_to_display_plane,
)
from .coverage import source_coverage_from_depth
from .pose import (
    LagerTargetPoseConfig,
    pose_sampler_contract,
    sample_safe_target_poses,
)


def __getattr__(name: str):
    if name in {
        "LAGERNVS_CHECKPOINT_ID",
        "LAGERNVS_CHECKPOINT_REVISION",
        "LAGERNVS_REPOSITORY_REVISION",
        "LagerNVSDROIDTeacher",
    }:
        from . import official

        return getattr(official, name)
    raise AttributeError(name)

__all__ = [
    "LAGERNVS_CHECKPOINT_ID",
    "LAGERNVS_CHECKPOINT_REVISION",
    "LAGERNVS_IMAGE_SIZE",
    "LAGERNVS_REPOSITORY_REVISION",
    "LagerNVSDROIDTeacher",
    "LagerTargetPoseConfig",
    "canonicalize_droid_views",
    "lager_to_display_plane",
    "pose_sampler_contract",
    "sample_safe_target_poses",
    "source_coverage_from_depth",
]
