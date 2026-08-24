from __future__ import annotations

from .config import CURRENT_FRAME_INDEX

TEMPORAL_ANCHOR = "current"


def validate_temporal_anchor(temporal_anchor: str) -> str:
    if temporal_anchor != TEMPORAL_ANCHOR:
        raise ValueError(
            "DROID Gaussian dynamics are anchored only at the current history frame; "
            f"got {temporal_anchor!r}."
        )
    return temporal_anchor


def temporal_anchor_index(temporal_anchor: str = TEMPORAL_ANCHOR) -> int:
    validate_temporal_anchor(temporal_anchor)
    return CURRENT_FRAME_INDEX
