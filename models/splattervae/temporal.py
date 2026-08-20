from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


TEMPORAL_ANCHORS = ("t0", "t2")


def validate_temporal_anchor(temporal_anchor: str) -> str:
    """Return a supported temporal anchor without coercing invalid values."""
    if temporal_anchor not in TEMPORAL_ANCHORS:
        raise ValueError(
            f"temporal_anchor must be exactly one of {TEMPORAL_ANCHORS}, "
            f"got {temporal_anchor!r}."
        )
    return temporal_anchor


def temporal_anchor_index(temporal_anchor: str) -> int:
    """Return the chronological index of the directly decoded Gaussian set."""
    return 0 if validate_temporal_anchor(temporal_anchor) == "t0" else 2


def combine_temporal_anchor_losses(
    losses: Sequence[Any],
    temporal_anchor: str,
    temporal_ramp: float,
) -> Any:
    """Give the anchor full weight and ramp the mean of both non-anchor losses."""
    if len(losses) != 3:
        raise ValueError(f"Expected exactly three chronological losses, got {len(losses)}.")
    anchor_index = temporal_anchor_index(temporal_anchor)
    non_anchor_indices = tuple(index for index in range(3) if index != anchor_index)
    return losses[anchor_index] + float(temporal_ramp) * 0.5 * (
        losses[non_anchor_indices[0]] + losses[non_anchor_indices[1]]
    )


def temporal_anchor_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    """Read checkpoint anchor metadata, treating legacy checkpoints as t0."""
    missing = object()
    temporal_anchor = checkpoint.get("temporal_anchor", missing)
    if temporal_anchor is missing:
        configuration = checkpoint.get("configuration", {})
        if isinstance(configuration, Mapping):
            train_configuration = configuration.get("train", {})
            if isinstance(train_configuration, Mapping):
                temporal_anchor = train_configuration.get("temporal_anchor", missing)
    if temporal_anchor is missing:
        temporal_anchor = "t0"
    return validate_temporal_anchor(temporal_anchor)


def validate_checkpoint_temporal_anchor(
    checkpoint: Mapping[str, Any],
    configured_temporal_anchor: str,
) -> str:
    """Reject shape-compatible checkpoints whose temporal output meaning differs."""
    checkpoint_anchor = temporal_anchor_from_checkpoint(checkpoint)
    configured_anchor = validate_temporal_anchor(configured_temporal_anchor)
    if checkpoint_anchor != configured_anchor:
        raise ValueError(
            "Checkpoint temporal_anchor does not match the current YAML: "
            f"checkpoint={checkpoint_anchor!r}, config={configured_anchor!r}."
        )
    return checkpoint_anchor
