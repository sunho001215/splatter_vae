"""Coordinator-only continuity checks; never imported by job runtime code."""

from __future__ import annotations


def releases(history: list, commit: str, image: str) -> set[tuple[str, str]]:
    """Validate an ordered, same-image chain ending at the current release; return all visited releases."""
    if not isinstance(history, list):
        raise ValueError("release history must be a list")
    visited, last = set(), None
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("invalid release history entry")
        values = [item.get(key) for key in ("from_commit", "from_image_digest", "to_commit", "to_image_digest")]
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("incomplete release history entry")
        source, target = tuple(values[:2]), tuple(values[2:])
        if (source[1] != image or target[1] != image or (last is not None and source != last)
                or target == source or target in visited):
            raise ValueError("release history has a gap, duplicate or image change")
        visited.update((source, target))
        last = target
    if last is not None and last != (commit, image):
        raise ValueError("release history does not end at the current release")
    return visited | {(commit, image)}


def registry_history(path, job_id):
    """Read transition provenance without adding fields to legacy reducer output."""
    import json
    history = []
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("id") == job_id and event.get("event") == "release_repinned":
            if ((history or "repin_chain" in event)
                    and (not history or event.get("repin_chain") != history)):
                raise ValueError("invalid release re-pin chain record")
            history.append({key: value for key, value in event.items() if key != "repin_chain"})
    return history


def validate_events(history, commit, image):
    if not isinstance(history, list) or any(not isinstance(e, dict) for e in history):
        raise ValueError("release re-pin history must be a list of events")
    if history and any(e.get("id") != history[0].get("id") for e in history):
        raise ValueError("release re-pin history mixes job ids")
    for event in history:
        if (event.get("sources_identical") is not True or event.get("host") != "remote"
                or not isinstance(event.get("authorization"), str) or not event["authorization"].strip()):
            raise ValueError("invalid release re-pin authorization")
    return releases([{**e, "to_commit": e.get("commit"), "to_image_digest": e.get("image_digest")}
                     for e in history], commit, image)


# These trainers share one entry/trainer provenance path, enforced by validate_output_arguments.
PRETRAINERS = frozenset({"scripts/train.py", "scripts/train_sincro.py", "scripts/train_reviwo.py",
                         "scripts/train_snerl.py", "scripts/train_dynamo.py", "scripts/train_mvp3d.py"})
