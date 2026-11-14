"""Label-aware gate using the unchanged validation/QA/calibration implementation."""
import copy
from s4d.rwlearn import gate as legacy
from s4d.rwlearn.dsreset_modules import private_copy
from s4d.rwlearn.dsreset_identity import match_recordings

_impl = private_copy(legacy)
_impl.match_recordings = match_recordings
_evaluate = _impl.evaluate


def evaluate(manifest, tasks, **kwargs):
    # The old gate requires unique integer indices within each task. Encode the
    # (label, original index) pair injectively in its private input view only.
    # Invalid labels/indices remain invalid; report/raw/store identities stay exact.
    view = copy.deepcopy(manifest)
    for entry in (view.get("episodes") or {}).values():
        idx, label = entry.get("episode_index"), entry.get("episode_label", "success")
        entry["episode_index"] = (2 * idx + (label == "failure")
                                  if type(idx) is int and idx >= 0 and label in ("success", "failure") else None)
    return _evaluate(view, tasks, **kwargs)


_impl.evaluate = evaluate


def __getattr__(name):
    return getattr(_impl, name)
