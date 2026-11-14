"""Ending diagnostics for the new session; labels come from manifest directories."""
from s4d.rwlearn import outcomes as legacy
from s4d.rwlearn.dsreset_modules import private_copy
from s4d.rwlearn.dsreset_tasks import TARGETS

_impl = private_copy(legacy)
_impl.TARGETS = TARGETS


def check_drawer_v2(sig):
    return {"phases": {}, "return": {}, "flags": [
        legacy._flag("drawer_v2_ending_review", "review", "New task ending thresholds require image review",
                     sig, [0, len(sig["robot/t"])-1])]}


_impl.CHECKS = {**legacy.CHECKS, "drawer-open-v2": check_drawer_v2}


def __getattr__(name):
    return getattr(_impl, name)
