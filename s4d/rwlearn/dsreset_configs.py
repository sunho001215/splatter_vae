"""New task configs in a private module; existing row recipes stay unchanged."""
from s4d.rwlearn import configs as legacy
from s4d.rwlearn.dsreset_modules import private_copy
from s4d.rwlearn.dsreset_tasks import TARGETS, TARGET_TASKS

_impl = private_copy(legacy)
_impl.TARGETS, _impl.TARGET_TASKS = TARGETS, TARGET_TASKS
_impl.SCENE_PARAMETERS = {**legacy.SCENE_PARAMETERS,
                          "drawer-open-v2": {"lookat": None, "pose_scale": None, "bounds": None}}


def __getattr__(name):
    return getattr(_impl, name)
