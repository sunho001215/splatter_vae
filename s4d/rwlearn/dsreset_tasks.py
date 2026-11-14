"""Task registry for new paired rows; legacy placed rows retain tasks.py."""
from s4d.rwlearn.tasks import TARGETS as LEGACY_TARGETS

TARGETS = (*LEGACY_TARGETS, "drawer-open-v2")
TARGET_TASKS = {task: (task,) for task in TARGETS}
