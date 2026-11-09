"""Authoritative registry of the real-world learning targets (docs/realworld_baselines.md, docs/realworld_fql.md).

Task-specific encoders only (user decision 2026-11-06 ~11:20: no joint-task pretraining by default): every target owns
its encoder, its seeded episode split and its 100-episode full-data gate. covering and pick-and-place joined on
2026-11-09 (real-world tasks; unrelated to the cancelled Meta-World task expansion). Standard library only and free of
imports: ``s4d.rwlearn.configs`` re-exports these names and ``ops.jobs.host_roles`` reads ``TARGETS``.
"""

TARGETS = ("drawer-open", "mug-hanging", "covering", "pick-and-place")
TARGET_TASKS = {task: (task,) for task in TARGETS}  # target -> the tasks in its store (task-specific: itself)
