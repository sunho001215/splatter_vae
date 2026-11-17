"""Opt-in, process-local new-task support. Never imported by existing jobs.

Invoke only in the dedicated PNP v2 entrypoint, before trainer/store imports.
No calibration acceptance criteria, task overrides or legacy files are changed.
"""

from s4d.rwlearn.pnpv2_spec import TASK


def install():
    from s4d.rwlearn import dsreset, dsreset_configs, dsreset_gate, dsreset_tasks
    from s4d.rwlearn.dsreset_runtime import install as paired_install

    # dsreset's private modules already isolate the placed legacy scopes. This
    # extra task is visible only to the new entrypoint's Python process.
    dsreset.TASKS = tuple(dict.fromkeys((*dsreset.TASKS, TASK)))
    dsreset_tasks.TARGETS = tuple(dict.fromkeys((*dsreset_tasks.TARGETS, TASK)))
    dsreset_tasks.TARGET_TASKS = {t: (t,) for t in dsreset_tasks.TARGETS}
    dsreset_configs._impl.TARGETS = dsreset_tasks.TARGETS
    dsreset_configs._impl.TARGET_TASKS = dsreset_tasks.TARGET_TASKS
    dsreset_configs._impl.SCENE_PARAMETERS = {
        **dsreset_configs._impl.SCENE_PARAMETERS,
        TASK: {"lookat": None, "pose_scale": None, "bounds": None},
    }
    dsreset_gate._impl.TARGETS = dsreset_tasks.TARGETS
    paired_install()
