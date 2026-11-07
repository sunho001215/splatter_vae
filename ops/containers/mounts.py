"""Role-specific bind mounts shared by remote launch and reconciliation."""

from ops.jobs.host_roles import DOMAIN, job_class
from ops.remote.access import REMOTE_ROOT
from s4d.rwlearn.configs import REMOTE_REALWORLD_ROOT

# Configs/store locators carry absolute paths; keep them identical inside the container. FQL dataset builds also
# read the raw robot/action signals (fql_data.build_fql_dataset -> actions.load_raw), not just compact RGB stores.
REALWORLD_READ_ROOTS = (f"{REMOTE_REALWORLD_ROOT}/learning_inputs", f"{REMOTE_REALWORLD_ROOT}/processed",
                        f"{REMOTE_ROOT}/data/real_world")


def realworld_mounts(job: dict) -> list[tuple[str, str, bool]]:
    """Return (source, destination, readonly) binds; never infer access from an ambiguous role."""
    classification = job_class(job)
    if classification.cls in ("invalid", "ambiguous"):
        raise ValueError(f"Invalid or ambiguous remote job role: {classification.evidence}")
    return ([(path, path, True) for path in REALWORLD_READ_ROOTS]
            if DOMAIN.get(classification.cls) == "realworld" else [])
