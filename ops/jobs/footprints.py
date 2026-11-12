"""Measured host RAM policy, shared by placement and running-job growth reservations."""
from ops.jobs.arguments import option


def fql_ram_gb(job: dict) -> float | None:
    # Local 1g.24gb measurements, 2026-11-12 (VmHWM): frozen encoder 1.5 GiB,
    # e2e_cnn 4.3 GiB. User-approved rounded admission charges: 3 / 8 GiB.
    # Keep all other families on their existing policy.
    if job.get("script") != "scripts/train_realworld_fql.py":
        return None
    args = list(map(str, job.get("args", [])))
    overrides = [value.split("=", 1)[1] for value in args if value.startswith("condition=")]
    condition = overrides[-1] if overrides else option(args, "--fql-condition") or job.get("fql_condition")
    return 8.0 if condition == "e2e_cnn" else 3.0


def ram_gb(job: dict) -> float:
    measured = fql_ram_gb(job)
    return float(job.get("ram_gb", 0)) if measured is None else measured


def exclusive(job: dict) -> bool:
    # Native verification and isolation measure device behaviour without co-tenant interference.
    return job.get("exclusive_gpu", False) is True or job.get("verification_purpose") is not None
