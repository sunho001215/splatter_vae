"""Pure admission/completion predicates for split pretraining jobs (no scheduler imports)."""


def profile_matches(job, uuid, known_mig):
    profile = job.get("gpu_profile")
    if profile not in (None, "1g.24gb"):
        raise ValueError("unsupported gpu_profile")
    return profile is None or uuid in known_mig


def continuation_endpoint(args, completion, metadata):
    try:
        if args.count("--stop-step") != 1:
            raise ValueError("one --stop-step required")
        step = int(args[args.index("--stop-step") + 1])
    except (IndexError, TypeError) as exc:
        raise ValueError("invalid continuation endpoint") from exc
    if (
        step not in (300000, 400000)
        or completion.get("status") != "completed"
        or completion != {"status": "completed", "steps": 400000}
        or metadata.get("status") != "completed"
        or metadata.get("stop_step") != step
        or metadata.get("schedule_steps") != 400000
    ):
        raise ValueError("Ours continuation did not complete its job endpoint")
    return step


def pretraining_endpoint(train):
    """Actual stop of a legacy train.py job, from identity-checked numeric config."""
    import math

    configured = train.get("steps") if isinstance(train, dict) else None
    if type(configured) is not int or configured <= 0:
        return None
    stop = train.get("stop_step")
    if not stop:
        return configured
    if type(stop) not in (int, float) or (type(stop) is float and not math.isfinite(stop)):
        return None
    return min(configured, int(stop)) if int(stop) > 0 else None


def resolved_pretraining_endpoint(args, numeric_config):
    """Apply the same scalar overrides as train.py before checking the actual endpoint."""
    import yaml

    train = dict((numeric_config or {}).get("train") or {})
    for key in ("steps", "stop_step"):
        prefix = f"train.{key}="
        values = [str(a)[len(prefix) :] for a in args if str(a).startswith(prefix)]
        if values:
            train[key] = yaml.safe_load(values[-1])
    return pretraining_endpoint(train)
