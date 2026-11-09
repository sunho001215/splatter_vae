"""Permanent launch refusals for algorithms retired on 2026-11-09; independent of campaign pause records."""

from __future__ import annotations

from pathlib import PurePosixPath

# Algorithm -> (trainer entry point, historical run id prefix). Keep historical validators separate.
RETIRED_ALGORITHMS = {
    "PPO": ("scripts/train_ppo.py", "ds-ppo-"),
    "Dreamer": ("scripts/train_dreamer.py", "ds-dreamer-"),
    "FlashSAC": ("scripts/train_flashsac.py", "ds-flashsac-"),
}


def retired_reason(job: dict, by_id: dict[str, dict] | None = None) -> str | None:
    """Refuse retired scripts/ids and RL evaluators targeting retired runs, including renamed trainers in context.

    Run targets may be --train-job, --run-dir (split or equals form), reevaluates, or dependencies. Path components
    recognize both repository-relative and container/host-absolute run directories. No run artifacts are read.
    """
    return _reason(job, by_id or {}, set())


def _reason(job: dict, by_id: dict[str, dict], seen: set[str]) -> str | None:
    script, job_id = str(PurePosixPath(job.get("script", ""))), job.get("id", "")
    for algorithm, (trainer, prefix) in RETIRED_ALGORITHMS.items():
        if script == trainer or job_id.startswith(prefix):
            return f"{algorithm} retired on 2026-11-09 ({trainer}; {prefix}*)"
    if script != "scripts/eval_rl.py" or job_id in seen:
        return None
    seen.add(job_id)
    targets = list(job.get("deps", []))
    if job.get("reevaluates"):
        targets.append(job["reevaluates"])
    arguments = list(map(str, job.get("args", [])))
    for index, argument in enumerate(arguments):
        flag, equal, value = argument.partition("=")
        if flag in ("--train-job", "--run-dir"):
            if equal:
                targets.append(value)
            elif index + 1 < len(arguments):
                targets.append(arguments[index + 1])
    for target in targets:
        for part in PurePosixPath(str(target)).parts:
            for algorithm, (_, prefix) in RETIRED_ALGORITHMS.items():
                if part.startswith(prefix):
                    return f"{algorithm} retired on 2026-11-09 (evaluator target {target})"
            if part in by_id:
                reason = _reason(by_id[part], by_id, seen)
                if reason is not None:
                    return f"evaluator target {target}: {reason}"
    return None
