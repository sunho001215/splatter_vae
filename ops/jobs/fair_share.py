"""Pure local admission policy: equal dominant-resource guarantees with work-conserving borrowing.

Callers supply ready candidates in their existing order, active local reservations, measured RAM plus remaining
growth, and the queue's MIG limits. Unknown domains are charged to Meta-World by the caller. No I/O occurs here.
RAM not attributable to a job still consumes the cap. Running jobs are never moved or preempted.
"""

from __future__ import annotations

import copy

RESOURCES = ("ram_bytes", "gpu_jobs", "gpu_mem_gb")


def demand(row: dict) -> dict:
    return {"ram_bytes": float(row.get("ram_bytes", 0)), "gpu_jobs": 1.0,
            "gpu_mem_gb": float(row.get("mem_gb", 0))}


def dominant(usage: dict, capacity: dict) -> float:
    return max((usage.get(k, 0) / capacity[k] if capacity[k] else
                float("inf") if usage.get(k, 0) else 0) for k in RESOURCES)


def permits(family: str, need: dict, usage: dict, capacity: dict, pending: dict, total: dict) -> bool:
    """Fit all caps, then stay within 1/n or leave every other's claimable unmet guarantee available.

    Claimable means min(remaining ready demand, positive guarantee minus current usage), independently for RAM,
    GPU jobs and GPU memory. A family without ready work reserves nothing. An active family still counts in n.
    """
    n = len(usage)
    after = {k: total.get(k, 0) + need.get(k, 0) for k in RESOURCES}
    if not n or any(after[k] > capacity[k] for k in RESOURCES):
        return False
    if dominant({k: usage[family].get(k, 0) + need.get(k, 0) for k in RESOURCES}, capacity) <= 1 / n + 1e-12:
        return True
    return all(capacity[k] - after[k] + 1e-8 >= sum(
        min(max(0, capacity[k] / n - used.get(k, 0)), pending.get(other, {}).get(k, 0))
        for other, used in usage.items() if other != family) for k in RESOURCES)


def select(candidates: list[dict], active: list[dict], limits: dict, ram_cap: float, ram_used: float) -> dict:
    """Return admitted ids with their MIG instance (dispatched with the row), per-family blocked trainer ids, MIG
    instances reserved for protected heads, and a usage/guarantee report.

    Each row has id, family, ram_bytes, mem_gb, gpu (or any); candidates may mark evaluator, trainer and trainer_ready.
    Within a family the given order holds and the first non-fitting trainer blocks the family's later trainers (short
    evaluators may still fill capacity). Across families, each pass FIRST serves the head trainer of every family that
    is below its 1/n guarantee (least-served first): it starts if it fits, otherwise it reserves what it needs (its RAM
    and the MIG instance with the most free memory large enough for it) before any other row of any family, evaluators
    included, is considered. Reservations are recomputed from the live state at every pass, so freed capacity reaches
    a protected head before any replacement can take it; the reserved instance only drains, never refills, until the
    head starts. Other rows then follow in order, borrowing only capacity no protected head or other family's unmet
    guarantee claims. Arguments are never mutated.
    """
    ids = {row["id"] for row in candidates}
    candidates = [r for r in candidates if not r.get("evaluator") or r.get("trainer_ready") or r.get("trainer") in ids]
    slots = {gpu: {**copy.deepcopy(limit), "jobs": 0, "mem_gb": 0.0} for gpu, limit in limits.items()}
    capacity = {"ram_bytes": float(ram_cap), "gpu_jobs": sum(int(s["max_jobs"]) for s in slots.values()),
                "gpu_mem_gb": sum(float(s["max_mem_gb"]) for s in slots.values())}
    families = sorted({r["family"] for r in active + candidates})
    usage = {f: dict.fromkeys(RESOURCES, 0.0) for f in families}
    for row in active:
        for k, v in demand(row).items():
            usage[row["family"]][k] += v
        if row.get("gpu") in slots:
            slots[row["gpu"]]["jobs"] += 1
            slots[row["gpu"]]["mem_gb"] += float(row.get("mem_gb", 0))
    total = {k: sum(u[k] for u in usage.values()) for k in RESOURCES}
    total["ram_bytes"] = max(float(ram_used), total["ram_bytes"])
    n = len(families)
    admitted, blocked, assigned = [], {}, {}
    reserved = {}  # MIG instance -> id of the protected head it is kept for
    held = {f: dict.fromkeys(RESOURCES, 0.0) for f in families}  # kept for protected heads that could not start yet

    def possible(row):
        # An intrinsically impossible head can neither start nor reserve; the trainers behind it are not blocked by it.
        return row.get("ram_bytes", 0) <= ram_cap and any(
            row.get("gpu", "any") in ("any", gpu) and row.get("mem_gb", 0) <= slot["max_mem_gb"]
            for gpu, slot in slots.items())

    def choose(row):
        """The scheduler's own rule (ops/jobs/scheduler.py choose_gpu: least memory after placement, first on ties)
        over the instances not reserved for another row."""
        best = None
        for gpu, slot in slots.items():
            if (row.get("gpu", "any") in ("any", gpu) and reserved.get(gpu, row["id"]) == row["id"]
                    and slot["jobs"] < int(slot["max_jobs"])
                    and slot["mem_gb"] + float(row.get("mem_gb", 0)) <= float(slot["max_mem_gb"])):
                memory = slot["mem_gb"] + float(row.get("mem_gb", 0))
                if best is None or memory < best[0]:
                    best = (memory, gpu)
        return None if best is None else best[1]

    def fits(need, family):
        """Capacity left after every OTHER family's protected-head hold. A family's own hold is open only to its
        evaluators (its trainers wait behind the head): the established short-evaluator exception of 2026-10-29."""
        free = {k: capacity[k] - total[k] - sum(h[k] for f, h in held.items() if f != family) for k in RESOURCES}
        return all(need[k] <= free[k] + 1e-9 for k in RESOURCES)

    def start(row, gpu):
        admitted.append(row["id"])
        assigned[row["id"]] = gpu
        slots[gpu]["jobs"] += 1
        slots[gpu]["mem_gb"] += float(row.get("mem_gb", 0))
        for k, v in demand(row).items():
            usage[row["family"]][k] += v
            total[k] += v

    def ready(row):
        return not row.get("evaluator") or row.get("trainer_ready") or row.get("trainer") in admitted

    # 1. protected heads: the first possible trainer of each family below its guarantee, least-served first
    heads = {}
    for row in candidates:
        if not row.get("evaluator") and row["family"] not in heads and possible(row):
            heads[row["family"]] = row
    protected = [heads[f] for f in sorted(heads, key=lambda f: (dominant(usage[f], capacity),
                                                                candidates.index(heads[f])))
                 if dominant(usage[f], capacity) < 1 / max(n, 1) - 1e-12 and possible(heads[f])]
    for head in protected:
        # the family's own ready evaluators that precede its head keep their place (the driver orders evaluators of
        # finished trainers first; Codex review 4): they start before the head, under the same rules as in step 2
        for early in candidates[:candidates.index(head)]:
            if (early["family"] == head["family"] and early.get("evaluator") and ready(early)
                    and early["id"] not in admitted):
                gpu = choose(early)
                if gpu is not None and fits(demand(early), early["family"]) and permits(
                        early["family"], demand(early), usage, capacity, {}, total):
                    start(early, gpu)
        need, gpu = demand(head), choose(head)
        if gpu is not None and fits(need, head["family"]):
            start(head, gpu)
            continue
        blocked[head["family"]] = head["id"]
        for k in RESOURCES:  # keep what is free now, up to its need: capacity accumulates for it as running jobs end
            taken = sum(h[k] for h in held.values())
            held[head["family"]][k] = min(need[k], max(0.0, capacity[k] - total[k] - taken))
        if gpu is None:
            large = [g for g, slot in slots.items() if head.get("gpu", "any") in ("any", g) and g not in reserved
                     and float(head.get("mem_gb", 0)) <= float(slot["max_mem_gb"])]
            if large:
                reserved[max(large, key=lambda g: (float(slots[g]["max_mem_gb"]) - slots[g]["mem_gb"],
                                                   int(slots[g]["max_jobs"]) - slots[g]["jobs"],
                                                   -list(slots).index(g)))] = head["id"]
    done = {row["id"] for row in protected}

    # 2. every other row in order, within the capacity the protected heads leave
    for row in candidates:
        if row["id"] in done or row["id"] in admitted or not ready(row):  # early evaluators started in step 1
            continue
        family = row["family"]
        evaluator = row.get("evaluator", False)
        if not evaluator and family in blocked:
            continue
        pending = {f: dict.fromkeys(RESOURCES, 0.0) for f in families}
        for waiting in candidates:
            if (waiting["id"] in admitted or waiting["id"] in done or not possible(waiting) or not ready(waiting)
                    or (not waiting.get("evaluator") and waiting["family"] in blocked)):
                continue
            for k, v in demand(waiting).items():
                pending[waiting["family"]][k] += v
        need, gpu = demand(row), choose(row)
        others = {k: sum(h[k] for f, h in held.items() if f != family) for k in RESOURCES}
        if gpu is None or not fits(need, family) or not permits(family, need, usage, capacity, pending,
                                                                 {k: total[k] + others[k] for k in RESOURCES}):
            if not evaluator and possible(row):
                blocked[family] = row["id"]
            continue
        start(row, gpu)
    guarantee = {k: capacity[k] / n for k in RESOURCES} if families else {}
    return {"admitted": admitted, "assigned": assigned, "reserved": reserved, "blocked": blocked,
            "total_ram_bytes": total["ram_bytes"],
            "report": {"capacity": capacity, "held_for_protected_heads": held, "reserved_instances": reserved,
                       "families": {f: {"usage": usage[f], "guarantee": guarantee,
                                        "dominant_share": dominant(usage[f], capacity)} for f in families}}}
