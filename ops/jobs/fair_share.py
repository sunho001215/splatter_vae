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
    """Fit all caps, then stay within 1/n (or start below it), or leave every other's claimable unmet guarantee.

    Claimable means min(remaining ready demand, positive guarantee minus current usage), independently for RAM,
    GPU jobs and GPU memory. A family without ready work reserves nothing. An active family still counts in n.
    """
    n = len(usage)
    after = {k: total.get(k, 0) + need.get(k, 0) for k in RESOURCES}
    if not n or any(after[k] > capacity[k] for k in RESOURCES):
        return False
    if dominant({k: usage[family].get(k, 0) + need.get(k, 0) for k in RESOURCES}, capacity) <= 1 / n + 1e-12:
        return True
    mine = dominant(usage[family], capacity)
    if mine < 1 / n - 1e-12 and all(dominant(used, capacity) >= mine - 1e-12 for other, used in usage.items()
                                    if other != family and any(pending.get(other, {}).values())):
        # The least-served family, still below its guarantee, may admit one indivisible row that overshoots it
        # (otherwise two heads larger than 1/n deadlock an empty machine). Only the least-served one: a family that
        # is merely below its share must not keep refilling past it while a less-served family's head waits.
        return True
    return all(capacity[k] - after[k] + 1e-8 >= sum(
        min(max(0, capacity[k] / n - used.get(k, 0)), pending.get(other, {}).get(k, 0))
        for other, used in usage.items() if other != family) for k in RESOURCES)


def select(candidates: list[dict], active: list[dict], limits: dict, ram_cap: float, ram_used: float) -> dict:
    """Return admitted ids with their MIG instance (dispatched with the row), per-family blocked trainer ids, MIG
    instances reserved for blocked heads, and a usage/guarantee report.

    Each row has id, family, ram_bytes, mem_gb, gpu (or any); candidates may mark evaluator, trainer and trainer_ready.
    The first non-fitting trainer reserves its family's turn; short evaluators may still fill capacity. Actual MIG
    packing is checked as well as aggregate GPU guarantees. Arguments are never mutated.
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
    remaining, admitted, blocked = list(candidates), [], {}
    assigned, reserved = {}, {}  # id -> MIG instance; MIG instance -> family whose blocked head reserves it

    def choose(row, family):
        """The scheduler's own rule (ops/jobs/scheduler.py choose_gpu: least memory after placement, first on ties),
        skipping instances reserved for another family's blocked head. The choice is dispatched with the row."""
        best = None
        for gpu, slot in slots.items():
            if (row.get("gpu", "any") in ("any", gpu) and reserved.get(gpu, family) == family
                    and slot["jobs"] < int(slot["max_jobs"])
                    and slot["mem_gb"] + float(row.get("mem_gb", 0)) <= float(slot["max_mem_gb"])):
                memory = slot["mem_gb"] + float(row.get("mem_gb", 0))
                if best is None or memory < best[0]:
                    best = (memory, gpu)
        return None if best is None else best[1]

    def reserve_for(row, family):
        """A head blocked only by MIG packing reserves the instance that will fit it soonest (most free memory among
        those large enough), so later small rows cannot keep every instance fragmented."""
        if any(owner == family for owner in reserved.values()):
            return
        fitting = [gpu for gpu, slot in slots.items() if row.get("gpu", "any") in ("any", gpu)
                   and float(row.get("mem_gb", 0)) <= float(slot["max_mem_gb"]) and gpu not in reserved]
        if fitting:
            gpu = max(fitting, key=lambda g: (float(slots[g]["max_mem_gb"]) - slots[g]["mem_gb"],
                                              int(slots[g]["max_jobs"]) - slots[g]["jobs"], -list(slots).index(g)))
            reserved[gpu] = family

    def possible(row):
        # An intrinsically impossible head cannot claim borrowed capacity, nor can trainers behind that head.
        # Temporarily occupied slots still reserve the guarantee: active work can finish on a later pass.
        return row.get("ram_bytes", 0) <= ram_cap and any(
            row.get("gpu", "any") in ("any", gpu) and row.get("mem_gb", 0) <= slot["max_mem_gb"]
            for gpu, slot in slots.items())

    for row in candidates:
        family = row["family"]
        evaluator = row.get("evaluator", False)
        if evaluator and not row.get("trainer_ready", False) and row.get("trainer") not in admitted:
            remaining.remove(row)
            continue
        if not evaluator and family in blocked:
            continue
        pending = {f: dict.fromkeys(RESOURCES, 0.0) for f in families}
        heads = {}
        for waiting in remaining:
            if not waiting.get("evaluator"):
                heads.setdefault(waiting["family"], waiting)
        for waiting in remaining:
            # A blocked head trainer reserves the family's guarantee. Later trainers cannot bypass it, while
            # evaluators retain their established exception to head-of-line reservation.
            if (waiting["family"] in blocked and not waiting.get("evaluator", False)
                    and waiting["id"] != blocked[waiting["family"]]):
                continue
            if not possible(waiting) or (not waiting.get("evaluator") and not possible(heads[waiting["family"]])):
                continue
            if (waiting.get("evaluator") and not waiting.get("trainer_ready")
                    and waiting.get("trainer") not in admitted):
                continue
            for k, v in demand(waiting).items():
                pending[waiting["family"]][k] += v
        gpu = choose(row, family)
        need = demand(row)
        if gpu is None or not permits(family, need, usage, capacity, pending, total):
            if not evaluator:
                blocked[family] = row["id"]
                if gpu is None and possible(row):
                    reserve_for(row, family)
            continue
        admitted.append(row["id"])
        assigned[row["id"]] = gpu
        remaining.remove(row)
        slots[gpu]["jobs"] += 1
        slots[gpu]["mem_gb"] += float(row.get("mem_gb", 0))
        for k, v in need.items():
            usage[family][k] += v
            total[k] += v
    guarantee = {k: capacity[k] / len(families) for k in RESOURCES} if families else {}
    return {"admitted": admitted, "assigned": assigned, "reserved": reserved, "blocked": blocked,
            "total_ram_bytes": total["ram_bytes"],
            "report": {"capacity": capacity, "families": {
                f: {"usage": usage[f], "guarantee": guarantee, "dominant_share": dominant(usage[f], capacity)}
                for f in families}}}
