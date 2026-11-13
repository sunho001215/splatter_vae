"""Pure, per-pass reservations for non-preemptive strict-priority backfill."""

import math

HOST_RESOURCES = ("ram_gb", "cpu_threads", "disk_gb")


def demand(row):
    return {key: float(row.get(key, 0)) for key in HOST_RESOURCES}


def log_unready(rows, ready_ids, state, messages):
    """Explain filtered waiting rows once per pass, including dependency and pause gates."""
    for row in rows:
        current = state.get(row['id'], {})
        waiting = (current.get('status', 'pending') == 'pending'
                   or (current.get('status') == 'crashed'
                       and current.get('failures', 0) <= int(row.get('max_restarts', 1))))
        if (waiting and row['id'] not in ready_ids
                and not any(message.startswith(f"{row['id']}:") for message in messages)):
            messages.append(f"{row['id']}: not ready; no priority reservation")


class PriorityReservations:
    """Call in priority order; ties retain the caller's admission policy.

    Only ready, empty-host-feasible waiting rows reserve. Each owns one distinct
    device and its full host demand. Live occupancy is never subtracted or moved.
    Callbacks expose current occupancy and remaining budgets after each admission.
    Nothing survives the scheduling pass. Units are GiB and CPU threads.
    """

    def __init__(self, limits, budgets, slots, remaining, *, request=demand, messages=None):
        self.limits, self.budgets = limits, budgets
        self.slots, self.remaining, self.request = slots, remaining, request
        self.reservations = {}
        self.messages, self.logged = messages, set()

    def log(self, row, reason):
        if row['id'] not in self.logged:
            self.logged.add(row['id'])
            if self.messages is not None:
                self.messages.append(f"{row['id']}: {reason}; no priority reservation")

    def devices(self, row):
        return [gpu for gpu, limit in self.limits.items()
                if row.get('gpu', 'any') in ('any', gpu) and int(limit['max_jobs']) > 0
                and (row.get('gpu_profile') is None or (row.get('gpu_profile') == '1g.24gb'
                                                       and gpu.startswith('MIG-')))
                and 0 <= float(row.get('mem_gb', 0)) <= float(limit['max_mem_gb'])]

    def feasible(self, row):
        need = self.request(row)
        return bool(self.devices(row)) and all(
            math.isfinite(need.get(k, 0)) and 0 <= need.get(k, 0) <= self.budgets.get(k, 0)
            for k in HOST_RESOURCES)

    def held(self, row):
        return [r for r in self.reservations.values() if r['priority'] < row.get('priority', 5)]

    def excluded(self, row):
        return {r['gpu'] for r in self.held(row)}

    def allows(self, row, gpu=None, *, remaining=None):
        held = self.held(row)
        if not held:
            return True
        free, need = self.remaining() if remaining is None else remaining, self.request(row)
        return (gpu not in {r['gpu'] for r in held} and all(
            free.get(k, 0) - need.get(k, 0) + 1e-9 >= sum(r['need'].get(k, 0) for r in held)
            for k in HOST_RESOURCES))

    def backfill(self, rows):
        """Filter a preplanned dispatch batch cumulatively against higher-tier reservations."""
        free, admitted = dict(self.remaining()), []
        for row in rows:
            if self.allows(row, row.get('gpu'), remaining=free):
                admitted.append(row)
                for key, value in self.request(row).items():
                    free[key] = free.get(key, 0)-value
        return admitted

    def reserve(self, row, *, ready=True):
        if not ready:
            self.log(row, 'not ready')
            return None
        if not self.feasible(row):
            self.log(row, 'infeasible on empty host')
            return None
        if row['id'] in self.reservations:
            return self.reservations[row['id']]['gpu']
        used = {r['gpu'] for r in self.reservations.values()}
        available = [g for g in self.devices(row) if g not in used]
        if not available:
            return None
        slots = self.slots()
        gpu = min(available, key=lambda g: (
            -(float(self.limits[g]['max_mem_gb']) - slots.get(g, {}).get('mem_gb', 0)),
            slots.get(g, {}).get('jobs', 0), list(self.limits).index(g)))
        self.reservations[row['id']] = dict(gpu=gpu, priority=row.get('priority', 5), need=self.request(row))
        return gpu
