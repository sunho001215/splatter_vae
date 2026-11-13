"""Queue priority and non-preemptive admission barriers (no I/O)."""


class PriorityBarrier:
    """After a ready row cannot fit, only its priority or higher may consume this host's capacity.

    Callers offer rows in ascending queue priority, preserving their established order on ties.
    Holding the whole host avoids unsafe backfill without runtime estimates or preemption.
    Recompute on every pass so running jobs drain and the waiting row gets first use of capacity.
    """

    def __init__(self):
        self.blocked = {}

    def allows(self, row, host="host"):
        return row.get("priority", 5) <= self.blocked.get(host, float("inf"))

    def block(self, row, host="host"):
        self.blocked[host] = min(row.get("priority", 5), self.blocked.get(host, float("inf")))
