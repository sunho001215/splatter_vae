"""Polling of a release's verification jobs (ops/campaign/release_chain.py; user 2026-11-12: release turnaround).

The chain ticks the scheduler only when something can change for its jobs: right after admission attempts, and when a
running verification container stops. Between ticks ``wait_for_exit`` watches the running containers with one cheap
``RemoteBackend.container_states`` call per poll (no snapshot, no registry lock) and returns as soon as one is no longer
running, after ``max_quiet`` seconds (the tick also refreshes the live evidence copy), or on any query error (the tick
decides; this module never changes state).
"""

from __future__ import annotations

import time

DONE, FAILED, RUNNING = "done", "failed", "running"


def next_action(state: dict, ids: list[str]) -> tuple[str, dict[str, str]]:
    """('done' | 'failed' | 'watch' | 'retry', {job id: container name to watch}).
    watch: every unfinished job is running in a known container; retry: an admission, launch or result sync is pending
    (tick again soon)."""
    status = {i: state.get(i, {}).get("status") for i in ids}
    if all(s == DONE for s in status.values()):
        return "done", {}
    if any(s == FAILED for s in status.values()):
        return "failed", {}
    running = {i: state[i]["container"] for i in ids if status[i] == RUNNING and state[i].get("container")}
    unfinished = [i for i in ids if status[i] != DONE]
    return ("watch", running) if running and len(running) == len(unfinished) else ("retry", {})


def wait_for_exit(container_states, names: list[str], *, renew=lambda: None, poll_seconds: float = 10.0,
                  max_quiet: float = 240.0, sleep=time.sleep, clock=time.time, log=lambda message: None) -> str:
    """Block until one of ``names`` is no longer running; returns 'stopped', 'quiet' (max_quiet elapsed) or 'error'."""
    deadline = clock() + max_quiet
    while clock() < deadline:
        sleep(poll_seconds)
        renew()
        try:
            states = container_states(names)
        except Exception as exc:  # noqa: BLE001  (the next scheduler tick re-checks everything)
            log(f"container poll failed ({type(exc).__name__}); ticking")
            return "error"
        if any(states.get(name) != RUNNING for name in names):
            return "stopped"
    return "quiet"
