"""Measured cgroup usage of our remote containers, and limits proposed from it (ARCH Phase 2, 2026-11-11).

Campaign containers run without --memory/--cpus today (probe: memory.max=max). Before enforcing limits on new attempts
(queue hosts.remote.enforce_resource_limits: true -> remote_backend.prepare / launch.py), their declarations are sized
from MEASURED usage: ``sample`` reads, read-only, the cgroup files of every running container of ours
(memory.peak exists only while the cgroup lives, so it is sampled periodically over full runs), and ``propose`` turns
the samples into per-row RAM/CPU proposals. Nothing here changes a declaration or a container.

    python -I ops/remote/cgroup_usage.py sample  [--out runs/remote/cgroup-usage.jsonl]      (one read-only SSH)
    python -I ops/remote/cgroup_usage.py propose [--samples runs/remote/cgroup-usage.jsonl]  (prints proposals)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GIB = 2**30
RAM_FACTOR, RAM_MARGIN_GIB = 1.25, 1.0
CPU_FACTOR, CPU_MIN, CPU_MAX = 1.25, 2, 32
# page cache and tmpfs are charged to the cgroup too (memmapped replay, /dev/shm): peak, not RSS, is what must fit

SAMPLE_SHELL = (
    "set -u; for id in $(docker ps -q --no-trunc --filter label=s4d.project=splatter4d "
    "--filter label=s4d.owner=kaist-sunho); do "
    "name=$(docker inspect --format '{{.Name}}' \"$id\" 2>/dev/null) || continue; "
    "scope=/sys/fs/cgroup/system.slice/docker-$id.scope; "
    "peak=$(cat \"$scope/memory.peak\" 2>/dev/null || echo -); cur=$(cat \"$scope/memory.current\" 2>/dev/null || echo -); "
    "lim=$(cat \"$scope/memory.max\" 2>/dev/null || echo -); "
    "oom=$(awk '$1==\"oom_kill\" {print $2}' \"$scope/memory.events\" 2>/dev/null); "
    "cpu=$(awk '$1==\"usage_usec\" {print $2}' \"$scope/cpu.stat\" 2>/dev/null); "
    "printf 'S4D_CG|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"${name#/}\" \"$id\" \"$(date +%s)\" \"$peak\" \"$cur\" \"$lim\" "
    "\"${oom:--}\" \"${cpu:--}\"; done")


def _int(value: str):
    return int(value) if value.isdigit() else None


def parse_samples(output: str) -> list[dict]:
    rows = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) != 9 or parts[0] != "S4D_CG" or not parts[1].startswith("s4d-") or len(parts[2]) != 64:
            continue
        rows.append({"container": parts[1], "id": parts[2], "at": _int(parts[3]), "memory_peak": _int(parts[4]),
                     "memory_current": _int(parts[5]), "memory_max": None if parts[6] == "max" else _int(parts[6]),
                     "oom_kill": _int(parts[7]), "cpu_usage_usec": _int(parts[8])})
    return rows


def job_of(container: str) -> str | None:
    """s4d-<job>-a<N> -> <job> (campaign / verification); rwprep containers are named s4d-rwprep-<job>-a<N>."""
    head, sep, attempt = container.rpartition("-a")
    return head.removeprefix("s4d-") if sep and attempt.isdigit() else None


def propose(samples: list[dict], rows: dict[str, dict] | None = None) -> dict:
    """Per job: measured peak / CPU cores and proposed ram_gb / cpu_threads (declared values shown for comparison)."""
    by_container: dict[str, list[dict]] = {}
    for sample in samples:  # one container ID per attempt: a re-created name is a separate series
        by_container.setdefault(f"{sample['container']}@{sample['id'][:12]}", []).append(sample)
    out = {}
    for container, items in sorted(by_container.items()):
        job = job_of(container.split("@", 1)[0])
        items.sort(key=lambda item: item["at"] or 0)
        readable = [item for item in items if item["memory_peak"] is not None and item["at"] is not None]
        peak = max((item["memory_peak"] for item in readable), default=None)
        span = (readable[-1]["at"] - readable[0]["at"]) if len(readable) > 1 else 0
        ooms = [item["oom_kill"] for item in items if item["oom_kill"] is not None]
        cores = []
        for before, after in zip(items, items[1:]):
            if None not in (before["cpu_usage_usec"], after["cpu_usage_usec"], before["at"], after["at"]) \
                    and after["at"] > before["at"] and after["cpu_usage_usec"] >= before["cpu_usage_usec"]:
                cores.append((after["cpu_usage_usec"] - before["cpu_usage_usec"]) / 1e6 / (after["at"] - before["at"]))
        cores.sort()
        high = cores[min(len(cores) - 1, math.ceil(0.95 * len(cores)) - 1)] if cores else None
        row = (rows or {}).get(job, {})
        out[container] = {
            "job": job, "samples": len(items), "readable_samples": len(readable),
            "memory_peak_gib": None if peak is None else round(peak / GIB, 2),
            "oom_kills": max(ooms) if ooms else None,  # None: unreadable, never reported as zero
            "cpu_cores_p95": None if high is None else round(high, 2),
            "proposed_ram_gb": math.ceil((peak * RAM_FACTOR) / GIB + RAM_MARGIN_GIB) if peak else None,
            "proposed_cpu_threads": None if high is None else min(CPU_MAX, max(CPU_MIN, math.ceil(high * CPU_FACTOR))),
            "declared_ram_gb": row.get("ram_gb"), "declared_cpu_threads": row.get("cpu_threads", 8 if row else None),
            # >= 12 readable samples spanning >= 1 h of this container; full runs per row family are better still
            "enough_evidence": len(readable) >= 12 and span >= 3600 and bool(ooms),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("sample", "propose"))
    parser.add_argument("--out", type=Path, default=REPO / "runs/remote/cgroup-usage.jsonl")
    parser.add_argument("--samples", type=Path, default=REPO / "runs/remote/cgroup-usage.jsonl")
    parser.add_argument("--queue", type=Path, default=REPO / "experiments/queue.yaml")
    args = parser.parse_args(argv)
    if args.command == "sample":
        sys.path.insert(0, str(REPO))
        from ops.remote.access import RemoteClient  # noqa: PLC0415

        output = RemoteClient().checked_ssh(SAMPLE_SHELL, wait=False, idempotent=True, timeout=120).stdout
        rows = parse_samples(output)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as stream:
            for row in rows:
                stream.write(json.dumps({**row, "sampled_local": time.time()}) + "\n")
        print(f"{len(rows)} containers sampled -> {args.out}")
        return 0
    import yaml  # noqa: PLC0415

    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    queue = yaml.safe_load(args.queue.read_text()) or {}
    rows = {job["id"]: job for job in queue.get("jobs", [])}
    print(json.dumps(propose(samples, rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
