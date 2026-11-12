"""Shared remote capacity admission, embedded by launchers under admission.lock.

SOURCE is stdlib-only and runs as ``python3 -I -c SOURCE <json>`` on the host.
Exit 0 admits, 20 refuses a resource decision, 1 means evidence/input error.
Only tests supply ``test_paths``; launchers construct the input themselves and
must never copy path overrides from a job spec. No cached evidence is accepted. Byte-based resource formulas and tenant validation
come from admission.py, also used by scheduler placement on the same snapshot.
"""

from ops.containers.admission import SOURCE as ADMISSION_SOURCE

SOURCE = ADMISSION_SOURCE + r'''
import json
import math
import re
import subprocess
import sys
from pathlib import Path


def run(program, *args):
    return subprocess.run([program, *args], check=True, capture_output=True,
                          text=True, timeout=30).stdout


def instance(listing, query, uuid, parent, gi, device):
    # The MIG_PROBE parsing rule: exact UUID, parent and device, framebuffer,
    # and a complete, readable parent process listing with instance attribution.
    if listing.count("(UUID: " + uuid + ")") != 1:
        raise ValueError("MIG UUID missing or duplicated")
    owner = None
    for line in listing.splitlines():
        if re.fullmatch(r"GPU \d+: .* \(UUID: GPU-[0-9a-f-]+\)\s*", line):
            owner = line.rsplit("(UUID: ", 1)[1].rstrip(") ")
        elif "(UUID: " + uuid + ")" in line:
            match = re.fullmatch(r"\s+MIG \S+\s+Device\s+(\d+): \(UUID: " + re.escape(uuid) + r"\)\s*", line)
            if owner != parent or not match or int(match[1]) != device:
                raise ValueError("MIG UUID parent/device mismatch")
    section = sub = current = process = None
    devices, processes, query_parents, modes = [], [], [], {}
    listed = 0
    for line in query.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, value = (part.strip() for part in line.partition(":"))
        if indent == 0 or indent <= 4:
            if process is not None:
                raise ValueError("unfinished MIG process record")
            section, sub = None, None
        if indent == 0:
            continue
        if indent <= 4:
            section = key
            if key == "GPU UUID":
                query_parents.append(value)
            if key == "MIG Device":
                current = {}
                devices.append(current)
            if key == "Processes":
                if value not in ("", "None"):
                    raise ValueError("unreadable MIG processes")
                listed += 1
            continue
        if section == "MIG Mode" and key in ("Current", "Pending"):
            modes[key] = value
        if section == "MIG Device" and current is not None:
            if indent == 8 and not sep:
                sub = key
            elif indent == 8 and key in ("Index", "GPU Instance ID"):
                current[key] = decimal(value, key)
            elif sub == "FB Memory Usage" and key in ("Total", "Used"):
                match = re.fullmatch(r"(\d+) MiB", value)
                if not match:
                    raise ValueError("unreadable MIG framebuffer")
                current[key] = int(match[1])
        elif section == "Processes":
            if key == "GPU instance ID":
                if process is not None:
                    raise ValueError("unfinished MIG process record")
                process = [decimal(value, key)]
            elif key == "Process ID":
                if process is None or len(process) != 1:
                    raise ValueError("process without MIG instance")
                process.append(decimal(value, key))
            elif key == "Used GPU Memory":
                if process is None or len(process) != 2:
                    raise ValueError("unreadable MIG process record")
                match = re.fullmatch(r"(\d+) MiB", value)
                if not match:
                    raise ValueError("unreadable process memory")
                processes.append([*process, int(match[1])])
                process = None
    if process is not None or listed != 1 or query_parents != [parent]:
        raise ValueError("incomplete MIG parent/process evidence")
    if modes != {"Current": "Enabled", "Pending": "Enabled"}:  # Codex review 2, finding 9: re-checked inside the lock
        raise ValueError(f"parent MIG mode {modes} is not Enabled/Enabled")
    found = [item for item in devices if item.get("Index") == device]
    if (len(found) != 1 or found[0].get("GPU Instance ID") != gi
            or "Total" not in found[0] or "Used" not in found[0]):
        raise ValueError("MIG instance/framebuffer mismatch")
    total, used = found[0]["Total"], found[0]["Used"]
    if total <= 0 or not 0 <= used <= total:
        raise ValueError("invalid MIG framebuffer")
    return total, used, {pid: memory * 1048576 for instance_id, pid, memory in processes if instance_id == gi}


def decide(data):
    uuid, parent = data["uuid"], data["parent"]
    if not re.fullmatch(r"MIG-[0-9a-f-]+", uuid) or not re.fullmatch(r"GPU-[0-9a-f-]+", parent):
        raise ValueError("invalid device identity")
    gi = integer(data["gpu_instance_id"], "gpu_instance_id")
    device = integer(data["device"], "device")
    max_jobs = integer(data["max_jobs"], "max_jobs", 1)
    exclusive = data.get("exclusive_gpu", False)
    if type(exclusive) is not bool:
        raise ValueError("invalid exclusive_gpu")
    threshold = integer(data["idle_mib"], "idle_mib")
    need = integer(data["gpu_bytes"], "gpu_bytes", 1)
    capacity = integer(data["capacity_bytes"], "capacity_bytes", 1)
    ram = integer(data["ram_bytes"], "ram_bytes", 1)
    reserve = integer(data["reserve_bytes"], "reserve_bytes")
    disk_need = integer(data["disk_bytes"], "disk_bytes")
    floor = integer(data["floor_bytes"], "floor_bytes")
    fraction = data["disk_floor_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0.15 <= fraction <= 1:
        raise ValueError("invalid disk floor fraction")
    paths = data.get("test_paths", {})
    smi, docker = paths.get("nvidia_smi", "nvidia-smi"), paths.get("docker", "docker")
    total, used, processes = instance(run(smi, "-L"), run(smi, "-q", "-i", parent), uuid, parent, gi, device)
    ids = run(docker, "ps", "-q", "--no-trunc", "--filter", "label=s4d.project=splatter4d",
              "--filter", "label=s4d.owner=kaist-sunho").split()
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[0-9a-f]{64}", cid) for cid in ids):
        raise ValueError("invalid live container listing")
    records = []
    for cid in ids:
        inspected = json.loads(run(docker, "inspect", cid))
        if not isinstance(inspected, list) or len(inspected) != 1 or inspected[0]["Id"] != cid:
            raise ValueError("invalid container inspection")
        item = inspected[0]
        labels = item["Config"]["Labels"]
        if labels.get("s4d.project") != "splatter4d" or labels.get("s4d.owner") != "kaist-sunho":
            raise ValueError("container ownership changed during inspection")
        state = item["State"]
        if not (state["Running"] or state.get("Restarting", False) or state.get("Paused", False)):
            continue
        pids = None
        if labels.get("s4d.gpu") == uuid and labels.get("s4d.role") != "migval":
            rows = run(docker, "top", cid, "-eo", "pid").strip().splitlines()
            if not rows or rows[0].strip().lower() != "pid" or any(not row.strip().isdigit() for row in rows[1:]):
                raise ValueError("unreadable container process list")
            pids = {int(row) for row in rows[1:]}
        cgroup = Path(paths.get("cgroup_root", "/sys/fs/cgroup/system.slice")) / ("docker-" + cid + ".scope")
        # non-reclaimable usage (anon + shmem): page cache is already counted as available in MemAvailable (review 3, 15)
        stat = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines() if len(line.split()) == 2)
        ram_used = decimal(stat.get("anon"), "cgroup anon") + decimal(stat.get("shmem"), "cgroup shmem")
        records.append((item, pids, ram_used))
    totals = tenant_totals(uuid, processes, records)
    reasons = gpu_reasons(total, used, need, capacity, totals['gpu_reserved'], totals['gpu_pending'],
                          totals['count'], max_jobs, exclusive, totals['tenant_exclusive'],
                          bool(set(processes) - totals['known']), threshold)
    cpu_need = integer(data.get("cpu_threads", 8), "cpu_threads", 1)
    cpu_budget = integer(data.get("cpu_thread_budget", 320), "cpu_thread_budget", 1)
    meminfo = Path(paths.get("meminfo", "/proc/meminfo")).read_text()
    available = re.findall(r"^MemAvailable:\s+(\d+) kB\s*$", meminfo, re.MULTILINE)
    if len(available) != 1:
        raise ValueError("unreadable MemAvailable")
    available = int(available[0]) * 1024
    disk = run(paths.get("df", "df"), "-B1", "--output=size,avail", "--", data["root"])
    rows = [line.split() for line in disk.splitlines() if re.fullmatch(r"\s*\d+\s+\d+\s*", line)]
    if len(rows) != 1:
        raise ValueError("unreadable disk evidence")
    disk_total, disk_free = map(int, rows[0])
    if disk_total <= 0 or not 0 <= disk_free <= disk_total:
        raise ValueError("invalid disk evidence")
    reasons += host_reasons(available, totals['pending_ram'], ram, reserve, totals['cpu_used'],
                            cpu_need * 10**9, cpu_budget * 10**9, disk_total, disk_free,
                            totals['reserved_disk'], disk_need, floor, fraction)
    return {"admit": not reasons, "reasons": reasons, "uuid": uuid, "containers": totals['count'],
            "used_mib": used, "gpu_reserved_bytes": totals['gpu_reserved'],
            "gpu_pending_bytes": totals['gpu_pending'], "pending_ram_bytes": totals['pending_ram'],
            "reserved_disk_bytes": totals['reserved_disk']}



try:
    result = decide(json.loads(sys.argv[1]))
except Exception as exc:
    print(json.dumps({"admit": False, "error": type(exc).__name__ + ": " + str(exc)}))
    sys.exit(1)
print(json.dumps(result, sort_keys=True))
sys.exit(0 if result["admit"] else 20)
'''
