"""Remote MIG inventory: discovery, validation and the reviewed allowlist file (user directive 2026-11-11 ~19:00).

The remote host's four authorized GPUs (REMOTE_PHYSICAL_GPUS) are to be split by the user into 4 x MIG 1g.24gb each.
Nothing here configures MIG. This module only

* parses read-only ``nvidia-smi -L`` and ``nvidia-smi -q -i <parents>`` output (the layouts of driver 580),
* builds an inventory of the usable instances on the AUTHORIZED parents only (other GPUs are never listed): an instance
  is usable when its parent is in MIG mode (current == pending == Enabled), its profile is the expected one, its
  framebuffer is at least ``min_memory_mib``, its UUID is unique and no process outside ``owned_pids`` runs on it.
  Every other instance is excluded WITH its reason (reported, never used);
* validates the reviewed allowlist file ``configs/hosts/remote-mig.yaml`` that ``ops/resources/gpu_guard.py`` reads:
  absent -> the physical allowlist is unchanged; present but invalid -> NO remote device is authorized (fail closed).

The allowlist is deliberately a committed file, not a live query: the in-container guard of a release must authorize
exactly the instances that release was verified on (native suite + one isolation check per instance).

    python -I ops/resources/mig_inventory.py discover --out <report.json>          (read-only SSH; exit 3: incomplete)
    python -I ops/resources/mig_inventory.py render --report <report.json> --repo <checkout>   (writes 2 config files)
    python -I ops/resources/mig_inventory.py check [--repo <checkout>]              (validates the committed file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import yaml

SCHEMA = "s4d-remote-mig-inventory/1"
# The remote host's authorized physical GPUs (user 2026-10-10; GPU 0-3 of 192.168.10.22). After MIG they are the
# parents of the authorized instances and are themselves never a launch target again.
REMOTE_PHYSICAL_GPUS = (
    "GPU-4391fcee-537f-d408-d138-10d8a3866eea",
    "GPU-6122c7a9-eaa6-b539-6005-f20e339da4ea",
    "GPU-9e9f1e97-b2ca-04e0-eb2a-8035398daa79",
    "GPU-ca65e1b7-c0ac-6757-7d3f-816c3119a4c1",
)
EXPECTED_PROFILE = "1g.24gb"
INSTANCES_PER_PARENT = 4
# 1g.24gb on the RTX PRO 6000 Blackwell (local GPUs 4/5): 24192 MiB framebuffer per instance
MIN_MEMORY_MIB = 23_000
# Admission headroom inside one instance: CUDA context, allocator fragmentation, the launch-guard margin
INSTANCE_RESERVE_GIB = 1.0
# Load units per instance (one GPU-heavy row = 1.0, an FQL / RL row = 0.5); physical GPUs keep their own limit
INSTANCE_LOAD_LIMIT = 1.0
PHYSICAL_MEMORY_GIB = 90.0
INVENTORY_RELATIVE = "configs/hosts/remote-mig.yaml"
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
MIG_UUID = re.compile(r"MIG-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_LIST_GPU = re.compile(r"GPU (\d+): (.+?) \(UUID: (GPU-[0-9a-f-]+)\)\s*\Z")
_LIST_MIG = re.compile(r"\s+MIG (\S+)\s+Device\s+(\d+): \(UUID: (MIG-[0-9a-f-]+)\)\s*\Z")
_QUERY_GPU = re.compile(r"GPU [0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f]")  # "GPU 00000000:C5:00.0"


class InventoryError(ValueError):
    """The MIG evidence or the allowlist file cannot be trusted."""


def parse_list(text: str) -> list[dict]:
    """``nvidia-smi -L`` -> [{"index", "uuid", "name", "mig": [{"profile", "device", "uuid"}]}] in listing order."""
    parents: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        gpu, mig = _LIST_GPU.fullmatch(line.rstrip()), _LIST_MIG.fullmatch(line.rstrip())
        if gpu:
            if not GPU_UUID.fullmatch(gpu[3]):
                raise InventoryError(f"malformed GPU UUID in nvidia-smi -L: {gpu[3]!r}")
            parents.append({"index": int(gpu[1]), "name": gpu[2], "uuid": gpu[3], "mig": []})
        elif mig:
            if not parents or not MIG_UUID.fullmatch(mig[3]):
                raise InventoryError(f"MIG line without its GPU or malformed UUID: {line!r}")
            parents[-1]["mig"].append({"profile": mig[1], "device": int(mig[2]), "uuid": mig[3]})
        else:
            raise InventoryError(f"unrecognized nvidia-smi -L line: {line!r}")
    return parents


def _mib(value: str) -> int | None:
    match = re.fullmatch(r"(\d+) MiB", value.strip())
    return int(match[1]) if match else None


def parse_query(text: str) -> dict[str, dict]:
    """``nvidia-smi -q -i <parents>`` -> {parent UUID: {"mig_current", "mig_pending", "instances": {device index:
    {"gpu_instance_id", "compute_instance_id", "total_mib", "used_mib"}}, "processes": [{"gpu_instance_id", "pid",
    "used_mib"}]}}. A process listing that is not readable ("Insufficient Permissions", ...) raises: ownership of an
    instance that cannot be inspected is never assumed."""
    gpus: dict[str, dict] = {}
    current: dict | None = None
    section = sub = None
    device: dict | None = None
    process: dict | None = None

    def close():
        if current is not None:
            uuid = current.get("uuid")
            if not isinstance(uuid, str) or not GPU_UUID.fullmatch(uuid):
                raise InventoryError("nvidia-smi -q section without a valid GPU UUID")
            if uuid in gpus:
                raise InventoryError(f"GPU {uuid} listed twice")
            gpus[uuid] = current

    for line in text.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, separator, value = (part.strip() for part in line.partition(":"))
        if indent == 0:
            if _QUERY_GPU.fullmatch(line.rstrip()):
                close()
                current = {"uuid": None, "mig_current": None, "mig_pending": None, "instances": {}, "processes": []}
            section = None
            continue
        if current is None:
            continue
        if indent <= 4:
            section, sub, device, process = key, None, None, None
            if key == "GPU UUID":
                current["uuid"] = value
            elif key == "MIG Device":
                device = {"gpu_instance_id": None, "compute_instance_id": None, "total_mib": None, "used_mib": None}
            elif key == "Processes" and value not in ("", "None"):
                raise InventoryError(f"GPU process listing unavailable: {value}")
            continue
        if section == "MIG Mode" and key in ("Current", "Pending"):
            current["mig_current" if key == "Current" else "mig_pending"] = value
        elif section == "MIG Device" and device is not None:
            if indent == 8 and not separator:
                sub = key
            elif indent == 8 and key == "Index":
                if not value.isdigit() or int(value) in current["instances"]:
                    raise InventoryError(f"invalid or repeated MIG device index {value!r}")
                current["instances"][int(value)] = device
            elif indent == 8 and key in ("GPU Instance ID", "Compute Instance ID"):
                if not value.isdigit():
                    raise InventoryError(f"invalid MIG {key}: {value!r}")
                device["gpu_instance_id" if key.startswith("GPU") else "compute_instance_id"] = int(value)
            elif sub == "FB Memory Usage" and key in ("Total", "Used"):
                device[f"{key.lower()}_mib"] = _mib(value)
        elif section == "Processes":
            if key == "GPU instance ID":
                process = {"gpu_instance_id": int(value) if value.isdigit() else None, "pid": None, "used_mib": None}
            elif key == "Process ID" and process is not None:
                if not value.isdigit():
                    raise InventoryError(f"unreadable process id {value!r}")
                process["pid"] = int(value)
            elif key == "Used GPU Memory" and process is not None:
                process["used_mib"] = _mib(value)
                current["processes"].append(process)
                process = None
    close()
    return gpus


def build_inventory(list_text: str, query_text: str, *, authorized=REMOTE_PHYSICAL_GPUS, owned_pids=frozenset(),
                    profile: str = EXPECTED_PROFILE, per_parent: int = INSTANCES_PER_PARENT,
                    min_memory_mib: int = MIN_MEMORY_MIB) -> dict:
    """The inventory report of the usable instances on the authorized parents (see the module docstring)."""
    listing = parse_list(list_text)
    query = parse_query(query_text)
    by_uuid = {item["uuid"]: item for item in listing}
    seen = [mig["uuid"] for item in listing for mig in item["mig"]]
    duplicated = {uuid for uuid in seen if seen.count(uuid) > 1}
    instances, excluded, problems, parents = [], [], [], []
    for parent in authorized:
        listed = by_uuid.get(parent)
        info = query.get(parent)
        if listed is None or info is None:
            problems.append(f"{parent}: authorized GPU missing from {'nvidia-smi -L' if listed is None else '-q'}")
            continue
        parents.append({"uuid": parent, "index": listed["index"]})
        if info["mig_current"] != "Enabled" or info["mig_pending"] != "Enabled":
            problems.append(f"{parent}: MIG mode current={info['mig_current']} pending={info['mig_pending']} "
                            "(MIG not (fully) enabled: no instance of this GPU is used)")
            excluded += [{"uuid": mig["uuid"], "parent": parent, "reason": "parent MIG mode not Enabled/Enabled"}
                         for mig in listed["mig"]]
            continue
        busy = {}
        for proc in info["processes"]:
            if proc["pid"] is None or proc["gpu_instance_id"] is None:
                problems.append(f"{parent}: a process without a readable PID / GPU instance (ownership unknown)")
                busy = None
                break
            if proc["pid"] not in owned_pids:
                busy.setdefault(proc["gpu_instance_id"], []).append(proc["pid"])
        usable = []
        for mig in listed["mig"]:
            device = info["instances"].get(mig["device"])
            reason = None
            if mig["uuid"] in duplicated:
                reason = "UUID listed more than once"
            elif device is None or device["gpu_instance_id"] is None:
                reason = "MIG device missing from nvidia-smi -q"
            elif mig["profile"] != profile:
                reason = f"profile {mig['profile']} != {profile}"
            elif device["total_mib"] is None or device["total_mib"] < min_memory_mib:
                reason = f"framebuffer {device['total_mib']} MiB < {min_memory_mib} MiB"
            elif busy is None:
                reason = "process ownership on the parent unreadable"
            elif busy.get(device["gpu_instance_id"]):
                reason = f"foreign processes {sorted(busy[device['gpu_instance_id']])} (used by others)"
            if reason:
                excluded.append({"uuid": mig["uuid"], "parent": parent, "reason": reason})
                continue
            usable.append({"uuid": mig["uuid"], "parent": parent, "parent_index": listed["index"],
                           "device": mig["device"], "gpu_instance_id": device["gpu_instance_id"],
                           "compute_instance_id": device["compute_instance_id"], "profile": mig["profile"],
                           "memory_total_mib": device["total_mib"], "memory_used_mib": device["used_mib"]})
        if len(usable) != per_parent:
            problems.append(f"{parent}: {len(usable)} usable {profile} instances, expected {per_parent}")
        if len({item["gpu_instance_id"] for item in usable}) != len(usable):
            problems.append(f"{parent}: repeated GPU instance id")
            excluded += [{"uuid": item["uuid"], "parent": parent, "reason": "repeated GPU instance id"} for item in usable]
            usable = []
        instances += usable
    evidence = hashlib.sha256((list_text + "\0" + query_text).encode()).hexdigest()
    return {"schema": SCHEMA, "host": "remote", "profile": profile, "instances_per_parent": per_parent,
            "min_memory_mib": min_memory_mib, "parents": parents, "instances": instances, "excluded": excluded,
            "problems": problems, "complete": not problems and len(instances) == per_parent * len(authorized),
            "evidence_sha256": evidence, "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def allowlist_document(report: dict) -> dict:
    """The committed allowlist (configs/hosts/remote-mig.yaml) for the usable instances of a report."""
    keys = ("uuid", "parent", "parent_index", "device", "gpu_instance_id", "compute_instance_id", "profile",
            "memory_total_mib")
    doc = {"schema": SCHEMA, "host": "remote", "profile": report["profile"],
           "instances_per_parent": report["instances_per_parent"], "min_memory_mib": report["min_memory_mib"],
           "parents": [p["uuid"] for p in report["parents"]],
           "instances": [{k: item[k] for k in keys} for item in report["instances"]],
           "excluded": report["excluded"], "evidence_sha256": report["evidence_sha256"],
           "discovered_at": report["discovered_at"]}
    validate_allowlist(doc)
    return doc


def validate_allowlist(doc, authorized=REMOTE_PHYSICAL_GPUS, profile: str = EXPECTED_PROFILE) -> tuple[dict, ...]:
    """Validated instance records of an allowlist document, or InventoryError. A subset of the 16 instances is allowed
    (bad / missing / foreign ones are excluded and reported), never an instance outside the authorized parents, of
    another profile than the directive's, or below the minimum framebuffer."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("host") != "remote":
        raise InventoryError("not a remote MIG allowlist")
    if doc.get("profile") != profile:
        raise InventoryError(f"allowlist profile {doc.get('profile')!r} is not {profile}")
    minimum = doc.get("min_memory_mib")
    if type(minimum) is not int or minimum < MIN_MEMORY_MIB:
        raise InventoryError("allowlist min_memory_mib missing or below the module minimum")
    per_parent = doc.get("instances_per_parent")
    if type(per_parent) is not int or not 1 <= per_parent <= 7:
        raise InventoryError("invalid instances_per_parent")
    if (not isinstance(doc.get("parents"), list) or not all(isinstance(x, str) for x in doc["parents"])
            or not set(doc["parents"]) <= set(authorized)):
        raise InventoryError("allowlist names a parent outside the authorized GPUs")
    items = doc.get("instances")
    if not isinstance(items, list) or not items:
        raise InventoryError("allowlist has no instance")
    seen, slots = set(), set()
    for item in items:
        if not isinstance(item, dict) or not MIG_UUID.fullmatch(str(item.get("uuid", ""))):
            raise InventoryError(f"invalid MIG instance record {item!r}")
        if item.get("parent") not in authorized or item["parent"] not in doc["parents"]:
            raise InventoryError(f"{item['uuid']}: parent not authorized")
        if item.get("profile") != doc.get("profile"):
            raise InventoryError(f"{item['uuid']}: profile differs from the allowlist profile")
        if any(type(item.get(k)) is not int or item[k] < 0
               for k in ("parent_index", "device", "gpu_instance_id", "compute_instance_id", "memory_total_mib")):
            raise InventoryError(f"{item['uuid']}: invalid numeric field")
        if item["memory_total_mib"] < minimum:
            raise InventoryError(f"{item['uuid']}: framebuffer below the minimum")
        if item["uuid"] in seen or (item["parent"], item["gpu_instance_id"]) in slots:
            raise InventoryError(f"{item['uuid']}: repeated instance")
        seen.add(item["uuid"])
        slots.add((item["parent"], item["gpu_instance_id"]))
    for parent in doc["parents"]:
        if sum(1 for item in items if item["parent"] == parent) > per_parent:
            raise InventoryError(f"{parent}: more instances than instances_per_parent")
    return tuple(dict(item) for item in items)


def load_allowlist(path: Path, authorized=REMOTE_PHYSICAL_GPUS) -> tuple[dict, ...] | None:
    """None when the file is absent (physical allowlist); the validated instances otherwise; InventoryError if
    present but not trustworthy."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InventoryError(f"unreadable MIG allowlist ({type(exc).__name__})") from exc
    return validate_allowlist(doc, authorized)


def device_capacity_gib(uuid: str, instances=()) -> float:
    """Admission capacity (GiB) of one remote device: an instance's framebuffer less INSTANCE_RESERVE_GIB; a physical
    GPU keeps the campaign's 90 GiB cap."""
    for item in instances:
        if item["uuid"] == uuid:
            return round(item["memory_total_mib"] / 1024 - INSTANCE_RESERVE_GIB, 3)
    return PHYSICAL_MEMORY_GIB


def queue_limits(instances, max_jobs: int = 2) -> dict:
    """``hosts.remote.limits`` for experiments/queue.yaml: one entry per instance (the scheduler refuses limits for
    non-authorized devices, so this block must replace the physical one in the same change as the allowlist)."""
    return {item["uuid"]: {"max_jobs": int(max_jobs), "max_mem_gb": device_capacity_gib(item["uuid"], instances)}
            for item in instances}


def oversized_rows(jobs, instances) -> list[tuple[str, float]]:
    """Remote rows whose declared GPU memory exceeds every instance's capacity (they can never be placed)."""
    largest = max((device_capacity_gib(item["uuid"], instances) for item in instances), default=0.0)
    return [(job["id"], float(job["mem_gb"])) for job in jobs
            if job.get("host") == "remote" and float(job.get("mem_gb", 0)) > largest]


def _remote_evidence() -> tuple[str, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ops.remote.access import RemoteClient  # noqa: PLC0415  (read-only SSH)

    client = RemoteClient()
    listing = client.checked_ssh("nvidia-smi -L", wait=False, idempotent=True, timeout=120).stdout
    parents = ",".join(REMOTE_PHYSICAL_GPUS)
    query = client.checked_ssh(f"nvidia-smi -q -i {parents}", wait=False, idempotent=True, timeout=180).stdout
    return listing, query


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover")
    discover.add_argument("--out", type=Path, required=True)
    discover.add_argument("--list-file", type=Path, help="saved nvidia-smi -L output instead of SSH")
    discover.add_argument("--query-file", type=Path, help="saved nvidia-smi -q -i <parents> output instead of SSH")
    discover.add_argument("--owned-pids", default="", help="comma-separated PIDs of our own containers")
    render = sub.add_parser("render")
    render.add_argument("--report", type=Path, required=True)
    render.add_argument("--repo", type=Path, required=True)
    render.add_argument("--allow-incomplete", action="store_true")
    check = sub.add_parser("check")
    check.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.command == "discover":
        if args.list_file and args.query_file:
            listing, query = args.list_file.read_text(), args.query_file.read_text()
        else:
            listing, query = _remote_evidence()
        owned = frozenset(int(p) for p in args.owned_pids.split(",") if p.strip())
        report = build_inventory(listing, query, owned_pids=owned)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".nvidia-smi-L.txt").write_text(listing)
        args.out.with_suffix(".nvidia-smi-q.txt").write_text(query)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({k: report[k] for k in ("complete", "problems")} | {
            "usable": len(report["instances"]), "excluded": report["excluded"]}, indent=1))
        return 0 if report["complete"] else 3
    if args.command == "render":
        report = json.loads(args.report.read_text())
        if not report.get("complete") and not args.allow_incomplete:
            print("inventory incomplete; refusing to render (use --allow-incomplete only with the user's approval)")
            return 3
        doc = allowlist_document(report)
        (args.repo / INVENTORY_RELATIVE).write_text(yaml.safe_dump(doc, sort_keys=False))
        host = {"name": "remote", "gpu_uuids": [item["uuid"] for item in doc["instances"]],
                "runtime_root": "/home/compu/kaist/sunho"}
        (args.repo / "configs/hosts/remote.yaml").write_text(yaml.safe_dump(host, sort_keys=False))
        print(yaml.safe_dump({"hosts": {"remote": {"limits": queue_limits(doc["instances"])}}}, sort_keys=False))
        return 0
    instances = load_allowlist(args.repo / INVENTORY_RELATIVE)
    print("physical allowlist (no MIG allowlist file)" if instances is None
          else f"{len(instances)} MIG instances authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
