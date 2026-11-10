"""Remote command line of the dataset downloader (``scripts/datasets/droid_datasets.py``).

    plan       list the official sources (pinned revisions) into .state/plan.json; print the storage projection
    storage    final + peak footprint of the enabled groups vs free space and the margin
    configure  enable/disable groups, set pointworld_selection (pending|all|<episode list>), margin, pool limits;
               refuses a configuration whose final footprint would cross the margin
    start      reconcile, then start the owner detached (setsid, nice, ionice) unless one is already running
    run        the owner process itself (foreground; start uses it)
    status     live owner check + the last status.json summary
    stop       ask the owner to stop (partials are kept) and wait for the lock to be released
    adopt      verify an existing copy of a planned plain file against its official checksum and install it
    retry      put failed items back to pending
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ops.datasets import catalog, storage
from ops.datasets.runner import DEFAULT_CONFIG, Runner, load_config, resolve_selection, temp_path
from ops.datasets.state import OwnerLock, StateStore, read_json, write_json_atomic
from ops.datasets.transfer import hash_file

DEFAULT_ROOT = "/data_8T/kaist/sunho/datasets"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "datasets" / "droid_datasets.py"


def tools_sha() -> str:
    marker = Path(__file__).resolve().parents[2] / "DEPLOYED_SHA"
    return marker.read_text().strip() if marker.exists() else "worktree"


def state_dir(root: Path) -> Path:
    path = root / ".state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def emit(payload) -> None:
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))


def storage_report(root: Path, plan: dict, config: dict, state_items: dict | None = None) -> dict:
    groups = [g for g in catalog.GROUPS if config["groups"].get(g)]
    selection = resolve_selection(config["pointworld_selection"], plan, root)
    selection = None if selection in (None, "pending") else selection
    usage = storage.fs_usage(root)
    enabled = storage.plan_fit(plan["items"], groups, usage, config["margin_fraction"], state_items, selection)
    complete = storage.plan_fit(plan["items"], catalog.GROUPS, usage, config["margin_fraction"], state_items, selection)
    return {"fs": usage, "margin_fraction": config["margin_fraction"], "enabled_groups": groups,
            "selection": config["pointworld_selection"], "enabled": enabled, "all_groups": complete}


def cmd_plan(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    existing = read_json(sd / "plan.json")
    plan = catalog.build_plan()
    plan["built"] = time.time()
    if existing:
        old = {i["id"]: (i["size"], i["checksum"]) for i in existing["items"]}
        new = {i["id"]: (i["size"], i["checksum"]) for i in plan["items"]}
        changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
        if changed or old.keys() != new.keys():
            emit({"error": "official listing differs from the stored plan", "changed": changed[:20],
                  "added": sorted(new.keys() - old.keys())[:20], "removed": sorted(old.keys() - new.keys())[:20]})
            return 2
        plan = existing
    else:
        write_json_atomic(sd / "plan.json", plan)
    if not (sd / "config.json").exists():
        write_json_atomic(sd / "config.json", DEFAULT_CONFIG, indent=1)
    report = storage_report(root, plan, load_config(sd))
    write_json_atomic(sd / "storage.json", report, indent=1)
    emit({"items": len(plan["items"]), "sources": plan["sources"], "storage": report})
    return 0


def cmd_storage(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    plan = read_json(sd / "plan.json")
    state = read_json(sd / "state.json", {}) or {}
    emit(storage_report(root, plan, load_config(sd), state.get("items")))
    return 0


def cmd_configure(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    plan = read_json(sd / "plan.json")
    config = load_config(sd)
    for group in args.enable or []:
        config["groups"][group] = True
    for group in args.disable or []:
        config["groups"][group] = False
    if args.pointworld_selection is not None:
        config["pointworld_selection"] = args.pointworld_selection
    if args.margin_fraction is not None:
        config["margin_fraction"] = args.margin_fraction
    for key in ("pool_initial", "pool_min", "pool_max"):
        if getattr(args, key) is not None:
            config["controller"][key] = getattr(args, key)
    state = read_json(sd / "state.json", {}) or {}
    report = storage_report(root, plan, config, state.get("items"))
    if not report["enabled"]["final_fits"] and config["pointworld_selection"] != "pending":
        emit({"error": "refused: the final footprint of the enabled groups would cross the margin",
              "deficit": report["enabled"]["deficit"], "report": report})
        return 2
    write_json_atomic(sd / "config.json", config, indent=1)
    emit({"config": config, "storage": report})
    return 0


def owner_summary(root: Path) -> dict:
    lock = OwnerLock(state_dir(root))
    live = lock.live_owner()
    return {"running": live is not None, "lock_held": lock.held_by_other(), "owner": live or lock.owner()}


def cmd_start(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    summary = owner_summary(root)
    if summary["lock_held"]:
        emit({"started": False, "reason": "an owner already holds the lock", **summary})
        return 0 if summary["running"] else 1
    if not (sd / "plan.json").exists():
        emit({"started": False, "reason": "no plan.json"})
        return 2
    (sd / "STOP").unlink(missing_ok=True)
    command = ["nice", "-n", str(args.nice), "ionice", "-c2", "-n7", sys.executable, "-I", str(SCRIPT), "run",
               "--root", str(root)]
    with open(sd / "runner.out", "ab") as out:
        subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        summary = owner_summary(root)
        if summary["running"]:
            break
        time.sleep(1)
    emit({"started": summary["running"], **summary})
    return 0 if summary["running"] else 1


def cmd_run(args) -> int:
    return Runner(Path(args.root), tools_sha=tools_sha()).run()


def cmd_status(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    status = read_json(sd / "status.json", {}) or {}
    summary = owner_summary(root)
    age = time.time() - status.get("time", 0) if status else None
    if args.full:
        emit({**summary, "status_age_s": age, "status": status})
        return 0
    brief = {k: status.get(k) for k in ("pool", "cap_bps", "last_action", "gates", "throughput_bps", "eta_s",
                                        "remaining_open_download_bytes", "errors_total", "space_wait", "selection",
                                        "stream", "staged_depth_bytes")}
    brief["fs"] = status.get("fs")
    brief["groups"] = {g: {"status": v["status"], "done_GB": round(v["download_done"] / 1e9, 1),
                           "total_GB": round(v["download_total"] / 1e9, 1), "gate": v["gate"]}
                       for g, v in (status.get("groups") or {}).items()}
    brief["inflight"] = len(status.get("inflight") or [])
    brief["health"] = {k: (status.get("health") or {}).get(k) for k in ("wan_rtt_ms", "gw_rtt_ms", "sshd_rtt_ms",
                                                                        "psi_io", "psi_cpu", "psi_mem", "disk_util",
                                                                        "mem_avail_frac")}
    emit({**summary, "status_age_s": age, **brief})
    return 0


def cmd_stop(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    if not owner_summary(root)["lock_held"]:
        emit({"stopped": True, "reason": "no owner running"})
        return 0
    (sd / "STOP").write_text(str(time.time()))
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        if not owner_summary(root)["lock_held"]:
            emit({"stopped": True})
            return 0
        time.sleep(2)
    emit({"stopped": False, "reason": f"owner still running after {args.wait} s (STOP file left in place)"})
    return 1


def cmd_adopt(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    plan = read_json(sd / "plan.json")
    item = next((i for i in plan["items"] if i["id"] == args.item), None)
    if item is None or "archive" in item:
        emit({"adopted": False, "reason": "unknown item or an archive (only plain files are adopted)"})
        return 2
    if owner_summary(root)["lock_held"]:
        emit({"adopted": False, "reason": "stop the owner first (single owner of the state)"})
        return 1
    source = Path(args.source)
    if source.stat().st_size != int(item["size"]) or not hash_file(source, item["checksum"], int(item["size"])):
        emit({"adopted": False, "reason": "size or official checksum mismatch"})
        return 2
    temp = temp_path(root, item)
    temp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temp)  # the source is never moved or modified
    if not hash_file(temp, item["checksum"], int(item["size"])):
        temp.unlink()
        emit({"adopted": False, "reason": "copy failed verification"})
        return 2
    final = root / item["dest"]
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, final)
    store = StateStore(sd / "state.json", [i["id"] for i in plan["items"]])
    store.update(item["id"], status="done", verified=item["checksum"], adopted_from=str(source), error=None)
    store.save(force=True)
    emit({"adopted": True, "item": item["id"], "dest": str(final)})
    return 0


def cmd_retry(args) -> int:
    root = Path(args.root)
    sd = state_dir(root)
    if owner_summary(root)["lock_held"]:
        emit({"reset": 0, "reason": "stop the owner first"})
        return 1
    plan = read_json(sd / "plan.json")
    store = StateStore(sd / "state.json", [i["id"] for i in plan["items"]])
    failed = store.by_status("failed")
    for item_id in failed:
        store.update(item_id, status="pending", attempts=0, extract_attempts=0, not_before=0)
    store.save(force=True)
    emit({"reset": len(failed)})
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="droid_datasets.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("plan", "storage", "start", "run", "status", "stop", "adopt", "retry", "configure"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=DEFAULT_ROOT)
        if name == "start":
            p.add_argument("--nice", type=int, default=10)
        if name == "status":
            p.add_argument("--full", action="store_true")
        if name == "stop":
            p.add_argument("--wait", type=float, default=300.0)
        if name == "adopt":
            p.add_argument("--item", required=True)
            p.add_argument("--source", required=True)
        if name == "configure":
            p.add_argument("--enable", action="append", choices=catalog.GROUPS)
            p.add_argument("--disable", action="append", choices=catalog.GROUPS)
            p.add_argument("--pointworld-selection", default=None, help="pending | all | <episode list file>")
            p.add_argument("--margin-fraction", type=float, default=None)
            p.add_argument("--pool-initial", dest="pool_initial", type=int, default=None)
            p.add_argument("--pool-min", dest="pool_min", type=int, default=None)
            p.add_argument("--pool-max", dest="pool_max", type=int, default=None)
    return ap


COMMANDS = {"plan": cmd_plan, "storage": cmd_storage, "configure": cmd_configure, "start": cmd_start, "run": cmd_run,
            "status": cmd_status, "stop": cmd_stop, "adopt": cmd_adopt, "retry": cmd_retry}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return COMMANDS[args.command](args)
