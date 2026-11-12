"""Remote MIG migration (user directive 2026-11-11): inventory discovery/validation, the fail-closed allowlist, MIG-aware
snapshot / launch-guard evidence, the outage guard's hold ownership and the release re-pin event. Synthetic
``nvidia-smi`` evidence only (the layouts of driver 580 as observed on the local MIG GPUs); nothing touches a GPU."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ops.resources import mig_inventory as inv

REPO = Path(__file__).resolve().parents[1]
PARENTS = inv.REMOTE_PHYSICAL_GPUS
OTHER = ("GPU-1aa64fba-f45e-ee28-f83e-a272cb624cb4", "GPU-83c912e1-65a8-9361-da0e-f8996e5971bb")  # other users' GPUs


def mig_uuid(parent_index: int, device: int) -> str:
    return f"MIG-{parent_index:08x}-{device:04x}-5000-8000-{parent_index * 16 + device:012x}"


def render_list(parents, profile="1g.24gb", per_parent=4, overrides=None):
    lines = []
    for index, (uuid, mig) in enumerate(parents):
        lines.append(f"GPU {index}: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: {uuid})")
        for device in range(per_parent if mig else 0):
            name = (overrides or {}).get((index, device), profile)
            lines.append(f"  MIG {name:<13} Device  {device}: (UUID: {mig_uuid(index, device)})")
    return "\n".join(lines) + "\n"


def render_query(parents, per_parent=4, processes=None, memory=24192, mode=("Enabled", "Enabled"), gi_base=3,
                 process_value=None):
    """``nvidia-smi -q -i`` text: per GPU the MIG mode, the MIG devices (index, GI, CI, FB memory), the processes."""
    out = ["", "==============NVSMI LOG==============", "", "Timestamp                       : x", ""]
    for index, (uuid, mig) in enumerate(parents):
        out.append(f"GPU 00000000:{0x41 + index:02X}:00.0")
        out.append("    Product Name                                       : NVIDIA RTX PRO 6000")
        out.append("    MIG Mode")
        out.append(f"        Current                                        : {mode[0] if mig else 'N/A'}")
        out.append(f"        Pending                                        : {mode[1] if mig else 'N/A'}")
        for device in range(per_parent if mig else 0):
            out += ["    MIG Device",
                    f"        Index                                          : {device}",
                    f"        GPU Instance ID                                : {gi_base + device}",
                    "        Compute Instance ID                            : 0",
                    "        Device Attributes",
                    "            Shared",
                    "                Multiprocessor count                   : 46",
                    "        FB Memory Usage",
                    f"            Total                                      : {memory} MiB",
                    "            Reserved                                   : 0 MiB",
                    "            Used                                       : 67 MiB",
                    "            Free                                       : 24125 MiB",
                    "        BAR1 Memory Usage",
                    "            Total                                      : 8327 MiB",
                    "            Used                                       : 0 MiB"]
        out.append("    Driver Model")
        out.append("        Current                                        : N/A")
        out.append(f"    GPU UUID                                           : {uuid}")
        procs = (processes or {}).get(index, [])
        if process_value is not None:
            out.append(f"    Processes                                          : {process_value}")
        elif not procs:
            out.append("    Processes                                          : None")
        else:
            out.append("    Processes")
            for gi, pid, mib in procs:
                out += [f"        GPU instance ID                                : {gi}",
                        "        Compute instance ID                            : 0",
                        f"        Process ID                                     : {pid}",
                        "            Type                                       : C",
                        "            Name                                       : python",
                        f"            Used GPU Memory                            : {mib} MiB"]
    return "\n".join(out) + "\n"


def remote_host(**kwargs):
    """The remote's 8 GPUs: ours (0-3, MIG) and other users' (4-5, also MIG) - the latter must never be listed."""
    parents = [(uuid, True) for uuid in PARENTS] + [(uuid, True) for uuid in OTHER]
    return render_list(parents), render_query(parents, **kwargs)


# ------------------------------------------------------------------------------------------------ discovery / inventory
def test_complete_inventory_lists_exactly_our_16_instances():
    listing, query = remote_host()
    report = inv.build_inventory(listing, query)
    assert report["complete"] and not report["problems"] and not report["excluded"]
    assert len(report["instances"]) == 16
    assert {item["parent"] for item in report["instances"]} == set(PARENTS)
    assert not any(item["parent"] in OTHER for item in report["instances"])  # other users' GPUs never listed
    first = report["instances"][0]
    assert first == {**first, "uuid": mig_uuid(0, 0), "gpu_instance_id": 3, "device": 0, "memory_total_mib": 24192}


def test_foreign_process_excludes_only_that_instance_and_owned_pids_do_not():
    listing, query = remote_host(processes={1: [(4, 999, 500)], 2: [(3, 4242, 900)]})
    report = inv.build_inventory(listing, query, owned_pids=frozenset({4242}))
    assert not report["complete"]
    assert [e for e in report["excluded"] if "foreign" in e["reason"]] == [
        {"uuid": mig_uuid(1, 1), "parent": PARENTS[1], "reason": "foreign processes [999] (used by others)"}]
    assert len(report["instances"]) == 15 and mig_uuid(2, 0) in {i["uuid"] for i in report["instances"]}
    assert any("3 usable" in problem for problem in report["problems"])


def test_wrong_profile_small_memory_and_pending_mode_are_excluded():
    parents = [(uuid, True) for uuid in PARENTS]
    listing = render_list(parents, overrides={(0, 3): "2g.48gb"})
    report = inv.build_inventory(listing, render_query(parents, memory=12000))
    assert not report["instances"] and not report["complete"]
    assert any("profile 2g.48gb" in e["reason"] for e in report["excluded"])
    assert any("framebuffer 12000 MiB" in e["reason"] for e in report["excluded"])
    pending = inv.build_inventory(render_list(parents), render_query(parents, mode=("Disabled", "Enabled")))
    assert not pending["instances"] and all("MIG mode current=Disabled" in p for p in pending["problems"])


def test_missing_parent_and_unreadable_processes_fail_closed():
    parents = [(uuid, True) for uuid in PARENTS[:3]]
    report = inv.build_inventory(render_list(parents), render_query(parents))
    assert not report["complete"] and any(PARENTS[3] in p for p in report["problems"])
    listing, query = remote_host(process_value="Insufficient Permissions")
    with pytest.raises(inv.InventoryError, match="process listing unavailable"):
        inv.build_inventory(listing, query)


def test_physical_mode_gpus_yield_no_instances():
    parents = [(uuid, False) for uuid in PARENTS]
    report = inv.build_inventory(render_list(parents), render_query(parents))
    assert not report["instances"] and len(report["problems"]) == 4


# ---------------------------------------------------------------------------------------------------- allowlist file
def allowlist(**kwargs):
    return inv.allowlist_document(inv.build_inventory(*remote_host(**kwargs)))


def test_allowlist_round_trip_and_tampering(tmp_path):
    doc = allowlist()
    path = tmp_path / "remote-mig.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert [item["uuid"] for item in inv.load_allowlist(path)] == [item["uuid"] for item in doc["instances"]]
    assert inv.load_allowlist(tmp_path / "absent.yaml") is None
    bad = [
        {**doc, "parents": doc["parents"] + [OTHER[0]]},
        {**doc, "instances": doc["instances"] + [doc["instances"][0]]},
        {**doc, "instances": [{**doc["instances"][0], "parent": OTHER[0]}]},
        {**doc, "instances": [{**doc["instances"][0], "memory_total_mib": 1000}]},
        {**doc, "instances": [{**doc["instances"][0], "uuid": "GPU-4391fcee-537f-d408-d138-10d8a3866eea"}]},
        {**doc, "instances_per_parent": 3},
        {**doc, "instances": []},
        {**doc, "schema": "other"},
    ]
    for item in bad:
        path.write_text(yaml.safe_dump(item))
        with pytest.raises(inv.InventoryError):
            inv.load_allowlist(path)
    path.write_text("{not yaml: [")
    with pytest.raises(inv.InventoryError):
        inv.load_allowlist(path)


def test_partial_allowlist_is_valid_but_incomplete_reports_refuse_render(tmp_path):
    report = inv.build_inventory(*remote_host(processes={3: [(6, 1, 1)]}))
    doc = inv.allowlist_document(report)
    assert len(doc["instances"]) == 15 and doc["excluded"]
    (tmp_path / "configs/hosts").mkdir(parents=True)
    out = tmp_path / "report.json"
    out.write_text(json.dumps(report))
    assert inv.main(["render", "--report", str(out), "--repo", str(tmp_path)]) == 3
    assert not (tmp_path / inv.INVENTORY_RELATIVE).exists()


def test_capacity_limits_and_oversized_rows():
    doc = allowlist()
    instances = inv.validate_allowlist(doc)
    assert inv.device_capacity_gib(instances[0]["uuid"], instances) == pytest.approx(21.5)
    assert inv.device_capacity_gib(PARENTS[0], ()) == 90.0
    limits = inv.queue_limits(instances)
    assert set(limits) == {item["uuid"] for item in instances} and all(v["max_jobs"] == 4 for v in limits.values())
    jobs = [{"id": "fql", "host": "remote", "mem_gb": 6}, {"id": "rwprep-ntb", "host": "remote", "mem_gb": 34},
            {"id": "prep", "host": "remote", "mem_gb": 24}, {"id": "local", "host": "local", "mem_gb": 40}]
    assert inv.oversized_rows(jobs, instances) == [("rwprep-ntb", 34.0), ("prep", 24.0)]


def test_discover_and_render_from_saved_evidence(tmp_path):
    listing, query = remote_host()
    (tmp_path / "L.txt").write_text(listing)
    (tmp_path / "q.txt").write_text(query)
    out = tmp_path / "report.json"
    assert inv.main(["discover", "--out", str(out), "--list-file", str(tmp_path / "L.txt"),
                     "--query-file", str(tmp_path / "q.txt")]) == 0
    repo = tmp_path / "repo"
    (repo / "configs/hosts").mkdir(parents=True)
    assert inv.main(["render", "--report", str(out), "--repo", str(repo)]) == 0
    host = yaml.safe_load((repo / "configs/hosts/remote.yaml").read_text())
    assert host["name"] == "remote" and len(host["gpu_uuids"]) == 16
    assert len(inv.load_allowlist(repo / inv.INVENTORY_RELATIVE)) == 16


# --------------------------------------------------------------------------------------------- gpu_guard (isolated copy)
def guard_copy(tmp_path, allowlist_doc):
    """A minimal repository copy whose gpu_guard reads the given allowlist (None: no file)."""
    root = tmp_path / "repo"
    for relative in ("ops/__init__.py", "ops/resources/__init__.py", "ops/resources/gpu_guard.py",
                     "ops/resources/mig_inventory.py", "configs/hosts/local.yaml", "configs/hosts/remote.yaml"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / relative, root / relative)
    (root / "configs/hosts/remote.yaml").write_text(yaml.safe_dump(  # physical, whatever the checkout under test has
        {"name": "remote", "gpu_uuids": list(PARENTS), "runtime_root": "/home/compu/kaist/sunho"}))
    (root / inv.INVENTORY_RELATIVE).unlink(missing_ok=True)
    if allowlist_doc is not None:
        (root / inv.INVENTORY_RELATIVE).write_text(allowlist_doc if isinstance(allowlist_doc, str)
                                                   else yaml.safe_dump(allowlist_doc))
        if isinstance(allowlist_doc, dict):
            (root / "configs/hosts/remote.yaml").write_text(yaml.safe_dump(
                {"name": "remote", "gpu_uuids": [i["uuid"] for i in allowlist_doc["instances"]],
                 "runtime_root": "/home/compu/kaist/sunho"}))
    return root


PROBE = """
import json, os, sys
sys.path.insert(0, sys.argv[1])
from ops.resources import gpu_guard as g
out = {"remote": list(g.APPROVED_HOST_GPUS["remote"]), "active": g.REMOTE_MIG_ACTIVE, "error": g.REMOTE_MIG_ERROR,
       "parents": list(g.REMOTE_MIG_PARENTS), "mig": {k: list(v) for k, v in g.REMOTE_MIG_DEVICES.items()}}
try:
    out["host"] = g.load_host_config(g.Path(sys.argv[1]) / "configs/hosts/remote.yaml")["name"]
except g.GPUIsolationError as exc:
    out["host"] = "refused"
checks = {}
for value in sys.argv[2:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    try:
        g.visible_device_uuids()
        checks[value] = "ok"
    except g.GPUIsolationError:
        checks[value] = "refused"
out["checks"] = checks
out["cuda_uuid"] = {k: g.cuda_uuid(k) for k in list(g.REMOTE_MIG_DEVICES)[:1]}
print(json.dumps(out))
"""


def run_guard(root, *values, host="remote"):
    env = {**os.environ, "S4D_HOST_CONFIG": str(root / f"configs/hosts/{host}.yaml")}
    result = subprocess.run([sys.executable, "-I", "-c", PROBE, str(root), *values], capture_output=True, text=True,
                            env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_guard_without_allowlist_keeps_the_physical_gpus(tmp_path):
    out = run_guard(guard_copy(tmp_path, None), PARENTS[0], mig_uuid(0, 0))
    assert out["remote"] == list(PARENTS) and not out["active"] and out["parents"] == []
    assert out["host"] == "remote" and out["checks"] == {PARENTS[0]: "ok", mig_uuid(0, 0): "refused"}


def test_guard_with_allowlist_authorizes_only_the_instances(tmp_path):
    doc = allowlist()
    out = run_guard(guard_copy(tmp_path, doc), PARENTS[0], mig_uuid(0, 0), f"{mig_uuid(0, 0)},{mig_uuid(0, 1)}",
                    OTHER[0])
    assert out["active"] and out["remote"] == [i["uuid"] for i in doc["instances"]] and out["parents"] == list(PARENTS)
    assert out["host"] == "remote"
    assert out["checks"] == {PARENTS[0]: "refused", mig_uuid(0, 0): "ok",
                             f"{mig_uuid(0, 0)},{mig_uuid(0, 1)}": "refused", OTHER[0]: "refused"}
    assert out["mig"][mig_uuid(0, 0)] == [PARENTS[0], 3, 0] and out["cuda_uuid"] == {mig_uuid(0, 0): PARENTS[0]}


def test_guard_with_invalid_allowlist_authorizes_nothing(tmp_path):
    doc = allowlist()
    broken = {**doc, "instances": [{**doc["instances"][0], "parent": OTHER[0]}]}
    root = guard_copy(tmp_path, yaml.safe_dump(broken))
    out = run_guard(root, host="local")  # the scheduler side: no remote device, local unaffected
    assert out["remote"] == [] and out["active"] and out["error"] and out["host"] == "refused"
    inside = subprocess.run([sys.executable, "-I", "-c", PROBE, str(root), mig_uuid(0, 0)], capture_output=True,
                            text=True, timeout=120, env={**os.environ, "S4D_HOST_CONFIG": str(root / "configs/hosts/remote.yaml")})
    assert inside.returncode != 0 and "approved GPU UUID set" in inside.stderr  # a remote container refuses to start


# ------------------------------------------------------------------------------------- snapshot / launch-guard evidence
def devices_of(doc):
    return {i["uuid"]: (i["parent"], i["gpu_instance_id"], i["device"]) for i in doc["instances"]}


def test_snapshot_attributes_processes_per_instance_and_fails_closed():
    from ops.jobs import remote_backend

    doc = allowlist()
    devices = devices_of(doc)
    listing, query = remote_host(processes={0: [(3, 11, 100), (5, 12, 200)], 4: [(3, 77, 300)]})
    usage, memory, ready = remote_backend.mig_usage(listing.splitlines(), query.splitlines(), devices)
    assert ready and usage[mig_uuid(0, 0)] == [(11, 100.0)] and usage[mig_uuid(0, 2)] == [(12, 200.0)]
    assert sum(len(v) for v in usage.values()) == 2  # other users' GPU 4 never attributed to our instances
    assert memory[mig_uuid(1, 3)] == (24192, 67)
    drifted = render_query([(uuid, True) for uuid in PARENTS], gi_base=7)
    assert not remote_backend.mig_usage(listing.splitlines(), drifted.splitlines(), devices)[2]
    _, unreadable = remote_host(process_value="Insufficient Permissions")
    assert not remote_backend.mig_usage(listing.splitlines(), unreadable.splitlines(), devices)[2]
    missing = render_list([(uuid, True) for uuid in PARENTS[:3]])
    assert not remote_backend.mig_usage(missing.splitlines(), query.splitlines(), devices)[2]
    assert remote_backend.mig_snapshot_command(()) == ""
    assert "nvidia-smi -q -i" in remote_backend.mig_snapshot_command(PARENTS)


def run_probe(text, gi, device, uuid=None, parent=PARENTS[0], listing=None):
    from ops.containers import launch

    listing = render_list([(p, True) for p in PARENTS]) if listing is None else listing
    uuid = mig_uuid(PARENTS.index(parent), device) if uuid is None else uuid
    return subprocess.run([sys.executable, "-I", "-c", launch.MIG_PROBE, str(gi), str(device), uuid, parent],
                          input=listing + "\n__S4D_QUERY__\n" + text, capture_output=True, text=True, timeout=60)


def test_launch_guard_probe_reports_instance_memory_and_processes():
    query = render_query([(PARENTS[0], True)], processes={0: [(4, 21, 512), (3, 22, 64)]})
    result = run_probe(query, 4, 1)
    assert result.returncode == 0 and result.stdout.splitlines() == ["MEM,24192,67", "APP,21,512"]
    assert run_probe(query, 9, 1).returncode == 4  # the device index no longer carries the allowlisted GPU instance
    # the UUID must be listed exactly once, under the allowlisted parent and device index (fresh -L evidence)
    assert run_probe(query, 4, 1, uuid=mig_uuid(1, 1)).returncode == 6  # an instance of another parent
    assert run_probe(query, 4, 1, uuid=mig_uuid(0, 2)).returncode == 6  # another device index of the same parent
    assert run_probe(query, 4, 1, uuid="MIG-ffffffff-0000-5000-8000-000000000000").returncode == 6  # not listed
    truncated = query.rsplit("            Used GPU Memory                            : 64 MiB", 1)[0]
    assert run_probe(truncated, 4, 1).returncode == 3  # an unfinished foreign process record is never "no process"
    denied = render_query([(PARENTS[0], True)], process_value="Insufficient Permissions")
    assert run_probe(denied, 3, 0).returncode == 2


def test_launch_guard_lines_keep_physical_gpus_and_probe_instances(monkeypatch):
    from ops.containers import launch

    physical = "\n".join(launch.gpu_probe_lines(PARENTS[0]))
    assert "nvidia-smi --query-compute-apps" in physical and "python3" not in physical  # established lines unchanged
    monkeypatch.setattr(launch, "REMOTE_MIG_DEVICES", devices_of(allowlist()))
    mig = "\n".join(launch.gpu_probe_lines(mig_uuid(2, 1)))
    assert f"parent={PARENTS[2]}" in mig and "python3 -I -c" in mig and ' 4 1 "$gpu" "$parent")' in mig
    assert "nvidia-smi --query-compute-apps" not in mig
    assert "mig_probe" in launch.gpu_free_command(mig_uuid(2, 1))


# ---------------------------------------------------------------------------------------------- outage guard ownership
def test_outage_guard_never_lifts_a_hold_it_did_not_write(tmp_path, monkeypatch):
    from ops.remote import outage_guard as guard

    hold, ledger = tmp_path / "remote-hold-admissions", tmp_path / "ledger"
    ledger.mkdir()
    monkeypatch.setattr(guard, "HOLD", hold)
    monkeypatch.setattr(guard, "LEDGER", ledger)
    monkeypatch.setattr(guard, "GUARD", ledger / "guard.json")
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    (tmp_path / "runs/setup").mkdir(parents=True)

    class Client:
        def check_tunnel(self, wait=False):
            return True

    monkeypatch.setattr(guard, "RemoteClient", Client)
    monkeypatch.setattr(guard, "recovery_check", lambda client: ({}, {"job": "ok"}))
    hold.write_text("REMOTE MIG MIGRATION HOLD (user 2026-11-11 ~19:00): ...\n")
    (ledger / "guard.json").write_text(json.dumps({"hold_by_guard": True}))  # a stale claim
    guard.main()
    assert hold.exists() and json.loads((ledger / "guard.json").read_text())["hold_by_guard"] is False
    hold.write_text(f"{guard.GUARD_HOLD_PREFIX}2026-11-11T20:00:00+0900: host unreachable\n")
    (ledger / "guard.json").write_text(json.dumps({"hold_by_guard": True}))
    guard.main()
    assert not hold.exists() and list((tmp_path / "runs/setup").glob("remote-hold-admissions.cleared-*"))


def test_outage_guard_unreachable_keeps_an_existing_foreign_hold(tmp_path, monkeypatch):
    from ops.remote import outage_guard as guard
    from ops.remote.access import RemoteUnreachable

    hold, ledger = tmp_path / "remote-hold-admissions", tmp_path / "ledger"
    ledger.mkdir()
    for name, value in (("HOLD", hold), ("LEDGER", ledger), ("GUARD", ledger / "guard.json"), ("ROOT", tmp_path)):
        monkeypatch.setattr(guard, name, value)

    class Down:
        def check_tunnel(self, wait=False):
            raise RemoteUnreachable("down")

    monkeypatch.setattr(guard, "RemoteClient", Down)
    hold.write_text("REMOTE MIG MIGRATION HOLD\n")
    (ledger / "guard.json").write_text(json.dumps({"unreachable_since": 0}))
    guard.main()
    state = json.loads((ledger / "guard.json").read_text())
    assert hold.read_text() == "REMOTE MIG MIGRATION HOLD\n" and not state.get("hold_by_guard")


# --------------------------------------------------------------------------------------------------- release re-pin
C1, C2, IMG = "a" * 40, "b" * 40, "sha256:" + "c" * 64


def registry_with(tmp_path, *events):
    path = tmp_path / "registry.jsonl"
    base = [{"event": "launched", "id": "j", "host": "remote", "gpu": PARENTS[0], "attempt": 1, "container": "s4d-j-a1",
             "container_id": "d" * 64, "commit": C1, "image_digest": IMG, "gpu_backend": "cdi", "time": 1.0},
            {"event": "exited", "id": "j", "host": "remote", "attempt": 1, "code": 137, "time": 2.0}]
    path.write_text("".join(json.dumps(e) + "\n" for e in base + list(events)))
    return path


def repin(**overrides):
    return {"event": "release_repinned", "id": "j", "host": "remote", "attempt": 1, "from_commit": C1,
            "from_image_digest": IMG, "commit": C2, "image_digest": IMG, "sources_identical": True,
            "authorization": "user 2026-11-12: MIG done, resume", "time": 3.0, **overrides}


def test_repin_event_moves_a_stopped_run_to_the_new_release(tmp_path):
    from ops.jobs import scheduler as jobs

    state = jobs.read_registry(registry_with(tmp_path, repin()))["j"]
    assert state["commit"] == C2 and state["status"] == "crashed" and state["failures"] == 1
    assert state["repinned_from"] == {"commit": C1, "image_digest": IMG, "attempt": 1}


@pytest.mark.parametrize("bad", [{"from_commit": C2}, {"authorization": " "}, {"sources_identical": False},
                                 {"commit": C1}, {"host": "local"}])
def test_repin_event_refuses_inconsistent_records(tmp_path, bad):
    from ops.jobs import scheduler as jobs

    with pytest.raises(ValueError, match="re-pin"):
        jobs.read_registry(registry_with(tmp_path, repin(**bad)))


def test_repin_event_refuses_an_active_run(tmp_path):
    from ops.jobs import scheduler as jobs

    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps({"event": "launched", "id": "j", "host": "remote", "gpu": PARENTS[0], "attempt": 1,
                                "container": "s4d-j-a1", "commit": C1, "image_digest": IMG, "gpu_backend": "cdi",
                                "time": 1.0}) + "\n" + json.dumps(repin()) + "\n")
    with pytest.raises(ValueError, match="re-pin"):
        jobs.read_registry(path)


def test_repin_scope_allows_only_the_gpu_allowlist_files():
    from ops.jobs import release_repin

    before = {"scripts/train.py": b"from ops.resources import gpu_guard\nimport s4d.model\n",
              "ops/resources/gpu_guard.py": b"OLD = 1\n", "s4d/model.py": b"x = 1\n", "s4d/__init__.py": b"",
              "ops/__init__.py": b"", "ops/resources/__init__.py": b"", "configs/realworld/a.yaml": b"a: 1\n",
              "uv.lock": b"lock\n", "pyproject.toml": b"[x]\n"}
    after = {**before, "ops/resources/gpu_guard.py": b"NEW = 2\nfrom ops.resources import mig_inventory\n",
             "ops/resources/mig_inventory.py": b"Y = 1\n", "configs/hosts/remote-mig.yaml": b"m: 1\n"}

    def reader(tree):
        return (lambda relative: tree.get(relative)), sorted(tree)

    same = release_repin.scope_comparison(["scripts/train.py"], *reader(before), *reader(after))
    assert same["identical"] and set(same["operational_differences"]) == {"ops/resources/gpu_guard.py",
                                                                         "ops/resources/mig_inventory.py",
                                                                         "configs/hosts/remote-mig.yaml"}
    for relative, value in (("s4d/model.py", b"x = 2\n"), ("configs/realworld/a.yaml", b"a: 2\n"), ("uv.lock", b"2")):
        changed = release_repin.scope_comparison(["scripts/train.py"], *reader(before),
                                                 *reader({**after, relative: value}))
        assert not changed["identical"] and changed["differences"] == [relative]
        with pytest.raises(release_repin.RepinError, match="scope differs"):
            release_repin.repin_event("j", {"commit": C1, "image_digest": IMG, "attempt": 1}, C2, IMG, changed,
                                      ["scripts/train.py"], "user ok")


def test_repin_state_checks():
    from ops.jobs import release_repin

    stopped = {"host": "remote", "commit": C1, "image_digest": IMG, "status": "crashed", "attempt": 1}
    release_repin.check_state("j", stopped, C2, IMG)
    for bad in ({**stopped, "status": "running"}, {**stopped, "unknown": True}, {**stopped, "host": "local"},
                {**stopped, "commit": None}):
        with pytest.raises(release_repin.RepinError):
            release_repin.check_state("j", bad, C2, IMG)
    with pytest.raises(release_repin.RepinError, match="already"):
        release_repin.check_state("j", stopped, C1, IMG)


# ------------------------------------------------------------------------------------------ no hard-coded remote GPUs
def test_remote_controllers_take_their_devices_from_the_guard():
    for relative in ("ops/campaign/remote_campaign_driver.py", "ops/remote/outage_guard.py", "ops/remote/ledger.py"):
        text = (REPO / relative).read_text()
        assert "4391fcee" not in text, relative
    driver = (REPO / "ops/campaign/remote_campaign_driver.py").read_text()
    assert "GPUS = tuple(APPROVED_HOST_GPUS['remote'])" in driver and "<= 90" not in driver and "GPU_CAP[g]" in driver
    source = (REPO / "ops/campaign/release_chain.py").read_text()
    assert "self.gpus = list(APPROVED_HOST_GPUS['remote'])" in source
    assert "range(len(self.gpus))" in source


def test_snapshot_mig_mode_conflicts():
    from ops.jobs import remote_backend

    enabled = [f"{PARENTS[0]}, Enabled, Enabled"] + [f"{u}, [N/A], [N/A]" for u in PARENTS[1:]]
    pending = [f"{PARENTS[1]}, Disabled, Enabled"]
    assert remote_backend.mig_mode_conflict(enabled, physical=PARENTS, parents=())
    assert remote_backend.mig_mode_conflict(pending, physical=PARENTS, parents=())
    assert not remote_backend.mig_mode_conflict([f"{u}, [N/A], [N/A]" for u in PARENTS], physical=PARENTS, parents=())
    assert not remote_backend.mig_mode_conflict([], physical=PARENTS, parents=())  # no evidence: no effect
    all_on = [f"{u}, Enabled, Enabled" for u in PARENTS]
    assert not remote_backend.mig_mode_conflict(all_on, physical=(), parents=PARENTS)
    assert remote_backend.mig_mode_conflict(all_on[:3] + [f"{PARENTS[3]}, Enabled, Disabled"], physical=(),
                                            parents=PARENTS)
    command = remote_backend.snapshot_command(remote_backend.ROOT)
    assert "mig.mode.current,mig.mode.pending" in command
    assert command.count("python3 -I -c") == 1 and "resume_valid" in command
    assert subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True).returncode == 0


def test_isolation_check_attributes_a_remote_instance_not_its_parent(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("isolation_mig", REPO / "scripts/check_gpu_isolation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    devices = devices_of(allowlist())
    monkeypatch.setattr(module, "KNOWN_MIG", {**module.MIG_DEVICES, **devices})
    monkeypatch.setattr(module, "MIG_PARENTS", {owner for owner, _, _ in module.KNOWN_MIG.values()})
    table = ("| Processes:                                                                              |\n"
             "|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n"
             "|    1    4    0            4321      C   python                                  500MiB |\n")
    outputs = {"--query-compute-apps=pid,gpu_uuid": f"4321, {PARENTS[1]}\n"}

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(module.subprocess, "run",
                        lambda cmd, **kw: Done(outputs.get(cmd[1], table) if len(cmd) > 1 else table))
    found, _, _ = module.pid_gpus(4321, {"1": PARENTS[1]})
    assert found == {mig_uuid(1, 1)}  # the instance on GI 4 of parent 1; the parent itself is not "seen"


def test_launch_guard_mig_fragment_runs_against_fake_nvidia_smi(tmp_path, monkeypatch):
    from ops.containers import launch

    devices = devices_of(allowlist())
    monkeypatch.setattr(launch, "REMOTE_MIG_DEVICES", devices)
    parents = [(uuid, True) for uuid in PARENTS]
    (tmp_path / "L.txt").write_text(render_list(parents))
    (tmp_path / "q.txt").write_text(render_query(parents[:1], processes={0: [(4, 31, 700), (5, 32, 100)]}))
    fake = tmp_path / "bin/nvidia-smi"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase "$1" in -L) cat {tmp_path}/L.txt;; -q) cat {tmp_path}/q.txt;; *) exit 9;; esac\n')
    fake.chmod(0o755)
    (tmp_path / "bin/python3").symlink_to(sys.executable)
    gpu = mig_uuid(0, 1)
    probe = launch.gpu_probe_lines(gpu)
    script = "\n".join(["set -eu", f"gpu={gpu}", *probe, launch.gpu_free_command(gpu),
                        'printf "APPS=%s\\nFREE=%s\\n" "$apps" "$gpu_free"'])
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"APPS={gpu},31,700", f"FREE={(24192 - 67) * 1048576}"]
    missing = "\n".join(["set -eu", f"gpu={mig_uuid(0, 9)}", *launch.gpu_probe_lines(gpu)[1:]])
    assert subprocess.run(["bash", "-c", missing], capture_output=True, text=True, env=env).returncode == 1


def test_allowlist_profile_and_minimum_are_strict_and_guard_never_crashes(tmp_path):
    doc = allowlist()
    for bad in ({**doc, "min_memory_mib": None}, {**doc, "min_memory_mib": 1},
                {**doc, "profile": "2g.48gb", "instances": [{**i, "profile": "2g.48gb"} for i in doc["instances"]]},
                {**doc, "parents": [None]}):
        with pytest.raises(inv.InventoryError):
            inv.validate_allowlist(bad)
    from ops.resources import gpu_guard

    path = tmp_path / "remote-mig.yaml"
    path.write_text(yaml.safe_dump({**doc, "instances": [{**doc["instances"][0], "device": None}]}))
    instances, error = gpu_guard._remote_mig_allowlist(path)
    assert instances == () and error


def test_repin_scope_includes_third_party_pins():
    from ops.jobs import release_repin

    tree = {"scripts/train.py": b"x = 1\n"}
    reader = (lambda relative: tree.get(relative)), sorted(tree)
    same = release_repin.scope_comparison(["scripts/train.py"], *reader, *reader, {"third_party/dino@commit": "a"},
                                          {"third_party/dino@commit": "a"})
    moved = release_repin.scope_comparison(["scripts/train.py"], *reader, *reader, {"third_party/dino@commit": "a"},
                                           {"third_party/dino@commit": "b"})
    assert same["identical"] and not moved["identical"] and moved["differences"] == ["third_party:third_party/dino@commit"]


def test_missing_process_listing_or_unreadable_framebuffer_fail_closed():
    from ops.jobs import remote_backend

    listing, query = remote_host()
    no_listing = "\n".join(line for line in query.splitlines() if not line.startswith("    Processes"))
    report = inv.build_inventory(listing, no_listing)
    assert not report["instances"] and not report["complete"]
    assert sum("ownership unknown" in p for p in report["problems"]) == 4
    devices = devices_of(allowlist())
    assert not remote_backend.mig_usage(listing.splitlines(), no_listing.splitlines(), devices)[2]
    na = query.replace("            Used                                       : 67 MiB",
                       "            Used                                       : N/A")
    assert not remote_backend.mig_usage(listing.splitlines(), na.splitlines(), devices)[2]
    single = render_query([(PARENTS[0], True)])
    assert run_probe("\n".join(l for l in single.splitlines() if not l.startswith("    Processes")), 3, 0).returncode == 5
    assert run_probe(single.replace(": 67 MiB", ": N/A"), 3, 0).returncode != 0


def test_unfinished_process_record_fails_inventory():
    listing, query = remote_host(processes={0: [(3, 11, 100)]})
    truncated = query.replace("            Used GPU Memory                            : 100 MiB\n", "", 1)
    with pytest.raises(inv.InventoryError, match="unfinished"):
        inv.build_inventory(listing, truncated)


def test_scheduler_central_remote_hold_and_quarantine(tmp_path, monkeypatch):
    from ops.jobs import scheduler as jobs
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS

    source = (REPO / "ops/jobs/scheduler.py").read_text()
    assert '(registry.parent.parent / "runs/setup/remote-hold-admissions").exists()' in source
    queue = {"hosts": {"local": {"enabled": False}, "remote": {"enabled": True, "limits": {
        APPROVED_HOST_GPUS["remote"][0]: {"max_jobs": 1, "max_mem_gb": 10}}}}, "jobs": []}
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue))
    assert jobs.load_queue(path)["hosts"]["remote"]["enabled"] is True
    monkeypatch.setattr(jobs, "REMOTE_MIG_ERROR", "bad allowlist")
    quarantined = jobs.load_queue(path)["hosts"]["remote"]
    assert quarantined["enabled"] is False and quarantined["limits"] == {} and "bad allowlist" in quarantined["quarantine"]


def test_outage_guard_lifts_only_its_exact_hold(tmp_path, monkeypatch):
    from ops.remote import outage_guard as guard

    hold = tmp_path / "remote-hold-admissions"
    monkeypatch.setattr(guard, "HOLD", hold)
    hold.write_text(f"{guard.GUARD_HOLD_PREFIX}x token 1\n")
    import hashlib

    state = {"hold_sha256": hashlib.sha256(hold.read_bytes()).hexdigest()}
    assert guard.guard_owns_hold(state)
    hold.write_text(f"{guard.GUARD_HOLD_PREFIX}x token 2\n")  # same prefix, different hold: not ours
    assert not guard.guard_owns_hold(state)


def test_runtime_provenance_accepts_exactly_the_recorded_repin():
    from ops.jobs import run_identity as ri

    old = {"host": "remote", "campaign_eligible": True, "git_commit": C1, "image_digest": IMG, "data_id": "d" * 64,
           "config_sha256": "e" * 64}
    new = {**old, "git_commit": C2, "attempt": 2}
    with pytest.raises(ri.RunIdentityError, match="git_commit"):
        ri._check_previous(old, new)  # no recorded transition: refused as before
    moved = {**new, "repinned_from": {"git_commit": C1, "image_digest": IMG, "to_git_commit": C2, "to_image_digest": IMG}}
    ri._check_previous(old, moved)
    history = ri._release_history(old, moved)
    assert len(history) == 1 and history[0]["from_commit"] == C1 and history[0]["to_commit"] == C2
    for bad in ({**moved, "repinned_from": {"git_commit": "f" * 40, "image_digest": IMG}},  # not from this release
                {**moved, "data_id": "0" * 64}, {**moved, "config_sha256": "0" * 64}):
        with pytest.raises(ri.RunIdentityError):
            ri._check_previous(old, bad)
    # after the transition the run is on C2: a later attempt with the same record is an ordinary match
    later = {**old, "git_commit": C2}
    ri._check_previous(later, {**moved, "attempt": 3})
    assert ri._release_history(later, {**moved, "attempt": 3}) == []


def test_repin_scope_treats_run_identity_as_operational():
    from ops.jobs import release_repin

    assert "ops/jobs/run_identity.py" in release_repin.REPIN_OPERATIONAL


def test_switch_preflight_blocks_live_or_unknown_physical_attempts_and_pinned_rows():
    from ops.jobs import mig_switch

    instances = inv.validate_allowlist(allowlist())
    state = {"snerl": {"host": "remote", "status": "running", "gpu": PARENTS[0], "attempt": 1},
             "lost": {"host": "remote", "status": "crashed", "unknown": True, "gpu": PARENTS[1], "attempt": 1},
             "done": {"host": "remote", "status": "sync_pending", "gpu": PARENTS[2], "attempt": 1},
             "paused": {"host": "remote", "status": "crashed", "gpu": PARENTS[3], "attempt": 1},
             "local": {"host": "local", "status": "running", "gpu": "MIG-x"}}
    queue = {"hosts": {"remote": {"limits": {PARENTS[0]: {"max_jobs": 20, "max_mem_gb": 90}}}},
             "jobs": [{"id": "pinned", "host": "remote", "gpu": PARENTS[0]}, {"id": "free", "host": "remote"}]}
    report = mig_switch.preflight(state, {"q": queue}, instances)
    assert len(report["blockers"]) == 3 and any("snerl" in b for b in report["blockers"])
    assert any("lost" in b for b in report["blockers"]) and any("pinned" in b for b in report["blockers"])
    assert any("done" in n for n in report["notes"]) and any("rewrite-queue" in n for n in report["notes"])
    clean = mig_switch.preflight({"paused": state["paused"]}, {"q": {"hosts": {}, "jobs": []}}, instances)
    assert clean == {"blockers": [], "notes": []}


def test_queue_rewrite_replaces_only_the_remote_limits(tmp_path):
    from ops.jobs import mig_switch

    doc = allowlist()
    text = ("limits:\n  MIG-local:\n    max_jobs: 3\nhosts:\n  remote:\n    enabled: false\n    hold: true  # keep me\n"
            "    limits:\n" + "".join(f"      {u}:\n        max_jobs: 20\n        max_mem_gb: 90\n" for u in PARENTS)
            + "    gpu_backend: cdi\njobs:\n- id: a\n  host: remote\n")
    new = mig_switch.rewrite_hosts(text, inv.validate_allowlist(doc))
    after = yaml.safe_load(new)
    assert "# keep me" in new and after["jobs"] == [{"id": "a", "host": "remote"}]
    assert after["hosts"]["remote"]["gpu_backend"] == "cdi" and after["limits"] == {"MIG-local": {"max_jobs": 3}}
    assert set(after["hosts"]["remote"]["limits"]) == {i["uuid"] for i in doc["instances"]}
    assert all(v == {"max_jobs": 4, "max_mem_gb": 21.5} for v in after["hosts"]["remote"]["limits"].values())
    allow = tmp_path / "remote-mig.yaml"
    allow.write_text(yaml.safe_dump(doc))
    queue = tmp_path / "queue.yaml"
    queue.write_text(text)
    assert mig_switch.main(["rewrite-queue", "--allowlist", str(allow), "--queue", str(queue), "--write"]) == 0
    assert yaml.safe_load(queue.read_text()) == after and list(tmp_path.glob("queue.yaml.before-mig-*"))


def test_validate_container_still_reconciles_a_historical_physical_attempt(monkeypatch):
    from ops.jobs import remote_backend

    source = (REPO / "ops/jobs/remote_backend.py").read_text()
    assert 'state["gpu"] not in (*APPROVED_HOST_GPUS["remote"], *REMOTE_MIG_PARENTS)' in source


def test_mig_variant_recipe_applies_only_the_container_side_change(tmp_path):
    from ops.jobs import mig_variant

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    files = {"ops/resources/gpu_guard.py": "A = 1\n", "scripts/train.py": "import ops.resources.gpu_guard\n",
             "scripts/remote/entry.py": "x = 1\n", "s4d/model.py": "M = 1\n", "uv.lock": "l\n"}
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git("add", "-A")
    git("commit", "-q", "-m", "old release")
    old = git("rev-parse", "HEAD")
    (repo / "s4d/model.py").write_text("M = 2\n")  # science moved on in main after the old release
    git("commit", "-qam", "main moves on")
    mig_base = git("rev-parse", "HEAD")
    (repo / "ops/resources/gpu_guard.py").write_text("A = 1\nMIG = True\n")
    (repo / "ops/resources/mig_inventory.py").write_text("I = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "MIG support")
    mig_head = git("rev-parse", "HEAD")
    configs = tmp_path / "configs"
    for name in mig_variant.CONFIG_FILES:
        (configs / name).parent.mkdir(parents=True, exist_ok=True)
        (configs / name).write_text("rendered: true\n")
    result = mig_variant.create(repo, old, mig_base, mig_head, configs, ["scripts/train.py"])
    assert result["identical_scope"] and result["base"] == old
    assert sorted(result["changed_files"]) == sorted([*mig_variant.CONFIG_FILES, "ops/resources/gpu_guard.py",
                                                       "ops/resources/mig_inventory.py"])
    show = subprocess.run(["git", "-C", str(repo), "show", f"{result['variant']}:s4d/model.py"], capture_output=True,
                          text=True).stdout
    assert show == "M = 1\n"  # the old release's science, not main's
    expected = mig_variant.expected_operational(repo, old, mig_base, mig_head)
    assert expected["ops/resources/gpu_guard.py"] == b"A = 1\nMIG = True\n" and expected["mujoco_mig_setup.py"] is None
    (configs / mig_variant.CONFIG_FILES[1]).unlink()
    with pytest.raises(mig_variant.VariantError, match="missing rendered"):
        mig_variant.create(repo, old, mig_base, mig_head, configs, ["scripts/train.py"])


def test_cgroup_usage_parses_samples_and_proposes_limits():
    from ops.remote import cgroup_usage as cg

    cid = "a" * 64
    lines = [f"S4D_CG|s4d-rwcal-x-pre-a1|{cid}|{1000 + 300 * i}|{(4 + i) * 2**30}|{3 * 2**30}|max|0|{i * 3_000_000_00}"
             for i in range(13)] + ["junk", f"S4D_CG|other|{cid}|1|2|3|max|0|1"]
    samples = cg.parse_samples("\n".join(lines))
    assert len(samples) == 13 and samples[0]["memory_max"] is None and samples[-1]["memory_peak"] == 16 * 2**30
    proposal = cg.propose(samples, {"rwcal-x-pre": {"ram_gb": 20, "cpu_threads": 8}})[f"s4d-rwcal-x-pre-a1@{cid[:12]}"]
    assert proposal["job"] == "rwcal-x-pre" and proposal["memory_peak_gib"] == 16.0 and proposal["enough_evidence"]
    unreadable = [{**sample, "memory_peak": None, "oom_kill": None} for sample in samples]
    blind = cg.propose(unreadable)[f"s4d-rwcal-x-pre-a1@{cid[:12]}"]
    assert blind["memory_peak_gib"] is None and blind["oom_kills"] is None and not blind["enough_evidence"]
    short = cg.propose(samples[:3])[f"s4d-rwcal-x-pre-a1@{cid[:12]}"]
    assert not short["enough_evidence"]
    assert proposal["proposed_ram_gb"] == math.ceil(16 * 1.25 + 1) and proposal["declared_ram_gb"] == 20
    assert proposal["cpu_cores_p95"] == 1.0 and proposal["proposed_cpu_threads"] == 2
    assert "memory.peak" in cg.SAMPLE_SHELL and "docker ps -q" in cg.SAMPLE_SHELL


def test_repin_hardening_image_chain_patch_and_reserved_field(tmp_path):
    from ops.jobs import release_repin
    from ops.jobs import scheduler as jobs

    stopped = {"host": "remote", "commit": C1, "image_digest": IMG, "status": "crashed", "attempt": 1}
    with pytest.raises(release_repin.RepinError, match="image change"):
        release_repin.check_state("j", stopped, C2, "sha256:" + "d" * 64)
    pending = {**stopped, "commit": C2, "repinned_from": {"commit": C1, "image_digest": IMG, "attempt": 1}}
    for attempt in (1, 2):  # one re-pin per run (Codex review 3, 13), whether or not it was taken up
        with pytest.raises(release_repin.RepinError, match="already re-pinned once"):
            release_repin.check_state("j", {**pending, "attempt": attempt}, "e" * 40, IMG)
    scope = {"operational_differences": ["ops/jobs/run_identity.py", "configs/hosts/remote.yaml"]}
    release_repin.check_operational(scope, {"ops/jobs/run_identity.py": b"reviewed\n"}, lambda name: b"reviewed\n")
    with pytest.raises(release_repin.RepinError, match="reviewed MIG patch"):
        release_repin.check_operational(scope, {"ops/jobs/run_identity.py": b"reviewed\n"}, lambda name: b"other\n")
    queue = {"hosts": {"local": {"enabled": False}}, "jobs": [{"id": "x", "script": "scripts/train.py",
                                                                "repinned_from": {"commit": C1}}]}
    path = tmp_path / "queue.yaml"
    path.write_text(yaml.safe_dump(queue))
    with pytest.raises(ValueError, match="reserved"):
        jobs.load_queue(path)




# Shared exclusive slot guard: executed in an isolated host Python with synthetic commands/files only.
@pytest.fixture
def slot_evidence(tmp_path):
    (tmp_path / 'listing').write_text(render_list([(PARENTS[0], True)]))
    (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)]))
    (tmp_path / 'containers').write_text('{}')
    (tmp_path / 'meminfo').write_text(f'MemAvailable: {1000 * 1024**2} kB\n')
    (tmp_path / 'disk').write_text(f'Size Avail\n{1000 * 2**30} {800 * 2**30}\n')
    script = '''
import json, sys
from pathlib import Path
root = Path(__file__).parent
with (root / 'calls').open('a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
if (root / (Path(sys.argv[0]).name + '.fail')).exists():
    sys.exit(2)
if Path(sys.argv[0]).name == 'nvidia-smi':
    assert sys.argv[1:] in (['-L'], ['-q', '-i', PARENT])
    print((root / ('listing' if sys.argv[1] == '-L' else 'query')).read_text())
elif Path(sys.argv[0]).name == 'df':
    print((root / 'disk').read_text())
else:
    containers = json.loads((root / 'containers').read_text())
    if sys.argv[1] == 'ps':
        assert sys.argv[2:] == ['-q', '--no-trunc', '--filter', 'label=s4d.project=splatter4d',
                              '--filter', 'label=s4d.owner=kaist-sunho']
        print('\\n'.join(containers))
    else:
        if sys.argv[1] == 'top':
            print('PID\\n' + '\\n'.join(map(str, containers[sys.argv[2]].get('Pids', []))))
        else:
            assert sys.argv[1] == 'inspect'
            print(json.dumps([containers[sys.argv[2]]]))
'''.replace('PARENT', repr(PARENTS[0]))
    for command in ('nvidia-smi', 'docker', 'df'):
        path = tmp_path / command
        path.write_text('#!' + sys.executable + '\n' + script)
        path.chmod(0o755)
    return {"uuid": mig_uuid(0, 0), "parent": PARENTS[0], "gpu_instance_id": 3, "device": 0,
            "max_jobs": 1, "idle_mib": 256, "gpu_bytes": 16 * 2**30, "capacity_bytes": int(21.5 * 2**30),
            "ram_bytes": 32 * 2**30, "reserve_bytes": 120 * 2**30, "disk_bytes": 4 * 2**30,
            "floor_bytes": 100 * 2**30, "disk_floor_fraction": 0.15, "root": str(tmp_path),
            "test_paths": {"nvidia_smi": str(tmp_path / 'nvidia-smi'), "docker": str(tmp_path / 'docker'),
                           "df": str(tmp_path / 'df'), "meminfo": str(tmp_path / 'meminfo'),
                           "cgroup_root": str(tmp_path)}}


def run_slot(data):
    from ops.containers.slot_guard import SOURCE

    result = subprocess.run([sys.executable, '-I', '-c', SOURCE, json.dumps(data)],
                            capture_output=True, text=True, timeout=30)
    assert len(result.stdout.splitlines()) == 1, result.stderr
    verdict = json.loads(result.stdout)
    assert verdict['admit'] is (result.returncode == 0)
    return result.returncode, verdict


def slot_container(tmp_path, gpu=None, managed='false', limit=48 * 2**30, current=16 * 2**30, **labels):
    cid = 'a' * 64
    folder = tmp_path / ('docker-' + cid + '.scope')
    folder.mkdir(exist_ok=True)
    (folder / 'memory.stat').write_text(f'anon {current}\nshmem 0\nfile 999999999999\n')
    item = {'Id': cid, 'State': {'Running': True}, 'Config': {'Labels': {
        's4d.project': 'splatter4d', 's4d.owner': 'kaist-sunho', 's4d.managed': managed,
        's4d.host': 'remote', 's4d.gpu': gpu, 's4d.ram_bytes': str(64 * 2**30),
        's4d.disk_bytes': str(10 * 2**30), 's4d.mem_bytes': str(2**30),
        's4d.role': 'rwprep', **labels}}, 'HostConfig': {'Memory': limit, 'NanoCpus': 8 * 10**9,
        'DeviceRequests': [{'Driver': 'cdi', 'DeviceIDs': ['nvidia.com/gpu=' + str(gpu)]}]}}
    (tmp_path / 'containers').write_text(json.dumps({cid: item}))
    return item


def test_slot_admits_idle_instance_and_ignores_sibling_processes(slot_evidence, tmp_path):
    data = slot_evidence
    (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)], processes={0: [(4, 777, 512)]}))
    slot_container(tmp_path, gpu=mig_uuid(0, 1))
    code, result = run_slot(data)
    assert code == 0 and result['pending_ram_bytes'] == 32 * 2**30 and result['containers'] == 0
    assert result['used_mib'] == 67
    calls = [json.loads(line) for line in (tmp_path / 'calls').read_text().splitlines()]
    assert calls[0] == ['nvidia-smi', '-L'] and calls[1] == ['nvidia-smi', '-q', '-i', PARENTS[0]]


@pytest.mark.parametrize('managed', ['true', 'false'])
@pytest.mark.parametrize('state', ['running', 'paused', 'restarting'])
def test_slot_counts_every_owned_live_container(slot_evidence, tmp_path, managed, state):
    item = slot_container(tmp_path, gpu=slot_evidence['uuid'], managed=managed)
    item['State'] = {'Running': state != 'restarting', 'Paused': state == 'paused', 'Restarting': state == 'restarting'}
    (tmp_path / 'containers').write_text(json.dumps({item['Id']: item}))
    code, result = run_slot(slot_evidence)
    assert code == 20 and 'slot container limit reached' in result['reasons'], result


@pytest.mark.parametrize('case,reason', [
    ('process', 'foreign processes on instance'), ('framebuffer', 'instance framebuffer above idle threshold'),
    ('capacity', 'declared GPU need exceeds capacity'), ('ram', 'host RAM reserve'),
    ('disk_need', 'disk floor'), ('disk_floor', 'disk floor'), ('disk_fraction', 'disk floor'),
    ('reserved_disk', 'disk floor'),
])
def test_slot_resource_refusals(slot_evidence, tmp_path, case, reason):
    data = slot_evidence
    if case == 'process':
        (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)], processes={0: [(3, 777, 0)]}))
    elif case == 'framebuffer':
        path = tmp_path / 'query'
        path.write_text(path.read_text().replace(': 67 MiB', ': 257 MiB'))
    elif case == 'capacity':
        data['gpu_bytes'] = data['capacity_bytes'] + 1
    elif case == 'ram':
        slot_container(tmp_path, current=0)
        (tmp_path / 'meminfo').write_text(f'MemAvailable: {180 * 1024**2} kB\n')  # 180-48 < 120+32
    elif case == 'disk_need':
        data['disk_bytes'] = 651 * 2**30
    elif case == 'disk_floor':
        data['floor_bytes'] = 797 * 2**30
    elif case == 'disk_fraction':
        data['disk_floor_fraction'] = 0.8
    else:
        slot_container(tmp_path, managed='true')
        data['disk_bytes'] = 641 * 2**30
    code, result = run_slot(data)
    assert code == 20 and reason in result['reasons']


@pytest.mark.parametrize('case', [
    'missing_uuid', 'duplicate_uuid', 'wrong_parent', 'wrong_device', 'wrong_gi', 'wrong_query_parent',
    'missing_instance', 'duplicate_instance', 'bad_framebuffer', 'missing_processes', 'denied_processes',
    'unfinished_process', 'unattributed_process', 'meminfo', 'cgroup_missing', 'cgroup_bad', 'ram_unknown',
    'docker', 'nvidia-smi', 'df', 'disk_bad', 'inspect_bad', 'max_jobs', 'capacity_drift',
])
def test_slot_evidence_errors_fail_closed(slot_evidence, tmp_path, case):
    data = slot_evidence
    path = tmp_path / 'query'
    query = path.read_text()
    if case in ('missing_uuid', 'duplicate_uuid', 'wrong_parent', 'wrong_device'):
        listing = tmp_path / 'listing'
        text = listing.read_text()
        if case == 'missing_uuid':
            text = text.replace(data['uuid'], mig_uuid(3, 3))
        elif case == 'duplicate_uuid':
            text += text
        elif case == 'wrong_parent':
            text = text.replace(PARENTS[0], PARENTS[1])
        else:
            text = text.replace('Device  0:', 'Device  9:')
        listing.write_text(text)
    elif case == 'wrong_gi':
        data['gpu_instance_id'] = 9
    elif case == 'wrong_query_parent':
        query = query.replace(PARENTS[0], PARENTS[1])
    elif case == 'missing_instance':
        query = render_query([(PARENTS[0], True)], per_parent=0)
    elif case == 'duplicate_instance':
        query = query.replace('Index                                          : 1', 'Index                                          : 0')
    elif case == 'bad_framebuffer':
        query = query.replace(': 24192 MiB', ': N/A')
    elif case == 'missing_processes':
        query = query[:query.index('    Processes')]
    elif case == 'denied_processes':
        query = render_query([(PARENTS[0], True)], process_value='Insufficient Permissions')
    elif case in ('unfinished_process', 'unattributed_process'):
        query = render_query([(PARENTS[0], True)], processes={0: [(4, 777, 512)]})
        if case == 'unfinished_process':
            query = query[:query.index('            Used GPU Memory')]
        else:
            query = query.replace('        GPU instance ID                                : 4\n', '')
    elif case == 'meminfo':
        (tmp_path / 'meminfo').write_text('MemAvailable: N/A\n')
    elif case in ('cgroup_missing', 'cgroup_bad', 'ram_unknown'):
        slot_container(tmp_path, limit=0 if case == 'ram_unknown' else 48 * 2**30,
                       **({'s4d.ram_bytes': ''} if case == 'ram_unknown' else {}))
        cgroup = tmp_path / ('docker-' + 'a' * 64 + '.scope') / 'memory.stat'
        if case == 'cgroup_missing':
            cgroup.unlink()
        elif case == 'cgroup_bad':
            cgroup.write_text('N/A')
    elif case in ('docker', 'nvidia-smi', 'df'):
        (tmp_path / (case + '.fail')).touch()
    elif case == 'disk_bad':
        (tmp_path / 'disk').write_text('Size Avail\n100 101\n')
    elif case == 'inspect_bad':
        (tmp_path / 'containers').write_text(json.dumps({'b' * 64: {}}))
    elif case == 'max_jobs':
        data['max_jobs'] = 0
    elif case == 'capacity_drift':
        data['capacity_bytes'] = 25 * 2**30
    path.write_text(query)
    code, result = run_slot(data)
    assert code not in (0, 20) and result['error']


def test_slot_ram_exact_resource_boundaries_and_no_declaration_fallback(slot_evidence, tmp_path):
    data = slot_evidence
    slot_container(tmp_path, limit=64 * 2**30, current=16 * 2**30)
    data.update(gpu_bytes=21 * 2**30, disk_bytes=650 * 2**30)
    (tmp_path / 'meminfo').write_text(f'MemAvailable: {200 * 1024**2} kB\n')  # 200 - (64-16) == 120+32
    path = tmp_path / 'query'
    path.write_text(path.read_text().replace(': 67 MiB', ': 256 MiB'))
    code, result = run_slot(data)
    assert code == 0 and result['pending_ram_bytes'] == 48 * 2**30  # anon only: the file cache is not credited
    slot_container(tmp_path, limit=0, current=16 * 2**30)  # Codex review 3 (16): no enforced limit -> refused
    code, result = run_slot(data)
    assert code != 0 and not result['admit']


def test_slot_guard_rechecks_the_parent_mig_mode_inside_the_lock(slot_evidence, tmp_path):
    code, verdict = run_slot(slot_evidence)
    assert code == 0 and verdict["admit"]
    (tmp_path / "query").write_text(render_query([(PARENTS[0], True)], mode=("Enabled", "Disabled")))
    code, verdict = run_slot(slot_evidence)
    assert code != 0 and not verdict["admit"]


def test_release_reservation_accepts_all_16_mig_slots_and_charges_exclusively(monkeypatch):
    from ops.jobs import release_reservation as rr, scheduler
    from ops.resources import gpu_guard
    import importlib.util

    monkeypatch.setattr(gpu_guard, 'REMOTE_MIG_ACTIVE', True)
    spec = importlib.util.spec_from_file_location('mig_verification_policy', REPO / 'ops/containers/verification.py')
    verification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verification)
    assert verification.POLICIES['native_suite'][3] == 12
    gpus = [i['uuid'] for i in inv.validate_allowlist(allowlist())]
    monkeypatch.setattr(scheduler, 'APPROVED_HOST_GPUS', {**gpu_guard.APPROVED_HOST_GPUS, 'remote': tuple(gpus)})
    native = dict(mem_gb=verification.POLICIES['native_suite'][3], load=1, ram_gb=96, cpu_threads=32, disk_gb=1)
    isolation = dict(mem_gb=4, load=0, ram_gb=8, cpu_threads=8, disk_gb=1)
    for stage in ('native', 'isolation'):
        doc = rr.validate(rr.plan(C1, C1[:7], stage, gpus[0], gpus, native=native, isolation=isolation))
        assert len(doc['gpus']) == 16 and len(doc['jobs']) == (17 if stage == 'native' else 16)
        charges = rr.charges(doc)
        limits = inv.queue_limits(inv.validate_allowlist(allowlist()))
        assert max(c['mem_gb'] for c in charges['gpus'].values()) <= 22.625
        phantom = [dict(host='remote', gpu=g, mem_gb=c['mem_gb'], exclusive_gpu=True) for g, c in charges['gpus'].items()]
        assert scheduler.choose_gpu(dict(host='remote', gpu='any', mem_gb=1), limits, phantom, set()) is None
    with pytest.raises(ValueError, match='gpu entry'):
        rr.validate({**doc, 'gpus': {'MIG-malformed': dict(mem_gb=1, load=0)}})


@pytest.mark.parametrize('managed', ['true', 'false'])
def test_capacity_guard_shares_with_training_or_rwprep_and_reserves_growth(slot_evidence, tmp_path, managed):
    data = {**slot_evidence, 'max_jobs': 4, 'gpu_bytes': 3 * 2**30}
    item = slot_container(tmp_path, gpu=data['uuid'], managed=managed, **{'s4d.mem_bytes': str(3 * 2**30)})
    item['Pids'] = [777]
    (tmp_path / 'containers').write_text(json.dumps({item['Id']: item}))
    (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)], processes={0: [(3, 777, 512)]}))
    code, result = run_slot(data)
    assert code == 0 and result['containers'] == 1
    assert result['gpu_reserved_bytes'] == 3 * 2**30
    assert result['gpu_pending_bytes'] == int(2.5 * 2**30)
    data['gpu_bytes'] = 19 * 2**30
    code, result = run_slot(data)
    assert code == 20 and 'GPU reservation sum exceeds capacity' in result['reasons']


@pytest.mark.parametrize('exclusive_new,exclusive_old', [(True, False), (False, True)])
def test_capacity_guard_honours_both_directions_of_exclusivity(slot_evidence, tmp_path, exclusive_new, exclusive_old):
    data = {**slot_evidence, 'max_jobs': 4, 'gpu_bytes': 2**30, 'exclusive_gpu': exclusive_new}
    slot_container(tmp_path, gpu=data['uuid'], managed='true',
                   **{'s4d.exclusive_gpu': str(exclusive_old).lower()})
    code, result = run_slot(data)
    assert code == 20 and 'exclusive GPU placement' in result['reasons']


def test_capacity_guard_refuses_foreign_pid_even_beside_known_tenant(slot_evidence, tmp_path):
    data = {**slot_evidence, 'max_jobs': 4, 'gpu_bytes': 2**30}
    slot_container(tmp_path, gpu=data['uuid'], managed='true')
    (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)], processes={0: [(3, 12345, 100)]}))
    code, result = run_slot(data)
    assert code == 20 and 'foreign processes on instance' in result['reasons']


def test_capacity_guard_checks_free_memory_headroom_and_cpu(slot_evidence, tmp_path):
    data = {**slot_evidence, 'max_jobs': 4, 'gpu_bytes': 3 * 2**30, 'cpu_thread_budget': 12}
    slot_container(tmp_path, gpu=data['uuid'], managed='true')
    code, result = run_slot(data)
    assert code == 20 and 'host CPU thread budget' in result['reasons']
    data['cpu_thread_budget'] = 320
    path = tmp_path / 'query'
    path.write_text(path.read_text().replace(': 67 MiB', ': 21000 MiB'))
    code, result = run_slot(data)
    assert code == 20 and 'measured free GPU memory/headroom' in result['reasons']


@pytest.mark.parametrize('host', ['local', 'remote'])
def test_scheduler_shares_but_refuses_memory_exclusive_and_foreign(host):
    from ops.jobs import scheduler
    gpu = scheduler.APPROVED_HOST_GPUS[host][0]
    limits = {gpu: {'max_jobs': 4, 'max_mem_gb': 21.5}}
    row = {'id': 'new', 'host': host, 'mem_gb': 3}
    running = [{'host': host, 'gpu': gpu, 'mem_gb': 3}]
    assert scheduler.choose_gpu(row, limits, running, set()) == gpu
    assert scheduler.choose_gpu({**row, 'mem_gb': 19}, limits, running, set()) is None
    assert scheduler.choose_gpu({**row, 'exclusive_gpu': True}, limits, running, set()) is None
    assert scheduler.choose_gpu(row, limits, [{**running[0], 'exclusive_gpu': True}], set()) is None
    assert scheduler.choose_gpu(row, limits, running, {gpu}) is None
    assert scheduler.choose_gpu({**row, 'verification_purpose': 'native_suite'}, limits, running, set()) is None


def test_committed_post_reboot_allowlist_and_limits():
    instances = inv.load_allowlist(REPO / inv.INVENTORY_RELATIVE)
    assert len(instances) == 16
    assert {item['parent'] for item in instances} == set(PARENTS)
    assert all(sum(item['parent'] == parent for item in instances) == 4 for parent in PARENTS)
    host = yaml.safe_load((REPO / 'configs/hosts/remote.yaml').read_text())
    assert set(host['gpu_uuids']) == {item['uuid'] for item in instances}
    assert host['limits'] == inv.queue_limits(instances)
    assert all(item['max_jobs'] == 4 and item['max_mem_gb'] == 21.5 for item in instances)


def test_rwprep_uses_the_same_capacity_guard_and_release_reservation(monkeypatch):
    from ops.containers import rwprep_capacity as adapter
    from ops.containers.slot_guard import SOURCE

    instances = inv.validate_allowlist(allowlist())
    monkeypatch.setattr(adapter, 'REMOTE_MIG_INSTANCES', instances)
    monkeypatch.setattr(adapter.release_reservation, 'reserved_gpus', lambda: set())
    job = {'gpu': instances[0]['uuid'], 'gpu_need_bytes': 4 * 2**30, 'ram_bytes': 8 * 2**30,
           'cpus': 4, 'test_paths': {'docker': '/untrusted'}}
    payload = adapter.guard_payload(job)
    assert payload['source'] == SOURCE
    assert payload['input']['max_jobs'] == 4 and payload['input']['capacity_bytes'] == 21.5 * 2**30
    assert 'test_paths' not in payload['input']
    assert adapter.guard_payload({**job, 'exclusive_gpu': True})['input']['exclusive_gpu'] is True
    with pytest.raises(ValueError, match='memory cap'):
        adapter.guard_payload({**job, 'gpu_cap_bytes': 90 * 2**30})
    monkeypatch.setattr(adapter.release_reservation, 'reserved_gpus', lambda: {job['gpu']})
    with pytest.raises(ValueError, match='reserved for release verification'):
        adapter.guard_payload(job)


def test_rwprep_cpu_budget_and_ram_reserve_follow_the_scheduler_queue(monkeypatch, tmp_path):
    """rwprep admits against the SAME hosts.remote budget the scheduler reads (no hard-coded 320)."""
    from ops.containers import rwprep_capacity as adapter
    from ops.jobs import scheduler

    assert adapter.DEFAULT_CPU_THREAD_BUDGET == scheduler.REMOTE_CPU_THREAD_BUDGET
    instances = inv.validate_allowlist(allowlist())
    monkeypatch.setattr(adapter, 'REMOTE_MIG_INSTANCES', instances)
    monkeypatch.setattr(adapter.release_reservation, 'reserved_gpus', lambda: set())
    job = {'gpu': instances[0]['uuid'], 'gpu_need_bytes': 2 * 2**30, 'ram_bytes': 8 * 2**30, 'cpus': 4}
    queue = tmp_path / 'queue.yaml'
    queue.write_text(yaml.safe_dump({'hosts': {'remote': {'host_ram_reserve_gb': 120}}, 'jobs': []}))
    data = adapter.guard_payload(job, queue)['input']
    assert data['cpu_thread_budget'] == 320 and data['reserve_bytes'] == 120 * 2**30
    queue.write_text(yaml.safe_dump({'hosts': {'remote': {'cpu_thread_budget': 352, 'host_ram_reserve_gb': 130}},
                                     'jobs': []}))
    data = adapter.guard_payload(job, queue)['input']
    assert data['cpu_thread_budget'] == 352 and data['reserve_bytes'] == 130 * 2**30
    for bad_reserve in (100, None, True):  # below launch.py's 120 GiB floor, or not a number
        queue.write_text(yaml.safe_dump({'hosts': {'remote': {'host_ram_reserve_gb': bad_reserve}}, 'jobs': []}))
        with pytest.raises(ValueError, match='host_ram_reserve_gb'):
            adapter.guard_payload(job, queue)
    for malformed in ('', 'hosts: null\n', 'hosts: {remote: null}\n', 'hosts: {remote: []}\n', 'hosts: {remote: false}\n'):
        queue.write_text(malformed)
        with pytest.raises(ValueError, match='hosts.remote must be a mapping'):
            adapter.guard_payload(job, queue)
    for bad in (0, 2000, 352.0, '352', True, None):
        queue.write_text(yaml.safe_dump({'hosts': {'remote': {'cpu_thread_budget': bad}}, 'jobs': []}))
        with pytest.raises(ValueError, match='cpu_thread_budget'):
            adapter.guard_payload(job, queue)
        with pytest.raises(ValueError, match='cpu_thread_budget'):
            scheduler.load_queue(queue)


def test_local_fair_share_keeps_exclusive_placement_and_shared_capacity():
    from ops.jobs.fair_share import select
    limits = {'mig': {'max_jobs': 3, 'max_mem_gb': 22}}
    first = {'id': 'a', 'family': 'realworld', 'ram_bytes': 3 * 2**30, 'mem_gb': 3, 'gpu': 'mig'}
    second = {**first, 'id': 'b'}
    assert select([first, second], [], limits, 100 * 2**30, 0)['admitted'] == ['a', 'b']
    assert select([{**first, 'exclusive_gpu': True}, second], [], limits, 100 * 2**30, 0)['admitted'] == ['a']
    assert select([{**second, 'exclusive_gpu': True}], [first], limits, 100 * 2**30, 0)['admitted'] == []


def test_local_gpu_query_requires_measured_usable_capacity(monkeypatch):
    from ops.jobs import scheduler
    gpu = scheduler.APPROVED_HOST_GPUS['local'][0]
    monkeypatch.setattr(scheduler, 'mig_query', lambda *a, **kw: {gpu: {'processes': []}})
    with pytest.raises(ValueError, match='framebuffer evidence unavailable'):
        scheduler.gpu_usage()


@pytest.mark.parametrize('managed', ['true', 'false'])
def test_scheduler_and_guard_same_snapshot_boundaries(slot_evidence, tmp_path, monkeypatch, managed):
    """Exact parity (stronger than fits => admit), including sincro's 961/965 MiB.

    Every item/measurement is shared by the scheduler fit and decide's test_paths.
    The grid exercises measured excess, pending growth, count, foreign/idle,
    exclusivity, RAM, CPU, disk and byte-rounding boundaries without Docker/SSH.
    """
    import itertools
    from ops.jobs import scheduler, remote_backend as remote
    gpu = slot_evidence['uuid']
    monkeypatch.setattr(scheduler, 'APPROVED_HOST_GPUS', {'remote': (gpu,)})
    base = dict(total=24192, used=961, measured=450, declared=6 * 2**30,
                need=int(9.5 * 2**30), tenants=2, jobs=4, available=1000 * 2**30,
                cpu=320, disk_free=800 * 2**30, exclusive=False, foreign=False)
    cases = [dict(base, used=used, need=need) for used, need in itertools.product(
        [900, 961, 965, 1024], [int(9.5 * 2**30) - 128 * 2**20, int(9.5 * 2**30), int(9.5 * 2**30) + 1])]
    cases += [dict(base, **override) for override in [
        dict(measured=7000, declared=2**30, need=8 * 2**30),
        dict(measured=7000, declared=2**30, need=9 * 2**30),
        dict(tenants=0, used=256), dict(tenants=0, used=257),
        dict(jobs=2), dict(exclusive=True), dict(foreign=True),
        dict(available=216 * 2**30), dict(available=216 * 2**30 - 1024),
        dict(cpu=24), dict(cpu=23),
        dict(disk_free=(174 if managed == 'true' else 154) * 2**30),
        dict(disk_free=(174 if managed == 'true' else 154) * 2**30 - 1),
        dict(need=22 * 2**30), dict(total=22016), dict(total=22015),
    ]]
    # Run decide directly with the same synthetic commands; no entry-point execution.
    from ops.containers.slot_guard import SOURCE
    namespace = {}
    exec(SOURCE[:SOURCE.index('\ntry:\n')], namespace)
    for case in cases:
        data = dict(slot_evidence, max_jobs=case['jobs'], gpu_bytes=case['need'],
                    cpu_thread_budget=case['cpu'], exclusive_gpu=case['exclusive'])
        records, items, own, state = [], {}, [], {}
        processes = [(3, 777 + i, case['measured']) for i in range(case['tenants'])]
        if case['foreign']:
            processes.append((3, 999, 1))
        (tmp_path / 'query').write_text(render_query([(PARENTS[0], True)], memory=case['total'],
            processes={0: processes}).replace(': 67 MiB', f": {case['used']} MiB"))
        for i in range(case['tenants']):
            item = slot_container(tmp_path, gpu=gpu, managed=managed,
                                  **{'s4d.mem_bytes': str(case['declared'])})
            item['Id'] = str(i + 1) * 64
            item['Pids'] = [777 + i]
            item['Name'] = '/s4d-old' + str(i)
            folder = tmp_path / ('docker-' + item['Id'] + '.scope')
            folder.mkdir(exist_ok=True)
            (folder / 'memory.stat').write_text(f'anon {16 * 2**30}\nshmem 0\n')
            records.append((item, set(item['Pids']), 16 * 2**30))
            items[item['Id']] = item
            if managed == 'true':
                state[str(i)] = dict(host='remote', status='running', gpu=gpu,
                                     container=item['Name'][1:], mem_gb=case['declared'] / 2**30)
        (tmp_path / 'containers').write_text(json.dumps(items))
        (tmp_path / 'meminfo').write_text(f"MemAvailable: {case['available'] // 1024} kB\n")
        (tmp_path / 'disk').write_text(f"Size Avail\n{1000 * 2**30} {case['disk_free']}\n")
        containers = {item['Name'][1:]: {**item, 'pids': pids} for item, pids, _ in records if managed == 'true'}
        snapshot = remote.RemoteSnapshot({gpu: [(pid, mem) for _, pid, mem in processes]},
            case['available'] / 2**30, 1000 * 2**30, case['disk_free'], containers,
            gpu_memory={gpu: (case['total'], case['used'])}, admission_records=records)
        resource = {'snapshot': snapshot, 'available': snapshot.available_gb,
                    'disk_total': snapshot.disk_total_bytes, 'disk_free': snapshot.disk_free_bytes}
        scheduler._charge_cotenants({'remote': resource}, state, {}, [])
        running = [dict(host='remote', gpu=gpu, mem_gb=value['mem_gb']) for value in state.values()]
        job = dict(id='new', host='remote', mem_gb=case['need'] / 2**30, ram_gb=32, disk_gb=4,
                   exclusive_gpu=case['exclusive'])
        host = dict(host_ram_reserve_gb=120, disk_floor_gb=100, disk_floor_fraction=.15,
                    cpu_thread_budget=case['cpu'])
        limits = {gpu: dict(max_jobs=case['jobs'], max_mem_gb=21.5)}
        fits = (scheduler.choose_gpu(job, limits, running, set(), resource['cotenant']) is not None
                and not scheduler.remote_host_refusals(resource, job, host, resource['guard_cpu_used']))
        try:
            verdict = namespace['decide'](data)
        except ValueError:
            assert not fits, case
        else:
            assert fits == verdict['admit'], (case, verdict)
            if case['used'] in (961, 965) and case['need'] == int(9.5 * 2**30):
                assert not fits and 'measured free GPU memory/headroom' in verdict['reasons']


@pytest.mark.parametrize('damage', [None, 'ram', 'pids', 'cpu', 'binding'])
def test_complete_snapshot_preserves_guard_evidence(slot_evidence, tmp_path, monkeypatch, damage):
    from ops.jobs import remote_backend as remote
    from tests.test_remote_jobs import snapshot_output
    gpu = slot_evidence['uuid']
    monkeypatch.setattr(remote, 'APPROVED_HOST_GPUS', {'remote': (gpu,)})
    monkeypatch.setattr(remote, 'REMOTE_MIG_PARENTS', (PARENTS[0],))
    monkeypatch.setattr(remote, 'REMOTE_MIG_DEVICES', {gpu: (PARENTS[0], 3, 0)})
    item = slot_container(tmp_path, gpu=gpu, managed='true')
    item['Name'] = '/s4d-owned'
    if damage == 'cpu':
        item['HostConfig']['NanoCpus'] = 0
    if damage == 'binding':
        item['HostConfig']['DeviceRequests'][0]['Driver'] = 'nvidia'
    listing, query = render_list([(PARENTS[0], True)]), render_query([(PARENTS[0], True)])
    text = snapshot_output().replace('__S4D_GPUS__\n', '__S4D_GPUS__\n99, ' + PARENTS[0] + '\n')
    text = text.replace('__S4D_END__', '\n'.join([
        '__S4D_NAMES__', 's4d-owned', 's4d-unowned-exited', '__S4D_MIG_LIST__', listing,
        '__S4D_MIG_QUERY__', query, '__S4D_COTENANTS__', json.dumps(item),
        '__S4D_TOP__ ' + item['Id'], 'PID' if damage != 'pids' else 'unreadable',
        '__S4D_RAM_USED__ ' + str(16 * 2**30) if damage != 'ram' else '',
        '__S4D_END_TOP__', '__S4D_END__']))
    value = remote.parse_snapshot(text)
    assert value.container_names == {'s4d-owned', 's4d-unowned-exited'}
    if damage == 'ram':
        assert not value.resources_ready
    elif damage == 'cpu':
        with pytest.raises(ValueError):
            remote.cotenant_charges(value)
    else:
        assert value.resources_ready
        charge = remote.cotenant_charges(value)
        if damage in ('pids', 'binding'):
            assert charge['gpus'][gpu]['admission_error']
        else:
            assert charge['guard_accounted'] and charge['ram_pending_bytes'] == 32 * 2**30
            assert charge['cpus'] == 8 and charge['reserved_disk_bytes'] == 10 * 2**30


@pytest.mark.parametrize('status', ['running', 'exited'])
def test_legacy_mig_snapshot_blocks_launches_but_preserves_reconciliation(tmp_path, monkeypatch, status):
    from tests.test_remote_jobs import (jobs, remote, FakeBackend, job, write_queue, tick,
                                        inspected, snapshot_output, launch_event, GIB, CONTAINER_ID)

    gpu = mig_uuid(0, 0)
    approved = {**jobs.APPROVED_HOST_GPUS, 'remote': (gpu,)}
    monkeypatch.setattr(jobs, 'APPROVED_HOST_GPUS', approved)
    monkeypatch.setattr(remote, 'APPROVED_HOST_GPUS', approved)
    monkeypatch.setattr(remote, 'REMOTE_MIG_PARENTS', (PARENTS[0],))
    monkeypatch.setattr(remote, 'REMOTE_MIG_DEVICES', {gpu: (PARENTS[0], 3, 0)})
    item = inspected('active', gpu=gpu, status=status)
    item['HostConfig'].update(Memory=48 * GIB, NanoCpus=8 * 10**9)
    # An old container still has unallocated RAM growth after the legacy ramp expires.
    text = snapshot_output([item]).replace('MemAvailable: 209715200 kB', 'MemAvailable: 62914560 kB')
    text = text.replace('__S4D_GPUS__\n', '__S4D_GPUS__\n99, ' + PARENTS[0] + '\n')
    text = text.replace('__S4D_END__', '\n'.join([
        '__S4D_MIG_LIST__', render_list([(PARENTS[0], True)]),
        '__S4D_MIG_QUERY__', render_query([(PARENTS[0], True)], processes={0: [(3, 123, 67)]}),
        '__S4D_END__']))
    snapshot = remote.parse_snapshot(text)
    assert snapshot.resources_ready and snapshot.admission_records is None
    assert snapshot.gpu_memory[gpu] == (24192, 67)
    with pytest.raises(ValueError, match='missing MIG admission inventory'):
        remote.cotenant_charges(snapshot)

    backend = FakeBackend(tmp_path / 'runs')
    backend.value = snapshot
    queue = write_queue(tmp_path, [job('active', gpu=gpu), job('new', gpu=gpu)],
                        limits={gpu: {'max_jobs': 4, 'max_mem_gb': 21.5}})
    launch_event(tmp_path, 'active', gpu=gpu, time=1)
    for _ in range(2):
        messages = tick(tmp_path, queue, backend)
        current = jobs.read_registry(tmp_path / 'registry.jsonl')['active']
        assert current['status'] == ('done' if status == 'exited' else 'running')
        assert not current['unknown'] and current['attempts'] == 1 and current['failures'] == 0
        assert not backend.launches
        assert any('admission inventory' in message for message in messages)
    if status == 'exited':
        assert current['sync']['final'] and backend.backups == ['active']
        assert len(backend.removals) == 1 and current['containers_removed'] == [CONTAINER_ID]
    else:
        assert not backend.removals


@pytest.mark.parametrize('source', ['allowlist', 'usage', 'memory', 'parents'])
def test_missing_admission_records_fail_closed_for_any_mig_host_evidence(monkeypatch, source):
    from ops.jobs import remote_backend as remote

    gpu = mig_uuid(0, 0)
    monkeypatch.setattr(remote, 'APPROVED_HOST_GPUS', {'remote': (gpu,) if source == 'allowlist' else ()})
    monkeypatch.setattr(remote, 'REMOTE_MIG_PARENTS', (PARENTS[0],) if source == 'parents' else ())
    snapshot = remote.RemoteSnapshot({gpu: []} if source == 'usage' else {}, 200, 1000, 800, {},
                                     gpu_memory={gpu: (24192, 67)} if source == 'memory' else {})
    with pytest.raises(ValueError, match='missing MIG admission inventory'):
        remote.cotenant_charges(snapshot)


def test_snapshot_uses_guard_identity_mode_and_process_memory_requirements():
    from ops.jobs import remote_backend as remote
    devices = devices_of(allowlist())
    listing, query = remote_host(processes={0: [(3, 11, 100)]})
    assert remote.mig_usage(listing.splitlines(), query.splitlines(), devices)[2]
    for bad_listing, bad_query in [
        (listing + next(line for line in listing.splitlines() if mig_uuid(0, 0) in line) + '\n', query),
        (listing, query.replace(': Enabled', ': Disabled', 1)),
        (listing, query.replace(': 100 MiB', ': N/A')),
    ]:
        assert not remote.mig_usage(bad_listing.splitlines(), bad_query.splitlines(), devices)[2]


def test_owned_cpu_threads_never_double_counts_guard_accounted_live_containers():
    """2026-11-12: guard-accounted charges['cpus'] already includes every live owned container (managed rows too).
    The remote driver once added its own reservation sum on top (40 x 8 + 328 > 352): nothing was ever admitted."""
    from ops.jobs.remote_backend import owned_cpu_threads
    rows = [(f's4d-job{i}-a1', 8) for i in range(40)]
    guard = {'guard_accounted': True, 'cpus': 328.0, 'live_names': {name for name, _ in rows} | {'s4d-rwx-temem-m1'}}
    assert owned_cpu_threads(guard, rows) == 328.0
    # a reserving row whose container is not live yet is charged once, by its reservation
    assert owned_cpu_threads(guard, rows + [('s4d-new-a1', 8)]) == 336.0
    # legacy charges count co-tenants only: every reserving row is added
    assert owned_cpu_threads({'cpus': 4.0}, rows) == 324.0
    assert owned_cpu_threads(None, rows) == 320.0


def test_remote_driver_uses_owned_cpu_threads():
    driver = (REPO / "ops/campaign/remote_campaign_driver.py").read_text()
    assert "cpu_used = owned_cpu_threads(COTENANT" in driver
    assert "cpu_used += (COTENANT or {}).get('cpus', 0) or 0" not in driver
