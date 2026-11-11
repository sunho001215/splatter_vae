"""Remote MIG migration (user directive 2026-11-11): inventory discovery/validation, the fail-closed allowlist, MIG-aware
snapshot / launch-guard evidence, the outage guard's hold ownership and the release re-pin event. Synthetic
``nvidia-smi`` evidence only (the layouts of driver 580 as observed on the local MIG GPUs); nothing touches a GPU."""

from __future__ import annotations

import json
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
    assert inv.device_capacity_gib(instances[0]["uuid"], instances) == pytest.approx(24192 / 1024 - 1.0, abs=1e-3)
    assert inv.device_capacity_gib(PARENTS[0], ()) == 90.0
    limits = inv.queue_limits(instances)
    assert set(limits) == {item["uuid"] for item in instances} and all(v["max_jobs"] == 2 for v in limits.values())
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
                                                                         "ops/resources/mig_inventory.py"}
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
    assert "range(len(APPROVED_HOST_GPUS['remote']))" in (REPO / "ops/campaign/release_chain.py").read_text()


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
    assert "mig.mode.current,mig.mode.pending" in command and "python" not in command
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

    source = (REPO / "ops/jobs/scheduler.py").read_text()
    assert '(registry.parent.parent / "runs/setup/remote-hold-admissions").exists()' in source
    queue = {"hosts": {"local": {"enabled": False}, "remote": {"enabled": True, "limits": {
        PARENTS[0]: {"max_jobs": 1, "max_mem_gb": 10}}}}, "jobs": []}
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
