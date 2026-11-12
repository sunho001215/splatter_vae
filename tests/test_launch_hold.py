"""Durable remote launch hold tied to the boot id (remote MIG migration, 2026-11-11)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ops.remote import launch_hold as lh

BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def run(fragment: str):
    return subprocess.run(["sh", "-c", f"set -eu; {fragment}\necho ALLOWED"], capture_output=True, text=True, timeout=30)


def test_hold_is_armed_by_default_and_lifted_only_by_a_record_for_this_boot(tmp_path):
    folder = str(tmp_path)
    check = lh.check_shell(folder=folder)
    held = run(check)
    assert held.returncode == lh.EXIT_HELD and "launch held" in held.stderr
    other = lh.resume_record("00000000-0000-4000-8000-000000000000", "user ok")
    (tmp_path / f"resume-{BOOT}.json").write_text(json.dumps(other))  # a record of another boot under this name
    assert run(check).returncode == lh.EXIT_HELD
    (tmp_path / f"resume-{BOOT}.json").write_text(json.dumps(lh.resume_record(BOOT, "user: resume"), sort_keys=True))
    assert run(check).stdout.strip() == "ALLOWED"
    (tmp_path / "HOLD").write_text("{}")
    assert run(check).returncode == lh.EXIT_HELD  # an explicit hold wins over the resume record


def test_status_shell_and_parser_fail_closed(tmp_path):
    out = subprocess.run(["sh", "-c", lh.status_shell(folder=str(tmp_path))], capture_output=True, text=True).stdout
    state = lh.parse_status(out)
    assert state == {"boot_id": BOOT, "hold": False, "resumed": False, "held": True}
    (tmp_path / f"resume-{BOOT}.json").write_text(json.dumps(lh.resume_record(BOOT, "ok"), sort_keys=True))
    out = subprocess.run(["sh", "-c", lh.status_shell(folder=str(tmp_path))], capture_output=True, text=True).stdout
    assert lh.parse_status(out)["held"] is False
    for bad in ("", "__S4D_LAUNCH_HOLD__ x 0 1", f"__S4D_LAUNCH_HOLD__ {BOOT} 0 1\n__S4D_LAUNCH_HOLD__ {BOOT} 0 1",
                f"__S4D_LAUNCH_HOLD__ {BOOT} 0 2"):
        assert lh.parse_status(bad)["held"] is True


def test_resume_record_requires_authorization_and_a_boot_id():
    with pytest.raises(ValueError):
        lh.resume_record(BOOT, " ")
    with pytest.raises(ValueError):
        lh.resume_record("not-a-boot", "ok")
    with pytest.raises(ValueError):
        lh.paths("/tmp")


def snapshot_text(extra: str = "") -> str:
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS

    gpus = "\n".join(f"{i}, {uuid}" for i, uuid in enumerate(APPROVED_HOST_GPUS["remote"]))
    return (f"__S4D_GPUS__\n{gpus}\n__S4D_USAGE__\n__S4D_GRAPHICS__\n__S4D_MEMORY__\nMemAvailable: 1048576000 kB\n"
            f"__S4D_DISK__\n1000 500\n__S4D_CONTAINERS__\n{extra}__S4D_END__\n")


def test_snapshot_carries_the_launch_hold_state():
    from ops.jobs import remote_backend

    assert remote_backend.parse_snapshot(snapshot_text()).launch_hold is None  # legacy fixture format
    held = remote_backend.parse_snapshot(snapshot_text(f"__S4D_LAUNCH_HOLD__ {BOOT} 0 0\n")).launch_hold
    assert held["held"] is True and held["boot_id"] == BOOT
    free = remote_backend.parse_snapshot(snapshot_text(f"__S4D_LAUNCH_HOLD__ {BOOT} 0 1\n")).launch_hold
    assert free["held"] is False
    assert "__S4D_LAUNCH_HOLD__" in remote_backend.snapshot_command(remote_backend.ROOT)


def test_every_scheduler_launch_checks_the_hold_inside_the_admission_lock():
    source = (Path(__file__).resolve().parents[1] / "ops/containers/launch.py").read_text()
    lock = source.index("flock -x 9")
    check = source.index("launch_hold.check_shell(root)")
    run = source.index('lines += ["cid=$(" + shlex.join(docker) + ")"')
    assert lock < check < run
    scheduler = (Path(__file__).resolve().parents[1] / "ops/jobs/scheduler.py").read_text()
    assert '(snapshot.launch_hold or {}).get("held")' in scheduler


def test_hold_writers_take_the_admission_lock_and_state_lives_outside_container_mounts(tmp_path):
    import time

    folder = str(tmp_path)
    p = lh.paths()
    assert p["dir"] == "/home/compu/kaist/sunho/control/launch-hold" and "/runtime/" not in p["dir"]
    assert p["lock"] == "/home/compu/kaist/sunho/runtime/launch/admission.lock"
    # a hold written while a launcher holds the lock lands only after the launcher released it
    lock = tmp_path / "admission.lock"
    holder = subprocess.Popen(["flock", str(lock), "sh", "-c", "sleep 1.5"])
    time.sleep(0.3)
    started = time.time()
    body = f"printf x > {tmp_path}/HOLD"
    result = subprocess.run(["sh", "-c", lh.locked(body, folder=folder)], capture_output=True, text=True, timeout=60)
    holder.wait()
    assert result.returncode == 0 and (tmp_path / "HOLD").exists() and time.time() - started >= 1.0


def test_process_record_without_gpu_instance_refuses(tmp_path):
    import sys

    from ops.containers import launch
    from ops.resources import mig_inventory as inv

    query = ("GPU 00000000:41:00.0\n    MIG Mode\n        Current                 : Enabled\n"
             "        Pending                 : Enabled\n    MIG Device\n        Index                   : 0\n"
             "        GPU Instance ID         : 3\n        FB Memory Usage\n            Total               : 24192 MiB\n"
             "            Used                : 67 MiB\n    GPU UUID                : GPU-4391fcee-537f-d408-d138-10d8a3866eea\n"
             "    Processes\n        Process ID              : 4242\n            Used GPU Memory     : 100 MiB\n")
    with pytest.raises(inv.InventoryError, match="without its GPU instance"):
        inv.parse_query(query)
    listing = ("GPU 0: X (UUID: GPU-4391fcee-537f-d408-d138-10d8a3866eea)\n"
               "  MIG 1g.24gb     Device  0: (UUID: MIG-00000000-0000-5000-8000-000000000000)\n")
    result = subprocess.run([sys.executable, "-I", "-c", launch.MIG_PROBE, "3", "0",
                             "MIG-00000000-0000-5000-8000-000000000000", "GPU-4391fcee-537f-d408-d138-10d8a3866eea"],
                            input=listing + "\n__S4D_QUERY__\n" + query, capture_output=True, text=True)
    assert result.returncode == 3
    pending = query.replace("Pending                 : Enabled", "Pending                 : Disabled").replace(
        "    Processes\n        Process ID              : 4242\n            Used GPU Memory     : 100 MiB\n",
        "    Processes               : None\n")
    result = subprocess.run([sys.executable, "-I", "-c", launch.MIG_PROBE, "3", "0",
                             "MIG-00000000-0000-5000-8000-000000000000", "GPU-4391fcee-537f-d408-d138-10d8a3866eea"],
                            input=listing + "\n__S4D_QUERY__\n" + pending, capture_output=True, text=True)
    assert result.returncode == 7  # the parent's MIG mode is re-checked inside the admission lock


@pytest.mark.parametrize("payload", [
    '{"boot_id": "' + BOOT + '"',
    json.dumps({"boot_id": BOOT}),
    json.dumps({"schema": "wrong", "boot_id": BOOT, "authorization": "ok"}),
    *[json.dumps({"schema": lh.SCHEMA, "boot_id": BOOT, "authorization": value})
      for value in (None, "", "  ", 1, [], {})],
    "null", "[]",
])
def test_invalid_resume_is_held_in_python_shell_and_snapshot(tmp_path, payload):
    path = tmp_path / f"resume-{BOOT}.json"
    path.write_text(payload)
    assert not lh.resume_valid(str(path), BOOT)
    assert run(lh.check_shell(folder=str(tmp_path))).returncode == lh.EXIT_HELD
    state = run(lh.status_shell(folder=str(tmp_path)))
    assert lh.parse_status(state.stdout)["held"] is True


def test_compact_valid_resume_is_accepted_consistently(tmp_path):
    path = tmp_path / f"resume-{BOOT}.json"
    path.write_text(json.dumps(lh.resume_record(BOOT, "approved"), separators=(",", ":")))
    assert lh.resume_valid(str(path), BOOT)
    assert run(lh.check_shell(folder=str(tmp_path))).returncode == 0
    assert not lh.parse_status(run(lh.status_shell(folder=str(tmp_path))).stdout)["held"]
