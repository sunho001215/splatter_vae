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
