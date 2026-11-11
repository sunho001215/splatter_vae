"""Durable remote launch hold tied to the remote boot (user directive 2026-11-11, remote MIG migration; ARCH review).

Every launcher that starts a container on the remote host checks this hold INSIDE the remote admission lock
(``$ROOT/runtime/launch/admission.lock``), immediately before ``docker run``:

* ``$ROOT/runtime/launch/HOLD`` exists                      -> held (an explicit hold; any content);
* no ``$ROOT/runtime/launch/resume-<boot_id>.json`` naming the CURRENT ``/proc/sys/kernel/random/boot_id``
                                                               -> held (fail closed: every reboot re-arms the hold).

A launch is allowed only after an explicit resume record for the running boot, written by ``resume`` with the user's
authorization text. Launchers: ops/containers/launch.py (scheduler campaign rows and typed verification),
the real-world preprocessing guard (realworld-ops/pre/tools/rwprep_guard.py) and manual tools, which embed
``check_shell()`` (exit EXIT_HELD) or call ``assert_launch_allowed()`` before their own ``docker run``. The scheduler
also reads the state in its snapshot (``status_shell``/``parse_status``) and launches nothing while held, so the in-lock
check is the race-free backstop rather than the normal path.

    python -I ops/remote/launch_hold.py status
    python -I ops/remote/launch_hold.py hold --reason TEXT                 (always allowed: the safe direction)
    python -I ops/remote/launch_hold.py release-hold --authorization TEXT  (removes HOLD; the boot record still applies)
    python -I ops/remote/launch_hold.py resume --authorization TEXT        (records the resume for the current boot)
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path

ROOT = "/home/compu/kaist/sunho"
EXIT_HELD = 70
SCHEMA = "s4d-remote-launch-resume/1"
BOOT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


def paths(root: str = ROOT, folder: str | None = None) -> dict:
    """The hold files under the authorized runtime root (``folder``: tests only)."""
    if folder is None:
        if root != ROOT:
            raise ValueError("launch hold root is not authorized")
        folder = f"{root}/runtime/launch"
    return {"dir": folder, "hold": f"{folder}/HOLD", "resume_prefix": f"{folder}/resume-"}


def check_shell(root: str = ROOT, folder: str | None = None) -> str:
    """POSIX shell fragment: exit EXIT_HELD (message on stderr) unless the launch is allowed for the current boot.
    Run it inside the admission lock, after the idempotent existing-container check and before ``docker run``."""
    p, q = paths(root, folder), shlex.quote
    return ('s4d_boot="$(cat /proc/sys/kernel/random/boot_id)"; '
            f'if [ -e {q(p["hold"])} ] || ! grep -Fqs -- "\\"boot_id\\": \\"$s4d_boot\\"" '
            f'{q(p["resume_prefix"])}"$s4d_boot".json; then '
            f'printf \'launch held (runtime/launch/HOLD or no resume record for boot %s)\\n\' "$s4d_boot" >&2; '
            f'exit {EXIT_HELD}; fi')


def status_shell(root: str = ROOT, folder: str | None = None) -> str:
    """Shell fragment printing one line ``__S4D_LAUNCH_HOLD__ <boot_id> <hold 0|1> <resume 0|1>`` (snapshot)."""
    p, q = paths(root, folder), shlex.quote
    return ('s4d_boot="$(cat /proc/sys/kernel/random/boot_id)"; '
            f's4d_hold=0; [ -e {q(p["hold"])} ] && s4d_hold=1; s4d_resume=0; '
            f'grep -Fqs -- "\\"boot_id\\": \\"$s4d_boot\\"" {q(p["resume_prefix"])}"$s4d_boot".json && s4d_resume=1; '
            'printf \'__S4D_LAUNCH_HOLD__ %s %s %s\\n\' "$s4d_boot" "$s4d_hold" "$s4d_resume"; ')


def parse_status(output: str) -> dict:
    """{"boot_id", "hold", "resumed", "held"}; held (True) unless exactly one well-formed line says not held."""
    lines = [line.split() for line in output.splitlines() if line.startswith("__S4D_LAUNCH_HOLD__ ")]
    if len(lines) != 1 or len(lines[0]) != 4 or not BOOT_ID.fullmatch(lines[0][1]) \
            or lines[0][2] not in ("0", "1") or lines[0][3] not in ("0", "1"):
        return {"boot_id": None, "hold": None, "resumed": None, "held": True}
    _, boot, hold, resumed = lines[0]
    return {"boot_id": boot, "hold": hold == "1", "resumed": resumed == "1", "held": hold == "1" or resumed != "1"}


def resume_record(boot_id: str, authorization: str, by: str = "coordinator") -> dict:
    if not BOOT_ID.fullmatch(boot_id):
        raise ValueError("invalid boot id")
    if not authorization.strip():
        raise ValueError("the user's explicit resume authorization must be recorded")
    return {"schema": SCHEMA, "boot_id": boot_id, "authorization": authorization.strip(), "by": by,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _client():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ops.remote.access import RemoteClient  # noqa: PLC0415

    return RemoteClient()


def remote_status(client) -> dict:
    return parse_status(client.checked_ssh(status_shell(client.root), wait=False, idempotent=True, timeout=60).stdout)


def assert_launch_allowed(client) -> dict:
    """For manual launch tools, before their own SSH ``docker run`` (which must ALSO embed check_shell() inside its
    admission lock): raises RuntimeError while held."""
    state = remote_status(client)
    if state["held"]:
        raise RuntimeError(f"remote launch hold in force: {state}")
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "hold", "release-hold", "resume"))
    parser.add_argument("--reason", default="")
    parser.add_argument("--authorization", default="")
    args = parser.parse_args(argv)
    client = _client()
    p, q = paths(client.root), shlex.quote
    if args.command == "status":
        print(json.dumps(remote_status(client)))
        return 0
    if args.command == "hold":
        text = json.dumps({"reason": args.reason or "hold", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        client.checked_ssh(f"mkdir -p {q(p['dir'])} && printf '%s\\n' {q(text)} > {q(p['hold'])}.tmp && "
                           f"mv -f {q(p['hold'])}.tmp {q(p['hold'])}", wait=False, idempotent=True, timeout=60)
    elif args.command == "release-hold":
        if not args.authorization.strip():
            raise SystemExit("release-hold requires --authorization (the user's explicit resume)")
        client.checked_ssh(f"mv -f {q(p['hold'])} {q(p['hold'])}.released-$(date +%Y%m%d-%H%M%S) 2>/dev/null; true",
                           wait=False, idempotent=True, timeout=60)
    else:
        state = remote_status(client)
        record = resume_record(state["boot_id"], args.authorization)
        target = f"{p['resume_prefix']}{record['boot_id']}.json"
        client.checked_ssh(f"mkdir -p {q(p['dir'])} && printf '%s\\n' {q(json.dumps(record, sort_keys=True))} > "
                           f"{q(target)}.tmp && mv -f {q(target)}.tmp {q(target)}", wait=False, idempotent=False,
                           timeout=60)
    print(json.dumps(remote_status(client)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
