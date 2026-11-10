"""Local side of the remote dataset downloader: deployment, control-channel latency probe, remote commands.

Everything goes through ``ops.remote.access.RemoteClient`` (credentials are never printed or put in arguments).
Deployment sends a ``git archive`` of the needed paths of one commit (tens of kilobytes) inside the SSH command and
unpacks it, checksum-verified and read-only, to ``/data_8T/kaist/sunho/tools/<sha>/``; the remote side then runs with
the host's ``python3``. The latency probe times a fresh SSH connection plus ``true`` and a command over the shared
control connection, the two things the campaign's own control traffic does.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS_ROOT = "/data_8T/kaist/sunho/tools"
DATA_ROOT = "/data_8T/kaist/sunho/datasets"
DEPLOY_PATHS = (
    "ops/__init__.py", "ops/datasets", "scripts/datasets/droid_datasets.py", "s4d/__init__.py", "s4d/data/__init__.py",
    "s4d/data/droid/__init__.py", "s4d/data/droid/largescale/__init__.py", "s4d/data/droid/largescale/extract.py",
)
MAX_PAYLOAD = 100_000  # one SSH command argument stays far below the kernel's 128 KiB limit


def client():
    from ops.remote.access import RemoteClient

    return RemoteClient()


def bundle(sha: str, repo: Path = REPO) -> tuple[bytes, str]:
    """(gzip tar of DEPLOY_PATHS at ``sha``, sha256sum manifest of its regular files)."""
    raw = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha, *DEPLOY_PATHS], check=True,
                         capture_output=True).stdout
    lines = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                lines.append(f"{digest}  {member.name}")
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as packed, tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.name.endswith(".pyc") or "__pycache__" in member.name:
                continue
            packed.addfile(member, archive.extractfile(member) if member.isfile() else None)
    return out.getvalue(), "\n".join(sorted(lines)) + "\n"


def deploy_command(sha: str, payload: bytes, manifest: str) -> str:
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("deploy needs a full commit sha")
    encoded = base64.b64encode(payload).decode()
    if len(encoded) > MAX_PAYLOAD:
        raise ValueError("deployment bundle too large for one SSH command")
    dest = f"{TOOLS_ROOT}/{sha}"
    return (
        "set -eu; umask 022; mkdir -p " + shlex.quote(TOOLS_ROOT) + "; dest=" + shlex.quote(dest) + "; "
        'if [ -d "$dest" ]; then cd "$dest" && sha256sum -c --quiet MANIFEST.sha256 && echo deployed-existing; '
        "exit 0; fi; "
        'tmp=$(mktemp -d "' + TOOLS_ROOT + '/.deploy-XXXXXX"); trap \'rm -rf "$tmp"\' EXIT; '
        "printf %s " + shlex.quote(encoded) + ' | base64 -d | tar -xz -C "$tmp"; '
        "printf %s " + shlex.quote(manifest) + ' > "$tmp/MANIFEST.sha256"; '
        "printf %s " + shlex.quote(sha) + ' > "$tmp/DEPLOYED_SHA"; '
        '(cd "$tmp" && sha256sum -c --quiet MANIFEST.sha256); chmod -R a-w "$tmp"; chmod u+w "$tmp"; '
        'mv -T "$tmp" "$dest"; chmod u-w "$dest"; echo deployed-new'
    )


def deploy(sha: str) -> str:
    payload, manifest = bundle(sha)
    return client().checked_ssh(deploy_command(sha, payload, manifest), wait=False, idempotent=True,
                                timeout=120).stdout.strip()


def remote_command(sha: str, args: list[str], root: str = DATA_ROOT) -> list[str]:
    script = f"{TOOLS_ROOT}/{sha}/scripts/datasets/droid_datasets.py"
    if args and "--root" not in args:
        args = [args[0], "--root", root, *args[1:]]
    return ["python3", "-I", script, *args]


def remote(sha: str, args: list[str], timeout: float = 900) -> subprocess.CompletedProcess:
    return client().ssh(shlex.join(remote_command(sha, args)), wait=False, idempotent=False, timeout=timeout)


def fresh_ssh_args(args: list[str]) -> list[str]:
    """RemoteClient's ssh arguments without connection sharing (a cold connection each time)."""
    out, skip = [], False
    for index, arg in enumerate(args):
        if skip:
            skip = False
            continue
        if arg == "-o" and index + 1 < len(args) and args[index + 1].split("=")[0] in (
                "ControlMaster", "ControlPersist", "ControlPath"):
            skip = True
            continue
        out.append(arg)
    return [*out, "-o", "ControlMaster=no", "-o", "ControlPath=none"]


def latency(samples: int = 8, pause: float = 0.5) -> dict:
    remote_client = client()
    fresh = fresh_ssh_args(remote_client.ssh_args())
    connect, command = [], []
    for _ in range(samples):
        start = time.monotonic()
        subprocess.run([*fresh, remote_client.target, "true"], env=remote_client._environment(), capture_output=True,
                       timeout=60)
        connect.append(time.monotonic() - start)
        start = time.monotonic()
        remote_client.checked_ssh("true", wait=False, idempotent=True, timeout=60)
        command.append(time.monotonic() - start)
        time.sleep(pause)

    def summary(values):
        return {"median_s": round(statistics.median(values), 4), "max_s": round(max(values), 4), "n": len(values)}

    return {"time": time.time(), "fresh_connect_true": summary(connect), "mux_command_rtt": summary(command)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="droid_datasets_remote.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("deploy")
    p.add_argument("--sha", required=True)
    p = sub.add_parser("latency")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--label", default="")
    p.add_argument("--log", default=None, help="append the JSON result to this file")
    p = sub.add_parser("remote")
    p.add_argument("--sha", required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("args", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.command == "deploy":
        print(deploy(args.sha))
        return 0
    if args.command == "latency":
        result = {**latency(args.n), "label": args.label}
        print(json.dumps(result))
        if args.log:
            with open(args.log, "a") as out:
                out.write(json.dumps(result) + "\n")
        return 0
    forwarded = [a for a in args.args if a != "--"]
    result = remote(args.sha, forwarded, timeout=args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
