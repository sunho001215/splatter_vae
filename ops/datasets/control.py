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
import math
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
UPLOAD_STAGING = DATA_ROOT + "/.downloads/adopt"  # on /data_8T, never on the REMOTE system disk
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


def staging_paths(name: str, digest: str) -> tuple[str, str]:
    if "/" in name or name.startswith(".") or len(digest) != 64:
        raise ValueError("plain file name and a sha256 digest expected")
    target = f"{UPLOAD_STAGING}/{name}"
    return target, f"{target}.s4d-partial-{digest}"


def stage_upload(local: Path, kib_per_s: int, floor_fraction: float = 0.085) -> dict:
    """Upload one local file into the adopt staging directory on /data_8T (then ``adopt`` installs it): the staging
    directory must resolve (no symlink) onto the /data_8T mount and the file must fit above the data disk's floor;
    explicit ``--bwlimit`` as the existing upload policy requires; partial name with rsync delta transfer (resumable,
    and a corrupt full-length partial is repaired, not skipped); sha256 verified on the remote host; published with an
    atomic no-replace hard link (an existing target is never overwritten)."""
    from ops.provenance.hashing import sha256_file

    if int(kib_per_s) <= 0:
        raise ValueError("an explicit positive --bwlimit (KiB/s) is required")
    remote_client, local = client(), Path(local)
    digest, start, size = sha256_file(local), time.monotonic(), local.stat().st_size
    target, partial = staging_paths(local.name, digest)
    staging = shlex.quote(UPLOAD_STAGING)

    def remote_sha(path: str) -> str | None:
        out = remote_client.checked_ssh(f"if [ -f {shlex.quote(path)} ]; then sha256sum -- {shlex.quote(path)}; fi",
                                        wait=False, idempotent=True, timeout=600).stdout.split()
        return out[0] if out else None

    check = remote_client.checked_ssh(
        f"mkdir -p {staging} && [ \"$(realpath -e {staging})\" = {staging} ] && "
        f"findmnt -n -o TARGET --target {staging} && df --output=avail,size -B1 {staging} | tail -1",
        wait=False, idempotent=True).stdout.split()
    if len(check) != 3 or check[0] != "/data_8T":
        raise RuntimeError(f"staging directory is not a plain directory on /data_8T: {check}")
    avail, fs_size = int(check[1]), int(check[2])
    if avail - size < math.ceil(floor_fraction * fs_size):
        raise RuntimeError("the upload would cross the data disk's free-space floor")
    existing = remote_sha(target)
    if existing is not None:
        if existing != digest:
            raise RuntimeError(f"{target} exists with another sha256; not overwritten")
        return {"file": local.name, "status": "present", "sha256": digest}
    remote_client._rsync_run(str(local), f"{remote_client.target}:{partial}", [f"--bwlimit={int(kib_per_s)}",
                                                                               "--timeout=300"], True, False, (0,))
    if remote_sha(partial) != digest or sha256_file(local) != digest:
        remote_client.checked_ssh(f"rm -f -- {shlex.quote(partial)}", wait=False, idempotent=True)
        raise RuntimeError(f"{local.name}: uploaded sha256 differs or the source changed (partial removed)")
    remote_client.checked_ssh(f"ln -T -- {shlex.quote(partial)} {shlex.quote(target)} && rm -f -- "
                              f"{shlex.quote(partial)}", wait=False, idempotent=False)
    if remote_sha(target) != digest:
        raise RuntimeError(f"{target}: sha256 differs after publication")
    return {"file": local.name, "status": "uploaded", "sha256": digest, "bytes": size,
            "seconds": round(time.monotonic() - start, 1), "kib_per_s": int(kib_per_s), "remote": target}


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
    p = sub.add_parser("stage-upload")
    p.add_argument("--kib", type=int, required=True, help="rsync --bwlimit in KiB/s (upload policy)")
    p.add_argument("files", nargs="+")
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
    if args.command == "stage-upload":
        for path in args.files:
            print(json.dumps(stage_upload(Path(path), args.kib)))
        return 0
    forwarded = [a for a in args.args if a != "--"]
    result = remote(args.sha, forwarded, timeout=args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode
