from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REMOTE_HOST = "192.168.10.22"
REMOTE_USER = "compu"
REMOTE_ROOT = "/home/compu/kaist/sunho"
DATA_ROOT = Path("/home/ws/data/metaworld/splatter4d_v1")
CREDENTIAL_KEYS = {"SSH_HOST", "SSH_USER", "SSH_PASSWORD", "REMOTE_ROOT"}


class RemoteUnreachable(RuntimeError):
    exit_code = 75


class RemoteCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class Credentials:
    host: str
    user: str
    password: str = field(repr=False)
    root: str = REMOTE_ROOT


def read_credentials(path: Path) -> Credentials:
    if path.stat().st_mode & 0o077:
        raise ValueError("Remote credentials must not be accessible to other users")
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        if not sep or key not in CREDENTIAL_KEYS or key in values:
            raise ValueError("Unsupported remote credential file format")
        try:
            tokens = shlex.split(value, comments=False)
        except ValueError:
            raise ValueError("Unsupported remote credential file format") from None
        if len(tokens) != 1 or not tokens[0] or "\x00" in tokens[0] or "\n" in tokens[0]:
            raise ValueError("Unsupported remote credential file format")
        values[key] = tokens[0]
    if set(values) != CREDENTIAL_KEYS:
        raise ValueError("Remote credential file is missing required fields")
    if values["SSH_HOST"] != REMOTE_HOST or values["SSH_USER"] != REMOTE_USER:
        raise ValueError("Remote credential target is not authorized")
    root = values["REMOTE_ROOT"].rstrip("/")
    if root not in (REMOTE_ROOT, "~/kaist/sunho"):
        raise ValueError("Remote root is not authorized")
    return Credentials(values["SSH_HOST"], values["SSH_USER"], values["SSH_PASSWORD"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main_repository(repo: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True,
    )
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = repo / common
    return common.resolve().parent


class RemoteClient:
    def __init__(self, credentials_path: Path | None = None, repo: Path | None = None, state_dir: Path | None = None):
        self.repo = Path(repo or Path(__file__).resolve().parents[1]).resolve()
        credential_path = credentials_path or Path(
            os.environ.get("S4D_REMOTE_CREDENTIALS", str(Path.home() / ".config/s4d_remote/credentials"))
        )
        self.credentials_path = Path(credential_path).resolve()
        self.credentials = read_credentials(self.credentials_path)
        self.root = self.credentials.root
        self.target = f"{self.credentials.user}@{self.credentials.host}"
        self.main_repo = main_repository(self.repo)
        override = state_dir or os.environ.get("S4D_REMOTE_STATE_DIR")
        self.state_dir = self._local_path(Path(override) if override else self.main_repo / "runs/remote/access")
        if DATA_ROOT == self.state_dir or DATA_ROOT in self.state_dir.parents:
            raise ValueError("Remote access state must be inside the repository")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.network_dir = self.state_dir.parent

    def __repr__(self) -> str:
        return f"RemoteClient(target={self.target!r}, root={self.root!r})"

    def _redact(self, text: str | None) -> str:
        return (text or "").replace(self.credentials.password, "[REDACTED]")

    def _environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_KEYS | {"SSHPASS"}}
        env["SSHPASS"] = self.credentials.password
        env["LC_ALL"] = "C"
        return env

    def ssh_args(self) -> list[str]:
        return [
            "sshpass", "-e", "ssh", "-F", "/dev/null", "-T", "-o", "ConnectTimeout=15",
            "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "HashKnownHosts=yes",
            "-o", f"UserKnownHostsFile={self.state_dir / 'known_hosts'}",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
            "-o", f"ControlPath={self.state_dir / 'cm-%C'}",
        ]

    def _network_event(self, reachable: bool, reason: str = "") -> None:
        state_file = self.network_dir / "network.json"
        event_file = self.network_dir / "network.jsonl"
        with (self.network_dir / "network.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(state_file.read_text()) if state_file.exists() else {"reachable": True}
            now = time.time()
            event = None
            if not reachable and state.get("reachable", True):
                state = {"reachable": False, "outage_started": now, "reason": reason}
                event = {"event": "REMOTE_UNREACHABLE", "time": now, "reason": reason}
            elif reachable and not state.get("reachable", True):
                start = float(state["outage_started"])
                event = {"event": "REMOTE_RECONNECTED", "time": now, "outage_started": start,
                         "duration_seconds": now - start}
                state = {"reachable": True, "last_reconnected": now}
            state["checked_at"] = now
            temp = state_file.with_suffix(".tmp")
            temp.write_text(json.dumps(state, sort_keys=True))
            os.replace(temp, state_file)
            if event:
                with event_file.open("a") as out:
                    out.write(json.dumps(event, sort_keys=True) + "\n")

    def check_tunnel(self, wait: bool = True, max_wait: float = 600.0) -> bool:
        start = time.monotonic()
        delay = 5.0
        while True:
            remaining = max_wait - (time.monotonic() - start)
            if wait and remaining <= 0:
                raise RemoteUnreachable("Remote SSH endpoint unreachable for ten minutes; host-side VPN may need attention")
            try:
                with socket.create_connection((self.credentials.host, 22), timeout=min(3.0, max(0.1, remaining))):
                    pass
                self._network_event(True)
                return True
            except OSError as exc:
                self._network_event(False, type(exc).__name__)
            if not wait or remaining <= 0:
                raise RemoteUnreachable("Remote SSH endpoint unreachable; remote jobs remain unknown, not failed")
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 60.0)

    def ssh(
        self, command: str, wait: bool = True, idempotent: bool = False, timeout: float = 120,
    ) -> subprocess.CompletedProcess:
        if "\x00" in command or self.credentials.password in command:
            raise ValueError("Unsafe remote command or credential in command arguments")
        attempts = 3 if idempotent else 1
        for attempt in range(attempts):
            self.check_tunnel(wait=wait)
            args = [*self.ssh_args(), self.target, command]
            try:
                result = subprocess.run(args, capture_output=True, text=True, env=self._environment(), timeout=timeout)
            except subprocess.TimeoutExpired:
                self._network_event(False, "SSHTimeout")
                if idempotent and attempt + 1 < attempts:
                    continue
                raise RemoteUnreachable("SSH timed out; command outcome is unknown and must be reconciled") from None
            result.stdout, result.stderr = self._redact(result.stdout), self._redact(result.stderr)
            if result.returncode == 255:
                self._network_event(False, "SSHTransportError")
                if idempotent and attempt + 1 < attempts:
                    continue
                raise RemoteUnreachable("SSH transport failed; remote jobs remain unknown, not failed")
            self._network_event(True)
            return result
        raise AssertionError("unreachable")

    def checked_ssh(self, command: str, **kwargs) -> subprocess.CompletedProcess:
        result = self.ssh(command, **kwargs)
        if result.returncode:
            raise RemoteCommandError(f"Remote operation exited {result.returncode}: {self._redact(result.stderr)[-1000:]}")
        return result

    def remote_path(self, path: str | Path, *, allow_root: bool = False) -> str:
        raw = str(path)
        if raw.startswith("~/"):
            raw = "/home/compu/" + raw[2:]
        posix = PurePosixPath(raw)
        if not posix.is_absolute():
            posix = PurePosixPath(self.root) / posix
        if ".." in posix.parts or "\x00" in raw or "\n" in raw or "\r" in raw:
            raise ValueError("Unsafe remote path")
        root = PurePosixPath(self.root)
        if posix == root and allow_root:
            return str(posix)
        if root not in posix.parents:
            raise ValueError("Remote path is outside the authorized root")
        return str(posix)

    def ensure_remote_path(
        self, path: str | Path, *, create_parent: bool = False, directory: bool = False, wait: bool = True,
    ) -> str:
        target = self.remote_path(path, allow_root=True)
        suffix = str(PurePosixPath(target).relative_to(self.root))
        parts = [] if suffix == "." else PurePosixPath(suffix).parts
        component_args = " ".join(shlex.quote(p) for p in parts)
        script = (
            "set -eu; root=" + shlex.quote(self.root) + "; "
            '[ ! -L "$root" ]; [ "$(realpath -m -- "$root")" = "$root" ]; '
            'current="$root"; for component in ' + component_args + '; do '
            'current="$current/$component"; [ ! -L "$current" ]; done; '
            '[ "$(realpath -m -- "$current")" = "$current" ]; '
        )
        if create_parent:
            parent = target if directory else str(PurePosixPath(target).parent)
            script += "mkdir -p -- " + shlex.quote(parent) + "; "
        self.checked_ssh(script, idempotent=True, wait=wait)
        return target

    def _local_path(self, path: Path) -> Path:
        absolute = Path(os.path.abspath(path))
        resolved = absolute.resolve()
        if absolute != resolved:
            raise ValueError("Local transfer path must not traverse symlinks")
        roots = (self.repo, self.main_repo, DATA_ROOT)
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise ValueError("Local transfer path is outside authorized repository/data roots")
        return resolved

    def _rsync(self, source: str, target: str, extra: list[str] | None = None, wait: bool = True) -> None:
        args = ["rsync", "-rt", "--partial", "--append-verify", "--checksum", "--protect-args",
                "-e", shlex.join(self.ssh_args()), *(extra or []), "--", source, target]
        if any(self.credentials.password in arg for arg in args):
            raise ValueError("Credential must not appear in transfer arguments")
        for attempt in range(3):
            self.check_tunnel(wait=wait)
            try:
                result = subprocess.run(args, capture_output=True, text=True, env=self._environment(), timeout=3600)
            except subprocess.TimeoutExpired:
                self._network_event(False, "TransferTimeout")
                if attempt < 2:
                    continue
                raise RemoteUnreachable("Transfer timed out; no remote job state changed") from None
            if result.returncode == 0:
                self._network_event(True)
                return
            if result.returncode in (10, 12, 30, 35, 255):
                self._network_event(False, "TransferTransportError")
                if attempt < 2:
                    continue
                raise RemoteUnreachable("Transfer interrupted; resumable partial files retained")
            raise RemoteCommandError(f"Transfer exited {result.returncode}: {self._redact(result.stderr)[-1000:]}")

    def _remote_checksum(self, remote: str, wait: bool = True, allow_missing: bool = False) -> str | None:
        command = "sha256sum -- " + shlex.quote(remote)
        if allow_missing:
            command = "if [ -f " + shlex.quote(remote) + " ]; then " + command + "; fi"
        result = self.checked_ssh(command, wait=wait, idempotent=True).stdout
        if allow_missing and not result:
            return None
        if not re.match(r"^[0-9a-f]{64}  ", result):
            raise RemoteCommandError("Invalid remote checksum response")
        return result[:64]

    def rsync_upload(self, source: Path, target: str, wait: bool = True) -> None:
        local = self._local_path(source)
        if not local.is_file() or local == self.credentials_path:
            raise ValueError("Upload requires an explicit non-credential regular file")
        remote = self.ensure_remote_path(self.remote_path(target), create_parent=True, wait=wait)
        before = sha256(local)
        if self._remote_checksum(remote, wait=wait, allow_missing=True) == before:
            if sha256(local) != before:
                raise RemoteCommandError("Source changed during checksum verification")
            return
        partial = self.ensure_remote_path(remote + ".s4d-partial-" + before, wait=wait)
        self._rsync(str(local), f"{self.target}:{partial}", wait=wait)
        if self._remote_checksum(partial, wait=wait) != before or sha256(local) != before:
            raise RemoteCommandError("Upload checksum differs or source changed during transfer")
        self.ensure_remote_path(remote, wait=wait)
        self.checked_ssh("mv -T -- " + shlex.quote(partial) + " " + shlex.quote(remote),
                         wait=wait, idempotent=False)
        if self._remote_checksum(remote, wait=wait) != before:
            raise RemoteCommandError("Published upload checksum differs")

    def rsync_download(self, source: str, target: Path, wait: bool = True) -> None:
        remote = self.ensure_remote_path(self.remote_path(source), wait=wait)
        local = self._local_path(target)
        local.parent.mkdir(parents=True, exist_ok=True)
        before = self._remote_checksum(remote, wait=wait)
        if local.is_file() and sha256(local) == before:
            if self._remote_checksum(remote, wait=wait) != before:
                raise RemoteCommandError("Source changed during checksum verification")
            return
        partial = self._local_path(local.with_name(local.name + ".s4d-partial-" + before))
        self._rsync(f"{self.target}:{remote}", str(partial), wait=wait)
        after = self._remote_checksum(remote, wait=wait)
        if before != after or sha256(partial) != after:
            raise RemoteCommandError("Downloaded file changed or failed checksum verification")
        self._local_path(local)
        os.replace(partial, local)

    def _git(self, *args: str, timeout: float = 90) -> str:
        env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_KEYS | {"SSHPASS"}}
        result = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, env=env,
                                timeout=timeout)
        if result.returncode:
            raise RemoteCommandError("Local Git provenance check failed: " + self._redact(result.stderr)[-1000:])
        return result.stdout.strip()

    def sync_code(self, commit: str) -> str:
        with (self.state_dir / "code.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._sync_code(commit)

    def _sync_code(self, commit: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            raise ValueError("A concrete hexadecimal commit is required")
        runtime_dirs = ("runs/", ".cache/", "outputs/")
        runtime_files = {"experiments/registry.jsonl", "experiments/daemon.log", "experiments/daemon.pid",
                         "experiments/jobs.pid", "experiments/HOLD"}
        status = self._git("status", "--porcelain", "-z", "--untracked-files=normal")
        for entry in status.split("\x00"):
            if not entry:
                continue
            path = entry[3:]
            if not entry.startswith("?? ") or not (path.startswith(runtime_dirs) or path in runtime_files):
                raise RemoteCommandError("Refusing to release code with dirty sources")
        sha = self._git("rev-parse", "--verify", "--end-of-options", commit + "^{commit}")
        origin = self._git("remote", "get-url", "origin")
        if origin not in ("https://github.com/sunho001215/splatter_vae.git", "git@github.com:sunho001215/splatter_vae.git"):
            raise RemoteCommandError("Repository origin is not the authorized campaign repository")
        advertised = self._git("-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
                               "ls-remote", "origin", "refs/heads/splatter4d").split()
        if len(advertised) != 2 or not re.fullmatch(r"[0-9a-f]{40}", advertised[0]):
            raise RemoteCommandError("Cannot verify the pushed splatter4d branch")
        tip = advertised[0]
        self._git("merge-base", "--is-ancestor", sha, tip)
        if self._git("rev-parse", "refs/remotes/origin/splatter4d") != tip:
            raise RemoteCommandError("Local origin/splatter4d is stale; refresh it before release")
        bundle_dir = self.network_dir / "bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundle_dir / f"{tip}.bundle"
        if not bundle.exists():
            temp = bundle.with_suffix(".bundle.partial")
            self._git("bundle", "create", str(temp), "refs/remotes/origin/splatter4d", timeout=300)
            os.replace(temp, bundle)
        self._git("bundle", "verify", str(bundle))
        remote_bundle = f"bundles/{tip}.bundle"
        self.rsync_upload(bundle, remote_bundle)
        code = self.ensure_remote_path(f"releases/{sha}/code", create_parent=True)
        self.ensure_remote_path(code + "/.git")
        bpath = self.remote_path(remote_bundle)
        expected = sha256(bundle)
        git = "git -c core.hooksPath=/dev/null -c " + shlex.quote("core.worktree=" + code) + " -C " + shlex.quote(code)
        script = (
            "set -eu; export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null; "
            "test \"$(sha256sum -- " + shlex.quote(bpath) + " | cut -d' ' -f1)\" = "
            + shlex.quote(expected) + "; "
            + "if [ -e " + shlex.quote(code) + " ]; then test -d " + shlex.quote(code + "/.git")
            + "; test -z \"$(find " + shlex.quote(code + "/.git") + " -type l -print -quit)\"; "
            + "test -z \"$(" + git + " status --porcelain)\"; "
            + "else git -c init.templateDir= init --quiet -- " + shlex.quote(code) + "; fi; "
            + git + " fetch --quiet -- " + shlex.quote(bpath)
            + " refs/remotes/origin/splatter4d:refs/remotes/origin/splatter4d; "
            + git + " remote remove origin 2>/dev/null || :; "
            + git + " remote add origin " + shlex.quote(origin) + "; "
            + git + " update-ref refs/remotes/origin/splatter4d " + shlex.quote(tip) + "; "
            + git + " checkout --quiet --detach " + shlex.quote(sha) + "; "
            + "test \"$(" + git + " rev-parse HEAD)\" = " + shlex.quote(sha) + "; "
            + "test -z \"$(" + git + " status --porcelain)\"; "
        )
        self.checked_ssh(script, idempotent=True, timeout=300)
        for mountpoint in ("runs", "outputs", ".cache"):
            self.ensure_remote_path(code + "/" + mountpoint, create_parent=True, directory=True)
        self.checked_ssh("test -z \"$(" + git + " status --porcelain)\"", idempotent=True)
        proof = {"commit": sha, "verified_origin_commit": tip, "origin_commit": tip,
                 "origin_branch": "splatter4d", "origin_ref": "refs/remotes/origin/splatter4d",
                 "origin_url": origin, "bundle_sha256": expected, "verified_ancestor": True,
                 "verified_at": datetime.now(timezone.utc).isoformat()}
        local_proof = self.network_dir / "releases" / sha / "provenance.json"
        local_proof.parent.mkdir(parents=True, exist_ok=True)
        local_proof.write_text(json.dumps(proof, indent=2, sort_keys=True))
        self.rsync_upload(local_proof, f"releases/{sha}/provenance.json")
        return code

    def sync_data(self, manifest: Path | dict) -> dict:
        payload = json.loads(manifest.read_text()) if isinstance(manifest, Path) else manifest
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list) or not payload["files"]:
            raise ValueError("Data manifest requires a nonempty files list")
        verified = []
        seen = set()
        for entry in payload["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
                raise ValueError("Data manifest files require a source path")
            local = self._local_path(Path(entry["source"]))
            if not local.is_file():
                raise ValueError("Data manifest sources must be explicit regular files")
            if not any(root in local.parents for root in (DATA_ROOT, self.repo, self.main_repo)):
                raise ValueError("Data manifest source is not repository-defined data")
            if "target" in entry:
                target = str(entry["target"])
            elif DATA_ROOT in local.parents:
                target = "data/metaworld/splatter4d_v1/" + str(local.relative_to(DATA_ROOT))
            else:
                raise ValueError("Repository data files require an explicit remote target")
            remote = self.remote_path(target)
            if not remote.startswith(self.root + "/data/") or remote in seen:
                raise ValueError("Data manifest target must be unique and inside remote data/")
            digest = sha256(local)
            if entry.get("sha256") not in (None, digest):
                raise ValueError("Data manifest checksum differs from source")
            seen.add(remote)
            verified.append({"source": str(local), "target": remote, "sha256": digest, "bytes": local.stat().st_size})
        for entry in verified:
            source = Path(entry["source"])
            self.rsync_upload(source, entry["target"])
            if sha256(source) != entry["sha256"]:
                raise RemoteCommandError("Data source changed during transfer")
        result = {"verified_at": datetime.now(timezone.utc).isoformat(), "files": verified}
        normalized = [{"target": str(PurePosixPath(e["target"]).relative_to(self.root)),
                       "sha256": e["sha256"], "bytes": e["bytes"]} for e in verified]
        normalized.sort(key=lambda e: e["target"])
        identity = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
        result["identity_sha256"] = identity
        output = self.network_dir / "data" / f"{identity}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
        return result

    @staticmethod
    def _result_manifest(text: str, excluded: list[str]) -> dict[str, str]:
        entries = {}
        for line in text.splitlines():
            if not re.match(r"^[0-9a-f]{64}  \./", line):
                raise RemoteCommandError("Unsafe remote result checksum manifest")
            digest, relative = line.split("  ./", 1)
            candidate = PurePosixPath(relative)
            if (candidate.is_absolute() or ".." in candidate.parts or not candidate.parts
                    or any(p in excluded for p in candidate.parts) or "\\" in relative
                    or any(ord(c) < 32 for c in relative) or relative in entries):
                raise RemoteCommandError("Unsafe remote result path")
            entries[relative] = digest
        return entries

    def sync_results(self, run_id: str, checkpoints: bool = False, final: bool = False, wait: bool = True) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError("Unsafe run id")
        with (self.state_dir / ("results-" + run_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._sync_results(run_id, checkpoints, final, wait)

    def _sync_results(self, run_id: str, checkpoints: bool, final: bool, wait: bool) -> dict:
        remote = self.ensure_remote_path("runs/" + run_id, wait=wait)
        local = self._local_path(self.main_repo / "runs" / run_id)
        local.mkdir(parents=True, exist_ok=True)
        excluded = ["replay", "wandb", ".cache"] + ([] if checkpoints else ["checkpoints", "snapshots"])
        names = ["exit_code", "*.json", "*.jsonl", "*.yaml", "*.yml", "*.txt", "*.log", "*.csv",
                 "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.pdf", "*.html", "*.npz"]
        if checkpoints:
            names += ["*.pt", "*.pth", "*.ckpt"]
        pruned = " -o ".join("-name " + shlex.quote(name) for name in excluded)
        patterns = " -o ".join("-name " + shlex.quote(name) for name in names)
        patterns += " -o -path './encoder.pt' -o -path './exports/*'"
        manifest_cmd = ("set -eu; cd " + shlex.quote(remote) + "; find . -type d \\( " + pruned
                        + " \\) -prune -o -type f \\( " + patterns + " \\) -exec sha256sum -- {} +")
        before = self._result_manifest(self.checked_ssh(manifest_cmd, idempotent=True, wait=wait).stdout, excluded)
        if final and not before:
            raise RemoteCommandError("Final result snapshot is empty")
        for relative in before:
            self._local_path(local / relative)
        output = self.network_dir / "results" / (run_id + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"run_id": run_id, "host": "remote", "verified": False,
                                      "final": False, "sync_started_at": datetime.now(timezone.utc).isoformat()}))
        for relative, digest in before.items():
            destination = self._local_path(local / relative)
            if not destination.is_file() or sha256(destination) != digest:
                self.rsync_download(remote + "/" + relative, destination, wait=wait)
        after = self._result_manifest(self.checked_ssh(manifest_cmd, idempotent=True, wait=wait).stdout, excluded)
        if final and before != after:
            raise RemoteCommandError("Final results changed during synchronization")
        for relative, digest in after.items():
            path = self._local_path(local / relative)
            if not path.is_file() or sha256(path) != digest:
                raise RemoteCommandError("Result checksum mismatch; results are not decision-ready")
        result = {"run_id": run_id, "host": "remote", "verified": True,
                  "verified_at": datetime.now(timezone.utc).isoformat(), "final": final, "files": after}
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2, sort_keys=True))
        os.replace(temp, output)
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-tunnel")
    check.add_argument("--no-wait", action="store_true")
    ssh = sub.add_parser("ssh")
    ssh.add_argument("--idempotent", action="store_true")
    ssh.add_argument("remote_command", nargs=argparse.REMAINDER)
    code = sub.add_parser("sync-code")
    code.add_argument("commit")
    data = sub.add_parser("sync-data")
    data.add_argument("manifest", type=Path)
    results = sub.add_parser("sync-results")
    results.add_argument("run_id")
    results.add_argument("--checkpoint", "--checkpoints", action="store_true")
    results.add_argument("--final", action="store_true")
    args = parser.parse_args()
    try:
        client = RemoteClient()
        if args.command == "check-tunnel":
            client.check_tunnel(wait=not args.no_wait)
        elif args.command == "ssh":
            values = args.remote_command
            if values and values[0] == "--":
                values = values[1:]
            if not values:
                raise ValueError("An explicit remote command is required")
            command = values[0] if len(values) == 1 else shlex.join(values)
            result = client.ssh(command, idempotent=args.idempotent)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
        elif args.command == "sync-code":
            print(client.sync_code(args.commit))
        elif args.command == "sync-data":
            print(json.dumps(client.sync_data(args.manifest), indent=2))
        else:
            print(json.dumps(client.sync_results(args.run_id, args.checkpoint, args.final), indent=2))
        return 0
    except RemoteUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, RemoteCommandError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
