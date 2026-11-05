"""The real-world inference server: container checks, RAM-only weight staging, persistence audit, remote runner.

One persistent container (``configs/realworld/inference_server.yaml``) on the inference host, reached through the
SSH ControlMaster alias in ``ssh`` (authenticated once interactively by the operator; no credential is stored or
passed here). Everything project-controlled on that host lives under ``host_root`` (``~/shkim``): environments,
upstream checkouts, the repository checkout and robot assets. Episode data never touches its disk:

* ``verify_container`` checks the container is the expected one: read-only root filesystem, the RAM tmpfs, exactly the
  configured GPU UUID (CDI device request + ``nvidia-smi -L`` inside), memory/CPU limits and bind mounts below
  ``host_root`` only;
* ``verify_source`` pins the server checkout to the local commit (clean worktree);
* ``stage_weights`` places model weights in the container's RAM tmpfs (uploaded from local storage, or for public
  checkpoints downloaded straight into RAM at a pinned revision), each sha256-checked; nothing is cached on disk;
* ``audit_start``/``audit_end`` record that no file below ``host_root`` changed during a run and that the container's
  writable layer is empty (``docker diff``), plus RAM usage.

``RemoteRunner`` adapts ``stream_client.RemoteWorkerPool`` to the stage-runner interface of ``s4d.realworld.runners``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Server:
    config: dict

    @classmethod
    def load(cls, path: Path) -> Server:
        return cls(yaml.safe_load(Path(path).read_text()))

    @property
    def container(self) -> str:
        return self.config["container"]

    def ssh(self, command: str, *, stdin: bytes | None = None, stdin_file=None, timeout: float = 600,
            check: bool = True):
        """``stdin_file`` streams an open file (bounded memory); ``stdin`` passes small bytes."""
        if stdin_file is not None:
            proc = subprocess.run([*self.config["ssh"], command], stdin=stdin_file, capture_output=True,
                                  timeout=timeout)
        else:
            proc = subprocess.run([*self.config["ssh"], command], input=stdin, capture_output=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise RuntimeError(f"remote command failed ({proc.returncode}): {command}\n{proc.stderr.decode()[-2000:]}")
        return proc

    def dexec(self, args: list[str], *, stdin: bytes | None = None, stdin_file=None, timeout: float = 600,
              check: bool = True):
        cmd = " ".join(shlex.quote(a) for a in ["docker", "exec", "-i", "-w", "/ram", self.container, *args])
        return self.ssh(cmd, stdin=stdin, stdin_file=stdin_file, timeout=timeout, check=check)

    # ---------------------------------------------------------------- checks

    def verify_container(self) -> dict:
        info = json.loads(self.ssh(f"docker inspect {shlex.quote(self.container)}").stdout)[0]
        hc, cfg = info["HostConfig"], self.config
        uuid = cfg["gpu_uuid"]
        devices = [d for r in (hc.get("DeviceRequests") or []) for d in (r.get("DeviceIDs") or [])]
        problems = []
        if not info["State"]["Running"]:
            problems.append("container not running")
        if not hc.get("ReadonlyRootfs"):
            problems.append("root filesystem is writable")
        if devices != [f"nvidia.com/gpu={uuid}"]:
            problems.append(f"GPU device requests {devices} != [{uuid}]")
        if "/ram" not in (hc.get("Tmpfs") or {}):
            problems.append("no /ram tmpfs")
        if not hc.get("Memory") or hc.get("MemorySwap") != hc.get("Memory"):  # equal limits = swap disabled
            problems.append(f"container may swap (Memory {hc.get('Memory')}, MemorySwap {hc.get('MemorySwap')}): "
                            "RAM-only episode data could reach the host swap")
        host_root = cfg["host_root"].rstrip("/") + "/"
        for m in info.get("Mounts", []):
            if m["Type"] == "bind" and not m["Source"].startswith(host_root):
                problems.append(f"bind mount outside {host_root}: {m['Source']}")
        visible = self.dexec(["nvidia-smi", "-L"]).stdout.decode().strip().splitlines()
        if len(visible) != 1 or uuid not in visible[0]:
            problems.append(f"GPUs visible in the container: {visible}")
        probe = self.dexec(["sh", "-c", "touch /.rw_probe 2>/dev/null && echo writable || echo readonly"]).stdout
        if b"readonly" not in probe:
            problems.append("root filesystem accepted a write")
        doc = {"container": self.container, "image": info["Config"]["Image"], "id": info["Id"][:12],
               "readonly_rootfs": hc.get("ReadonlyRootfs"), "gpu_device_requests": devices, "nvidia_smi": visible,
               "memory_limit_bytes": hc.get("Memory"), "memory_swap_limit_bytes": hc.get("MemorySwap"),
               "nano_cpus": hc.get("NanoCpus"), "tmpfs": hc.get("Tmpfs"),
               "mounts": [{"source": m["Source"], "destination": m["Destination"], "rw": m["RW"]}
                          for m in info.get("Mounts", [])], "problems": problems}
        if problems:
            raise RuntimeError(f"inference container check failed: {problems}")
        return doc

    def verify_source(self, expected_commit: str) -> dict:
        repo = shlex.quote(self.config["host_source"])
        head = self.ssh(f"git -C {repo} rev-parse HEAD").stdout.decode().strip()
        dirty = self.ssh(f"git -C {repo} status --porcelain --untracked-files=no").stdout.decode().strip()
        if head != expected_commit or dirty:
            raise RuntimeError(f"server checkout {head} (dirty: {bool(dirty)}) != local {expected_commit}")
        return {"host_source": self.config["host_source"], "commit": head, "clean": True}

    def sync_source(self, local_repo: Path, branch: str) -> None:
        """Push ``branch`` to the server's bare repository and fast-forward its checkout (code only)."""
        subprocess.run(["git", "-C", str(local_repo), "push", "-q", self.config["git_remote"], branch], check=True,
                       timeout=300)
        repo = shlex.quote(self.config["host_source"])
        self.ssh(f"git -C {repo} fetch -q origin {shlex.quote(branch)} && git -C {repo} checkout -q {shlex.quote(branch)}"
                 f" && git -C {repo} merge -q --ff-only origin/{shlex.quote(branch)}")

    # ---------------------------------------------------------------- RAM-only weights

    def _remote_sha(self, path: str) -> str | None:
        code = ("import hashlib,sys,os\np=sys.argv[1]\nif not os.path.isfile(p): print('missing'); raise SystemExit\n"
                "h=hashlib.sha256()\nf=open(p,'rb')\nfor b in iter(lambda: f.read(1<<24), b''): h.update(b)\n"
                "print(h.hexdigest())")
        out = self.dexec([self.config["python_for_tools"], "-I", "-B", "-c", code, path],
                         timeout=900).stdout.decode().strip()
        return None if out == "missing" else out

    def stage_weights(self, fetch: bool = True, workers=None) -> dict:
        """Every configured weight file present in /ram with its pinned sha256 (upload or RAM download if not;
        ``fetch=False`` only verifies and refuses a missing or different file). ``workers``: stage only the entries
        of these workers (entries without ``worker`` always); ``None`` stages every entry."""
        report = {}
        self.dexec(["sh", "-c", "mkdir -p /ram/weights /ram/work /ram/tmp /ram/torch/hub /ram/s4d-cache"])
        for item in self.config["weights"]:
            dest, want = item["dest"], item["sha256"]
            if workers is not None and item.get("worker", None) not in (None, *workers):
                report[dest] = {"skipped": f"worker {item['worker']} is not used by this run"}
                continue
            assert dest.startswith("/ram/"), dest
            t0 = time.time()
            have = self._remote_sha(dest)
            action = "present"
            if have != want and not fetch:
                raise RuntimeError(f"{dest}: sha256 {have} != pinned {want}; stage weights (run without --skip-weights)")
            if have != want:
                self.dexec(["mkdir", "-p", str(Path(dest).parent)])
                if "local" in item:
                    with open(item["local"], "rb") as f:  # streamed through ssh into the tmpfs, never a disk file
                        self.dexec(["sh", "-c", f"cat > {shlex.quote(dest)}.part && mv {shlex.quote(dest)}.part "
                                                f"{shlex.quote(dest)}"], stdin_file=f, timeout=3600)
                    action = "uploaded"
                else:
                    code = ("import sys\nfrom huggingface_hub import hf_hub_download\n"
                            "hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[4])")
                    self.ssh(" ".join(shlex.quote(a) for a in [
                        "docker", "exec", "-i", "-w", "/ram", "-e", "HF_HUB_OFFLINE=0", self.container,
                        item["python"], "-I", "-B", "-c", code, item["hf_repo"], item["hf_file"], item["hf_revision"],
                        str(Path(dest).parent)]), timeout=3600)
                    action = "downloaded_to_ram"
                have = self._remote_sha(dest)
            if have != want:
                raise RuntimeError(f"{dest}: sha256 {have} != pinned {want}")
            report[dest] = {"sha256": have, "action": action, "seconds": round(time.time() - t0, 1)}
        return report

    # ---------------------------------------------------------------- persistence audit

    def _docker_diff(self) -> list[str]:
        out = self.ssh(f"docker diff {shlex.quote(self.container)}").stdout.decode().split("\n")
        return sorted(d for d in out if d.strip())

    def audit_start(self) -> dict:
        """Start time plus the container's current ``docker diff`` (entries created by the NVIDIA CDI hook when the
        container started: ld.so config, Vulkan/EGL/OpenCL ICDs, the hook socket). Any entry added later fails."""
        stamp = self.ssh("date -u +%Y-%m-%dT%H:%M:%S.%NZ").stdout.decode().strip()
        return {"started_utc": stamp, "docker_diff_baseline": self._docker_diff()}

    def audit_end(self, start: dict) -> dict:
        root = shlex.quote(self.config["host_root"])
        changed = self.ssh(f"find {root} -xdev -newermt {shlex.quote(start['started_utc'])} -type f "
                           "-not -path '*/src/splatter4d.git/*' 2>/dev/null | head -200").stdout.decode().split()
        diff = self._docker_diff()
        new = sorted(set(diff) - set(start.get("docker_diff_baseline", [])))
        ram = self.dexec(["sh", "-c", "du -sh /ram/* 2>/dev/null; df -h /ram /tmp | tail -2"]).stdout.decode()
        return {"files_changed_below_host_root": changed, "docker_diff_new_entries": new,
                "docker_diff_baseline_entries": len(start.get("docker_diff_baseline", [])),
                "ram_usage": ram.strip().splitlines(), "ok": not changed and not new}

    def resources(self) -> dict:
        stats = self.ssh(f"docker stats --no-stream --format '{{{{json .}}}}' {shlex.quote(self.container)}").stdout
        gpu = self.ssh("nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader "
                       f"--id={shlex.quote(self.config['gpu_uuid'])}").stdout.decode().strip()
        return {"docker_stats": json.loads(stats), "gpu": gpu}


class RemoteRunner:
    """``s4d.realworld.runners`` interface over a ``RemoteWorkerPool`` (kernels run in the ``s4d_kernels`` worker)."""

    kind = "remote_stream"

    def __init__(self, pool, kernel_spec, *, describe: dict | None = None):
        self.pool, self.kernel_spec, self._describe = pool, kernel_spec, dict(describe or {})
        self.calls: list[dict] = []

    def kernel(self, task: str, params: dict, arrays: dict, *, chunk_id: str):
        response, outputs = self.pool.call_worker(self.kernel_spec, task, params, arrays, chunk_id=chunk_id)
        self._log(self.kernel_spec.name, task, chunk_id, response)
        return outputs, response["meta"], (self.kernel_spec.name, response["transport"]["seq"])

    def worker(self, spec, task: str, params: dict, arrays: dict, *, chunk_id: str, allow_mock: bool = False):
        response, outputs = self.pool.call_worker(spec, task, params, arrays, allow_mock=allow_mock, chunk_id=chunk_id)
        self._log(spec.name, task, chunk_id, response)
        return response, outputs, (spec.name, response["transport"]["seq"])

    def ack(self, handle) -> None:
        if handle is not None:
            self.pool.ack(*handle)

    def _log(self, name, task, chunk_id, response):
        res = (response.get("meta") or {}).get("resources") or {}
        self.calls.append({"worker": name, "task": task, "chunk_id": chunk_id,
                           "elapsed_s": response["transport"].get("elapsed_s"), "resources": res})

    def describe(self) -> dict:
        return {"kind": self.kind, **self._describe}
