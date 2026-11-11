"""Coordinator-side MIG acceptance, without changing MIG, CDI, allowlists or launch holds.

Run with ``python -I ops/remote/mig_validation.py discover --out discovery.json``;
then, after explicit resumption, ``instances --allowlist configs/hosts/remote-mig.yaml
--image sha256:<digest> --out validation.json [--release-code <remote release>]``.
``report validation.json`` summarizes the archived evidence. Only RemoteClient performs
remote operations; importing this module and building/parsing commands have no side effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.remote.launch_hold import BOOT_ID, EXIT_HELD, ROOT, check_shell, parse_status, status_shell
from ops.resources.mig_inventory import REMOTE_PHYSICAL_GPUS, build_inventory, load_allowlist, parse_list, parse_query
from ops.containers.launch import MIG_PROBE

IDLE_FB_MIB = 256

SCHEMA = "s4d-mig-validation/1"
PROBE_PREFIX = "__S4D_MIGVAL__ "
CDI = "/var/run/cdi/nvidia.yaml"
IMAGE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}\Z")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")

# This string is only executed by python INSIDE the selected remote container.
PROBE = r'''
import ctypes as C
import gc, importlib.util, json, os, re, subprocess, sys, time, traceback

prefix = "__S4D_MIGVAL__ "
uuid, parent, other, marker = (os.environ[k] for k in
    ("CUDA_VISIBLE_DEVICES", "S4D_MIGVAL_PARENT", "S4D_MIGVAL_OTHER", "S4D_MIGVAL_MARKER"))
result = {"uuid": uuid, "parent": parent, "marker": marker, "passed": False, "checks": {}}
checks = result["checks"]
try:
    import torch
    checks["device_count"] = torch.cuda.device_count()
    assert checks["device_count"] == 1, "expected exactly one CUDA device"
    props = torch.cuda.get_device_properties(0)
    checks["cuda_uuid"] = str(props.uuid).lower().removeprefix("gpu-")
    assert checks["cuda_uuid"] == parent.lower().removeprefix("gpu-"), "CUDA parent UUID mismatch"
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True, timeout=20).stdout
    checks["nvidia_smi_list"] = listing
    checks["mig_uuids"] = re.findall(r"UUID: (MIG-[0-9a-f-]+)\)", listing)
    listed_parents = re.findall(r"UUID: (GPU-[0-9a-f-]+)\)", listing)
    assert checks["mig_uuids"] == [uuid] and listed_parents == [parent], "CDI exposes extra or wrong devices"
    allocation = torch.empty(20 * 1024**3, dtype=torch.uint8, device="cuda")
    allocation.fill_(1)
    torch.cuda.synchronize()
    checks["allocation_20gib"] = True
    del allocation
    gc.collect()
    torch.cuda.empty_cache()
    checks["allocation_24gib_rejected"] = False
    try:
        allocation = torch.empty(24 * 1024**3, dtype=torch.uint8, device="cuda")
        allocation.fill_(1)
        torch.cuda.synchronize()
        del allocation
    except torch.cuda.OutOfMemoryError as exc:
        checks["allocation_24gib_rejected"] = True
        checks["oom"] = str(exc)
    assert checks["allocation_24gib_rejected"], "24 GiB allocation escaped the instance limit"
    torch.cuda.empty_cache()

    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    setup = None
    if os.environ.get("S4D_MIGVAL_RELEASE") == "1":
        sys.path.insert(0, "/workspace/splatter4d")
        if importlib.util.find_spec("mujoco_mig_setup") is not None:
            import mujoco_mig_setup as setup
    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes, egl.eglGetProcAddress.restype = [C.c_char_p], C.c_void_p
    egl.eglInitialize.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
    egl.eglInitialize.restype = C.c_uint32
    def extension(name, restype, *args):
        pointer = egl.eglGetProcAddress(name)
        assert pointer, name
        return C.CFUNCTYPE(restype, *args)(pointer)
    query_devices = extension(b"eglQueryDevicesEXT", C.c_uint32,
                              C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))
    query_cuda = extension(b"eglQueryDeviceAttribEXT", C.c_uint32,
                           C.c_void_p, C.c_int, C.POINTER(C.c_long))
    get_display = extension(b"eglGetPlatformDisplayEXT", C.c_void_p, C.c_int, C.c_void_p, C.c_void_p)
    devices, count = (C.c_void_p * 64)(), C.c_int()
    assert query_devices(64, devices, C.byref(count)), "EGL enumeration failed"
    matching = []
    for index in range(count.value):
        ordinal = C.c_long(-1)
        if query_cuda(devices[index], 0x323A, C.byref(ordinal)) and ordinal.value == 0:
            matching.append(index)
    if setup is not None:
        selected = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
        assert selected in matching, "repo EGL device does not map to CUDA ordinal 0"
        create_display = setup.create_initialized_egl_device_display_full
    else:
        assert len(matching) == 1, "ambiguous or absent EGL device for CUDA ordinal 0"
        selected = matching[0]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected)
        def create_display():
            ordinal = C.c_long(-1)
            assert query_cuda(devices[selected], 0x323A, C.byref(ordinal)) and ordinal.value == 0
            display = get_display(0x313F, devices[selected], None)
            assert display and egl.eglInitialize(display, None, None), "headless EGL initialization failed"
            return display
        import mujoco.egl as mujoco_egl
        mujoco_egl.create_initialized_egl_device_display = create_display
    display = create_display()
    checks["egl"] = {"cuda_ordinal": 0, "device_index": selected, "initialized": bool(display),
                     "source": "mujoco_mig_setup" if setup is not None else "ctypes"}
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=32, width=32)
    renderer.update_scene(data)
    frame = renderer.render()
    checks["frame"] = {"shape": list(frame.shape), "mean": float(frame.mean())}
    assert list(frame.shape) == [32, 32, 3] and float(frame.max()) > 0, "invalid MuJoCo frame"
    renderer.close()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=other)
    rejected = subprocess.run([sys.executable, "-I", "-c",
        'import json,torch; print(json.dumps({"device_count":torch.cuda.device_count()}))'],
        env=env, capture_output=True, text=True, timeout=30)
    checks["other_instance"] = {"uuid": other, "returncode": rejected.returncode,
                                 "stdout": rejected.stdout, "stderr": rejected.stderr}
    assert rejected.returncode == 0, "alternate UUID check crashed"
    checks["other_instance"]["device_count"] = json.loads(rejected.stdout)["device_count"]
    assert checks["other_instance"]["device_count"] == 0, "another MIG instance is accessible"

    child = subprocess.Popen(["bash", "-c", 'exec -a "$1" sleep 600', "migval", marker],
        start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for attempt in range(100):
        assert child.poll() is None and os.getsid(child.pid) == child.pid, "detached child did not start"
        with open(f"/proc/{child.pid}/cmdline", "rb") as command_line:
            if command_line.read().split(b"\0")[:2] == [marker.encode(), b"600"]:
                break
        time.sleep(0.01)
    else:
        raise RuntimeError("detached sleep did not publish its marker")
    checks["detached_child"] = {"pid": child.pid, "marker": marker, "detached": True}
    result["passed"] = True
except Exception:
    result["error"] = traceback.format_exc()
print(prefix + json.dumps(result), flush=True)
sys.exit(0 if result["passed"] else 1)
'''


def discovery_commands() -> dict[str, str]:
    parents = ",".join(REMOTE_PHYSICAL_GPUS)
    return {
        "boot_id": "cat /proc/sys/kernel/random/boot_id",
        "listing": "nvidia-smi -L",
        "query": f"nvidia-smi -q -i {parents}",
        "modes": "nvidia-smi --query-gpu=uuid,mig.mode.current,mig.mode.pending "
                 f"--format=csv,noheader,nounits -i {parents}",
        "cdi_mtime": f"stat -Lc %Y {CDI}",
        "cdi": f"cat {CDI}",
        "boot_time": "awk '$1 == \"btime\" {print $2}' /proc/stat",
        "cdi_mtime_after": f"stat -Lc %Y {CDI}",
        "boot_id_after": "cat /proc/sys/kernel/random/boot_id",
    }


def parse_modes(text: str) -> dict:
    modes = {}
    for fields in csv.reader(text.splitlines()):
        fields = [value.strip() for value in fields]
        if len(fields) != 3 or fields[0] in modes:
            raise ValueError("malformed or duplicate MIG mode row")
        if fields[0] in REMOTE_PHYSICAL_GPUS:
            modes[fields[0]] = fields[1:]
    return modes


def parse_cdi(text: str) -> list[str]:
    # BaseLoader preserves numeric/index-style device names, including YAML 1.1 sexagesimal strings.
    doc = yaml.load(text, Loader=yaml.BaseLoader)
    if not isinstance(doc, dict) or doc.get("kind") != "nvidia.com/gpu" or not isinstance(doc.get("devices"), list):
        raise ValueError("invalid NVIDIA CDI devices document")
    names = [device.get("name") if isinstance(device, dict) else None for device in doc["devices"]]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("invalid or duplicate CDI device name")
    return names


def validate_discovery(evidence: dict, allowed=None) -> dict:
    """Pure validation. Reports only authorized devices, even if global discovery lists other users."""
    inventory = build_inventory(evidence["listing"], evidence["query"])
    errors, warnings = list(inventory["problems"]), []
    boot = evidence["boot_id"].strip()
    if not BOOT_ID.fullmatch(boot) or evidence["boot_id_after"].strip() != boot:
        errors.append("remote boot changed or boot ID is unreadable")
    modes = parse_modes(evidence["modes"])
    if set(modes) != set(REMOTE_PHYSICAL_GPUS) or any(v != ["Enabled", "Enabled"] for v in modes.values()):
        errors.append("all four parents must have MIG mode Enabled/Enabled")
    names = parse_cdi(evidence["cdi"])
    missing = sorted({item["uuid"] for item in inventory["instances"]} - set(names))
    if missing:
        errors.append("CDI is missing inventory MIG UUIDs: " + ", ".join(missing))
    mtime, boot_time = int(evidence["cdi_mtime"]), int(evidence["boot_time"])
    if mtime <= 0 or boot_time <= 0 or mtime < boot_time:
        errors.append("stale CDI spec: root must regenerate CDI after creating the instances")
    if evidence["cdi_mtime_after"].strip() != evidence["cdi_mtime"].strip():
        errors.append("CDI spec changed during discovery; retry discovery")
    full = sorted(set(names) & set(REMOTE_PHYSICAL_GPUS))
    indexes = {item["index"] for item in inventory["parents"]}
    full_indexes = sorted(set(names) & {str(index) for index in indexes})
    if full or full_indexes:
        warnings.append("CDI still contains full-GPU entries for authorized parents")
    drift = None
    if allowed is not None:
        live = {item["uuid"]: item for item in inventory["instances"]}
        reviewed = {item["uuid"]: item for item in allowed}
        changed = sorted(uuid for uuid in live.keys() & reviewed.keys()
                         if any(live[uuid][key] != reviewed[uuid][key]
                                for key in ("parent", "device", "gpu_instance_id", "compute_instance_id")))
        drift = {"missing": sorted(reviewed.keys() - live.keys()), "new": sorted(live.keys() - reviewed.keys()),
                 "mapping_changed": changed}
        if any(drift.values()):
            errors.append("allowlist UUID drift or instance mapping drift; review and re-pin the allowlist")
    index_names = sorted(name for name in names if re.fullmatch(r"\d+:\d+", name)
                         and int(name.split(":")[0]) in indexes)
    return {"passed": inventory["complete"] and not errors, "errors": errors, "warnings": warnings,
            "boot_id": boot, "inventory": inventory, "modes": modes, "allowlist_drift": drift,
            "cdi": {"missing_mig_uuids": missing, "full_gpu_entries": full, "full_gpu_index_names": full_indexes,
                    "index_names": index_names,
                    "mtime": mtime, "boot_time": boot_time,
                    "mig_names": sorted(set(names) & {i["uuid"] for i in inventory["instances"]})},
            "authorized_listing": [p for p in parse_list(evidence["listing"]) if p["uuid"] in REMOTE_PHYSICAL_GPUS]}


def container_spec(boot: str, index: int, instance: dict, other: str, image: str, release_code=None) -> dict:
    if not BOOT_ID.fullmatch(boot) or not IMAGE.fullmatch(image):
        raise ValueError("a valid boot ID and pinned sha256 image digest are required")
    if instance["parent"] not in REMOTE_PHYSICAL_GPUS or not re.fullmatch(r"MIG-[0-9a-f-]{36}", instance["uuid"]):
        raise ValueError("only allowlisted MIG instances of authorized parents may be validated")
    if other == instance["uuid"] or not re.fullmatch(r"MIG-[0-9a-f-]{36}", other):
        raise ValueError("alternate UUID must be another allowlisted instance")
    if type(index) is not int or not 0 <= index < 16:
        raise ValueError("invalid instance index")
    if release_code is not None:
        path = PurePosixPath(release_code)
        if not path.is_relative_to(ROOT + "/") or ".." in path.parts or any(c in str(path) for c in ",\n\r\x00"):
            raise ValueError("release code must be a path within the authorized remote root")
        release_code = str(path)
    spec = {"name": f"s4d-migval-{boot[:8]}-{index}", "boot_id": boot, "uuid": instance["uuid"],
            "parent": instance["parent"], "gpu_instance_id": int(instance["gpu_instance_id"]),
            "device": int(instance["device"]), "other": other, "image": image, "release_code": release_code,
            "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest()}
    spec["fingerprint"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    spec["marker"] = "s4d-migval-child-" + spec["fingerprint"][:32]
    # s4d.gpu publishes the reservation to every other launcher's exclusive-slot check (Codex review 2, finding 4)
    spec["labels"] = {"s4d.project": "splatter4d", "s4d.owner": "kaist-sunho", "s4d.role": "migval",
                      "s4d.migval.spec": spec["fingerprint"], "s4d.gpu": spec["uuid"]}
    return spec


def container_command(spec: dict) -> str:
    args = ["docker", "run", "-d", "--pull", "never", "--name", spec["name"], "--network", "none",
            "--read-only", "--memory", "32g", "--memory-swap", "32g", "--cpus", "4", "--init", "--pid=host",
            "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--device", "nvidia.com/gpu=" + spec["uuid"]]
    for key, value in spec["labels"].items():
        args += ["--label", key + "=" + value]
    env = {"NVIDIA_VISIBLE_DEVICES": "void", "CUDA_VISIBLE_DEVICES": spec["uuid"],
           "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics", "MUJOCO_GL": "egl",
           "PYOPENGL_PLATFORM": "egl", "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp",
           "XDG_CACHE_HOME": "/tmp/.cache", "OMP_NUM_THREADS": "4", "S4D_MIGVAL_PARENT": spec["parent"],
           "S4D_MIGVAL_OTHER": spec["other"], "S4D_MIGVAL_MARKER": spec["marker"],
           "S4D_MIGVAL_RELEASE": "1" if spec["release_code"] else "0"}
    if spec["release_code"]:
        args += ["--mount", f"type=bind,src={spec['release_code']},dst=/workspace/splatter4d,readonly"]
        env["S4D_HOST_CONFIG"] = "/workspace/splatter4d/configs/hosts/remote.yaml"
    for key, value in env.items():
        args += ["--env", key + "=" + value]
    return shlex.join(args + ["--entrypoint", "python", spec["image"], "-I", "-B", "-c", PROBE])


def launch_command(spec: dict, root: str = ROOT) -> str:
    q = shlex.quote
    # The admission lock covers name reconciliation and the final boot-bound hold check.
    lines = ["set -eu", f"exec 9>{q(root + '/runtime/launch/admission.lock')}", "flock -x 9",
             check_shell(root), f'test "$(cat /proc/sys/kernel/random/boot_id)" = {q(spec["boot_id"])}',
             "cid=$(docker ps -aq --no-trunc --filter " + q("name=^/" + spec["name"] + "$") + ")",
             'if [ -n "$cid" ]; then', 'test "${#cid}" -eq 64']
    for key, value in spec["labels"].items():
        template = '{{index .Config.Labels "' + key + '"}}'
        lines.append(f'test "$(docker inspect --format {q(template)} "$cid")" = {q(value)}')
    # inside the lock, from fresh evidence (Codex review 2, finding 4): no container of ours bound to the instance, the
    # UUID bound to its parent/device/GPU instance with MIG mode Enabled/Enabled, no process, framebuffer idle
    probe = (f'probe="$(printf \'%s\\n__S4D_QUERY__\\n%s\\n\' "$listing" "$mig_query" | python3 -I -c {q(MIG_PROBE)} '
             f'{int(spec["gpu_instance_id"])} {int(spec["device"])} {q(spec["uuid"])} {q(spec["parent"])})"')
    lines += ['printf \'%s\\n\' "$cid"', "else",
              "test -z \"$(docker ps -q --filter " + q("label=s4d.gpu=" + spec["uuid"]) + ")\"",
              f'listing="$(nvidia-smi -L)"; mig_query="$(nvidia-smi -q -i {q(spec["parent"])})"', probe,
              'test -z "$(printf \'%s\\n\' "$probe" | grep \'^APP,\')"',
              'used="$(printf \'%s\\n\' "$probe" | awk -F, \'$1=="MEM" {print $3}\')"; '
              'test -n "$used" && test "$used" -le ' + str(IDLE_FB_MIB),
              container_command(spec), "fi"]
    return "\n".join(lines)


def exact_command(action: str, cid: str) -> str:
    if not CONTAINER_ID.fullmatch(cid):
        raise ValueError("an exact container ID is required")
    commands = {"inspect": f"docker inspect {cid}", "logs": f"docker logs {cid}",
                "wait": f"timeout 180 docker wait {cid}", "stop": f"docker stop --time 5 {cid}",
                "remove": f"docker rm {cid}"}
    return commands[action]


def survivor_command(marker: str) -> str:
    if not re.fullmatch(r"s4d-migval-child-[0-9a-f]{32}", marker):
        raise ValueError("invalid validation child marker")
    # Brackets keep pgrep and its ancestor SSH shell from matching themselves.
    pattern = "[s]" + marker[1:]
    return f"pgrep -f -- {shlex.quote(pattern)}; rc=$?; [ \"$rc\" -le 1 ] || exit \"$rc\""


def parse_survivors(text: str) -> list[int]:
    if any(not line.isdigit() or int(line) <= 0 for line in text.splitlines()):
        raise ValueError("unreadable surviving-process evidence")
    return [int(line) for line in text.splitlines()]


def memory_command(instance: dict, boot: str) -> str:
    if instance["parent"] not in REMOTE_PHYSICAL_GPUS or not BOOT_ID.fullmatch(boot):
        raise ValueError("unauthorized parent or invalid boot ID")
    return (f'test "$(cat /proc/sys/kernel/random/boot_id)" = {shlex.quote(boot)} && '
            f'nvidia-smi -q -i {instance["parent"]}')


def parse_memory(text: str, instance: dict) -> dict:
    info = parse_query(text)[instance["parent"]]
    device = info["instances"][instance["device"]]
    if (info["mig_current"] != "Enabled" or info["mig_pending"] != "Enabled"
            or not info["processes_listed"] or type(device["used_mib"]) is not int
            or device["gpu_instance_id"] != instance["gpu_instance_id"]
            or device["compute_instance_id"] != instance["compute_instance_id"]):
        raise ValueError("MIG mapping or framebuffer/process evidence is unreadable or changed")
    processes = [p for p in info["processes"] if p["gpu_instance_id"] in (None, instance["gpu_instance_id"])]
    return {"used_mib": device["used_mib"], "processes": processes,
            "idle": device["used_mib"] <= 256 and not processes}


def parse_inspect(text: str, cid: str, spec: dict) -> dict:
    docs = json.loads(text)
    if not isinstance(docs, list) or len(docs) != 1:
        raise ValueError("expected exactly one inspected container")
    doc = docs[0]
    config, host = doc["Config"], doc["HostConfig"]
    env = config.get("Env") or []
    requests = host.get("DeviceRequests") or []
    expected = "nvidia.com/gpu=" + spec["uuid"]
    if (doc["Id"] != cid or doc["Name"] != "/" + spec["name"]
            or any((config.get("Labels") or {}).get(k) != v for k, v in spec["labels"].items())
            or config["Image"] != spec["image"] or config.get("Entrypoint") != ["python"]
            or config.get("Cmd") != ["-I", "-B", "-c", PROBE]
            or f"CUDA_VISIBLE_DEVICES={spec['uuid']}" not in env or "NVIDIA_VISIBLE_DEVICES=void" not in env
            or f"S4D_MIGVAL_MARKER={spec['marker']}" not in env
            or host.get("Privileged") or host.get("AutoRemove") or not host.get("ReadonlyRootfs")
            or host.get("PidMode") != "host" or host.get("NetworkMode") != "none" or not host.get("Init")
            or host.get("Memory") != 32 * 1024**3 or host.get("NanoCpus") != 4 * 10**9
            or host.get("Devices") or len(requests) != 1 or requests[0].get("Driver") != "cdi"
            or requests[0].get("DeviceIDs") != [expected]):
        raise ValueError("container identity, ownership, confinement or CDI device differs from validation spec")
    binds = [m for m in doc.get("Mounts", []) if m.get("Type") == "bind"]
    if spec["release_code"]:
        if (len(binds) != 1 or binds[0].get("Source") != spec["release_code"]
                or binds[0].get("Destination") != "/workspace/splatter4d" or binds[0].get("RW") is not False):
            raise ValueError("release bind mount differs from validation spec")
    elif binds:
        raise ValueError("unexpected bind mounts")
    return doc


def parse_probe(text: str, spec: dict) -> dict:
    lines = [line[len(PROBE_PREFIX):] for line in text.splitlines() if line.startswith(PROBE_PREFIX)]
    if len(lines) != 1:
        raise ValueError("expected exactly one probe evidence record")
    result = json.loads(lines[0])
    checks = result.get("checks", {})
    egl, frame = checks.get("egl", {}), checks.get("frame", {})
    alternate, child = checks.get("other_instance", {}), checks.get("detached_child", {})
    if (result.get("passed") is not True or result.get("uuid") != spec["uuid"]
            or result.get("parent") != spec["parent"] or result.get("marker") != spec["marker"]
            or checks.get("device_count") != 1 or checks.get("cuda_uuid") != spec["parent"][4:].lower()
            or checks.get("mig_uuids") != [spec["uuid"]] or checks.get("allocation_20gib") is not True
            or checks.get("allocation_24gib_rejected") is not True
            or egl.get("initialized") is not True or egl.get("cuda_ordinal") != 0
            or type(egl.get("device_index")) is not int or egl["device_index"] < 0
            or frame.get("shape") != [32, 32, 3] or not 0 < frame.get("mean", 0) <= 255
            or alternate.get("uuid") != spec["other"] or alternate.get("returncode") != 0
            or alternate.get("device_count") != 0 or child.get("marker") != spec["marker"]
            or child.get("detached") is not True or type(child.get("pid")) is not int or child["pid"] <= 0):
        raise ValueError("probe failed or omitted required CUDA, memory, EGL, isolation or child evidence")
    return result


def atomic_report(path: Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name + ".", delete=False) as out:
            temporary = out.name
            json.dump(report, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)  # the new name is durable before any removal
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def new_report(command: str) -> dict:
    return {"schema": SCHEMA, "command": command, "passed": False, "errors": [], "warnings": [],
            "steps": [], "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def remote(client, report: dict, step: str, command: str, *, idempotent=True, timeout=60):
    record = {"step": step, "command": command, "idempotent": idempotent}
    report.setdefault("steps", []).append(record)
    try:
        result = client.checked_ssh(command, wait=False, idempotent=idempotent, timeout=timeout)
        record.update(returncode=result.returncode, stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                      stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f"{step} exited {result.returncode}")
        return result
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise


def discover(client, allowlist=None, *, reviewed=None) -> dict:
    report = new_report("discover")
    try:
        allowed = load_allowlist(Path(allowlist)) if allowlist is not None else None
        if allowlist is not None and allowed is None:
            raise ValueError("given allowlist file is missing")
        if reviewed is not None and allowed != reviewed:
            raise ValueError("allowlist changed during validation; review and rerun")
        evidence = {key: remote(client, report, key, command).stdout
                    for key, command in discovery_commands().items()}
        report["discovery"] = validate_discovery(evidence, allowed)
        report["errors"] += report["discovery"]["errors"]
        report["warnings"] += report["discovery"]["warnings"]
        report["passed"] = report["discovery"]["passed"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def wait_idle(client, report: dict, instance: dict, boot: str, *, deadline=None) -> list[dict]:
    samples = []
    if deadline is None:
        deadline = time.monotonic() + 30
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("instance framebuffer did not return to idle (<= 256 MiB) within 30 s")
        sample = parse_memory(remote(client, report, "framebuffer", memory_command(instance, boot),
                                     timeout=min(remaining, 10)).stdout, instance)
        sample["elapsed_seconds"] = 30 - (deadline - time.monotonic())
        samples.append(sample)
        report["framebuffer_samples"] = samples
        if sample["elapsed_seconds"] > 30:
            raise RuntimeError("framebuffer idle evidence arrived after the 30 s deadline")
        if sample["idle"]:
            return samples
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def validate_instance(client, report: dict, instance: dict, spec: dict, out: Path) -> dict:
    item = {"spec": spec, "passed": False, "errors": [], "steps": []}
    report["instances"].append(item)
    trusted, stopped, archived, cid = False, False, False, None
    try:
        survivors = parse_survivors(remote(client, item, "preflight_child", survivor_command(spec["marker"])).stdout)
        if survivors:
            raise RuntimeError(f"surviving process from a previous validation: {survivors}")
        item["preflight_memory"] = parse_memory(remote(client, item, "preflight_memory",
                                                       memory_command(instance, spec["boot_id"])).stdout, instance)
        if not item["preflight_memory"]["idle"]:
            raise RuntimeError("instance is busy or has foreign processes; refusing validation")
        cid = remote(client, item, "launch", launch_command(spec, client.root)).stdout.strip()
        if not CONTAINER_ID.fullmatch(cid):
            raise ValueError("launch returned no exact container ID; rerun to reconcile the deterministic name")
        item["container_id"] = cid
        item["initial_inspect"] = parse_inspect(remote(client, item, "inspect", exact_command("inspect", cid)).stdout,
                                                 cid, spec)
        trusted = True
        try:
            item["wait_exit_code"] = int(remote(client, item, "wait", exact_command("wait", cid), timeout=190).stdout)
        except Exception as exc:
            item["errors"].append(f"container wait failed: {exc}")
            remote(client, item, "stop", exact_command("stop", cid), idempotent=False, timeout=20)
        idle_deadline = time.monotonic() + 30
        item["inspect"] = parse_inspect(remote(client, item, "final_inspect", exact_command("inspect", cid)).stdout,
                                         cid, spec)
        state = item["inspect"]["State"]
        stopped = state.get("Running") is False and state.get("Status") in ("exited", "dead")
        logs = remote(client, item, "logs", exact_command("logs", cid))
        item["logs"] = {"stdout": logs.stdout, "stderr": logs.stderr}
        if not stopped or state.get("ExitCode") != 0 or state.get("OOMKilled") or item.get("wait_exit_code") != 0:
            item["errors"].append("container failed, was OOM-killed or did not exit normally")
        try:
            item["probe"] = parse_probe(logs.stdout, spec)
        except Exception as exc:
            item["errors"].append(str(exc))
        item["surviving_pids"] = parse_survivors(remote(client, item, "kill_all", survivor_command(spec["marker"])).stdout)
        if item["surviving_pids"]:
            item["errors"].append(f"surviving process after container exit: {item['surviving_pids']}")
        wait_idle(client, item, instance, spec["boot_id"], deadline=idle_deadline)
    except Exception as exc:
        item["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # Persist logs + inspect before removing ONLY this verified, stopped container, by full ID.
        if trusted and stopped and "logs" in item:
            atomic_report(out, report)
            archived = True
        if archived:
            try:
                item["removal"] = remote(client, item, "remove", exact_command("remove", cid), idempotent=False).stdout
            except Exception as exc:
                item["errors"].append(f"container cleanup failed: {exc}")
        item["passed"] = not item["errors"] and archived and "removal" in item
        atomic_report(out, report)
    return item


def instances(client, *, allowlist: Path, image: str, out: Path, release_code=None) -> dict:
    report = new_report("instances")
    report["instances"] = []
    if Path(out).exists():  # the report holds the only archives of removed containers: never overwritten
        report["errors"].append(f"FileExistsError: {out} exists; choose a new --out (evidence is never overwritten)")
        return report
    try:
        allowed = load_allowlist(Path(allowlist))
        if not allowed or len(allowed) != 16:
            raise ValueError("instances requires a reviewed allowlist containing all 16 MIG instances")
        if not IMAGE.fullmatch(image):
            raise ValueError("--image must be pinned by sha256 digest")
        state = parse_status(remote(client, report, "launch_hold", status_shell(client.root)).stdout)
        report["launch_hold"] = state
        if state["held"]:
            raise RuntimeError(f"remote launch hold in force (exit {EXIT_HELD}); "
                               "explicit user-authorized resumption required")
        discovery = discover(client, allowlist, reviewed=allowed)
        report["discovery"] = discovery
        report["warnings"] += discovery["warnings"]
        if not discovery["passed"]:
            raise RuntimeError("discovery failed: " + "; ".join(discovery["errors"]))
        if discovery["discovery"]["boot_id"] != state["boot_id"]:
            raise RuntimeError("remote boot changed after checking the launch hold")
        for index, instance in enumerate(allowed):
            spec = container_spec(state["boot_id"], index, instance, allowed[(index + 1) % len(allowed)]["uuid"],
                                  image, release_code)
            if not validate_instance(client, report, instance, spec, out)["passed"]:
                raise RuntimeError(f"validation failed for {instance['uuid']}; see instance evidence")
        report["passed"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    atomic_report(out, report)
    return report


def summary(report: dict) -> str:
    checked = report.get("instances", [])
    lines = [f"{'PASS' if report.get('passed') else 'FAIL'} {report.get('command', 'unknown')}"]
    if checked:
        lines.append(f"Instances passed: {sum(item.get('passed') is True for item in checked)}/{len(checked)}")
    discovery = report.get("discovery", {})
    inventory = discovery.get("inventory", discovery.get("discovery", {}).get("inventory"))
    if inventory is not None:
        lines.append(f"Usable MIG instances: {len(inventory['instances'])}/16 on {len(inventory['parents'])}/4 parents")
    lines += ["ERROR: " + error for error in report.get("errors", [])]
    lines += ["WARNING: " + warning for warning in report.get("warnings", [])]
    for item in checked:
        lines += [item["spec"]["uuid"] + ": " + error for error in item.get("errors", [])]
    return "\n".join(lines)


def main(argv=None, *, client_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    discovery = sub.add_parser("discover", help="read-only inventory and CDI acceptance")
    discovery.add_argument("--allowlist", type=Path)
    discovery.add_argument("--out", type=Path, required=True)
    validation = sub.add_parser("instances", help="sequential GPU probes after explicit resumption")
    validation.add_argument("--allowlist", type=Path, required=True)
    validation.add_argument("--image", required=True)
    validation.add_argument("--release-code")
    validation.add_argument("--out", type=Path, required=True)
    display = sub.add_parser("report", help="summarize a saved JSON report; no remote access")
    display.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    report = new_report(args.command)
    try:
        if args.command == "report":
            saved = json.loads(args.path.read_text())
            if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
                raise ValueError("not a MIG validation report")
            report = saved
        else:
            if client_factory is None:
                from ops.remote.access import RemoteClient

                client_factory = RemoteClient
            client = client_factory()
            if args.command == "discover":
                report = discover(client, args.allowlist)
            else:
                report = instances(client, allowlist=args.allowlist, image=args.image, out=args.out,
                                   release_code=args.release_code)
    except Exception as exc:
        report["passed"] = False
        report.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
    if args.command != "report" and not (args.command == "instances" and any(
            str(error).startswith("FileExistsError") for error in report.get("errors", []))):
        atomic_report(args.out, report)  # an existing instances report (the only archive of removed containers) stays
    print(summary(report))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
