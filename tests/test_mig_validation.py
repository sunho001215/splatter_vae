"""Synthetic acceptance evidence only: no SSH, Docker, CUDA or campaign imports."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shlex
import socket
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from ops.remote import mig_validation as val
from ops.resources import mig_inventory as inv

PARENTS = inv.REMOTE_PHYSICAL_GPUS
FOREIGN = "GPU-1aa64fba-f45e-ee28-f83e-a272cb624cb4"
BOOT = "12345678-1234-5678-9abc-123456789abc"
IMAGE = "registry.example/s4d@sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("tests must never execute a subprocess or open a network connection")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def mig_uuid(parent, device):
    return f"MIG-{parent:08x}-{device:04x}-5000-8000-{parent * 16 + device:012x}"


def render_list(parents=(*PARENTS, FOREIGN)):
    lines = []
    for index, parent in enumerate(parents):
        lines.append(f"GPU {index}: NVIDIA RTX PRO 6000 (UUID: {parent})")
        lines += [f"  MIG 1g.24gb Device {device}: (UUID: {mig_uuid(index, device)})" for device in range(4)]
    return "\n".join(lines) + "\n"


def render_query(parents=PARENTS, *, used=67, processes=None):
    lines = []
    for index, parent in enumerate(parents):
        lines += [f"GPU 00000000:{0x41 + index:02X}:00.0", f"    GPU UUID : {parent}", "    MIG Mode",
                  "        Current : Enabled", "        Pending : Enabled"]
        for device in range(4):
            lines += ["    MIG Device", f"        Index : {device}", f"        GPU Instance ID : {3 + device}",
                      "        Compute Instance ID : 0", "        FB Memory Usage", "            Total : 24192 MiB",
                      f"            Used : {used} MiB"]
        if processes:
            lines.append("    Processes")
            for gi, pid in processes:
                lines += [f"        GPU instance ID : {gi}", "        Compute instance ID : 0",
                          f"        Process ID : {pid}", "            Used GPU Memory : 100 MiB"]
        else:
            lines.append("    Processes : None")
    return "\n".join(lines) + "\n"


def evidence():
    names = [mig_uuid(parent, device) for parent in range(5) for device in range(4)]
    names += list(PARENTS) + [FOREIGN, "0:0", "1:1", "4:0"]
    return {"listing": render_list(), "query": render_query(), "boot_id": BOOT, "boot_id_after": BOOT,
            "modes": "\n".join(f"{parent}, Enabled, Enabled" for parent in (*PARENTS, FOREIGN)),
            "cdi": yaml.safe_dump({"cdiVersion": "0.8.0", "kind": "nvidia.com/gpu",
                                   "devices": [{"name": name, "containerEdits": {}} for name in names]}),
            "cdi_mtime": "200", "cdi_mtime_after": "200", "boot_time": "100"}


def allowed():
    return inv.allowlist_document(inv.build_inventory(render_list(), render_query()))


@pytest.fixture
def allowlist(tmp_path):
    path = tmp_path / "remote-mig.yaml"
    path.write_text(yaml.safe_dump(allowed()))
    return path


def spec(index=0, release_code=None):
    items = allowed()["instances"]
    return val.container_spec(BOOT, index, items[index], items[(index + 1) % 16]["uuid"], IMAGE, release_code)


def probe(specification):
    return {"passed": True, "uuid": specification["uuid"], "parent": specification["parent"],
            "marker": specification["marker"], "checks": {
                "device_count": 1, "cuda_uuid": specification["parent"][4:],
                "mig_uuids": [specification["uuid"]], "allocation_20gib": True, "allocation_24gib_rejected": True,
                "egl": {"initialized": True, "device_index": 0, "cuda_ordinal": 0, "source": "ctypes"},
                "frame": {"shape": [32, 32, 3], "mean": 8.5},
                "other_instance": {"uuid": specification["other"], "returncode": 0, "device_count": 0},
                "detached_child": {"pid": 123, "marker": specification["marker"], "detached": True}}}


def inspection(specification, cid, *, running=False, exit_code=0):
    return {"Id": cid, "Name": "/" + specification["name"], "Config": {
        "Labels": dict(specification["labels"]), "Image": specification["image"], "Entrypoint": ["python"],
        "Cmd": ["-I", "-B", "-c", val.PROBE], "Env": [
            "CUDA_VISIBLE_DEVICES=" + specification["uuid"], "NVIDIA_VISIBLE_DEVICES=void",
            "S4D_MIGVAL_MARKER=" + specification["marker"]]},
        "HostConfig": {"Privileged": False, "AutoRemove": False, "ReadonlyRootfs": True, "PidMode": "host",
                       "NetworkMode": "none", "Init": True, "Memory": 32 * 1024**3, "NanoCpus": 4 * 10**9,
                       "Devices": [], "DeviceRequests": [{"Driver": "cdi", "DeviceIDs": [
                           "nvidia.com/gpu=" + specification["uuid"]]}]},
        "Mounts": ([{"Type": "bind", "Source": specification["release_code"],
                     "Destination": "/workspace/splatter4d", "RW": False}] if specification["release_code"] else []),
        "State": {"Running": running, "Status": "running" if running else "exited", "ExitCode": exit_code,
                  "OOMKilled": False}}


class FakeClient:
    root = val.ROOT

    def __init__(self, out=None, *, held=False, survivor=False, probe_failure=False, wait_failure=False,
                 release_code=None, preflight_survivor=False, foreign_process=False):
        self.out, self.held, self.survivor, self.probe_failure = out, held, survivor, probe_failure
        self.wait_failure, self.preflight_survivor = wait_failure, preflight_survivor
        self.foreign_process = foreign_process
        self.calls, self.live, self.removed, self.waited = [], {}, [], []
        self.evidence = evidence()
        self.specs = {spec(i, release_code)["name"]: spec(i, release_code) for i in range(16)}
        self.process_queries = {}

    def checked_ssh(self, command, *, wait, idempotent, timeout):
        assert wait is False and timeout > 0
        self.calls.append((command, idempotent))
        stdout, stderr = "", ""
        if command == val.status_shell(self.root):
            stdout = f"__S4D_LAUNCH_HOLD__ {BOOT} {int(self.held)} 1\n"
        elif command in val.discovery_commands().values():
            key = next(k for k, v in val.discovery_commands().items() if v == command)
            stdout = self.evidence[key]
        elif command.startswith("pgrep -f"):
            self.process_queries[command] = self.process_queries.get(command, 0) + 1
            if self.preflight_survivor or (self.survivor and self.process_queries[command] > 1):
                stdout = "9876\n"
        elif "nvidia-smi -q -i" in command:
            parent = command.rsplit(" ", 1)[1]
            assert parent in PARENTS
            stdout = render_query([parent], processes=[(3, 9999)] if self.foreign_process else None)
        elif "\nelse\ndocker run " in command:
            args = shlex.split(command.split("\nelse\n", 1)[1].rsplit("\nfi", 1)[0])
            name = args[args.index("--name") + 1]
            cid = hashlib.sha256(name.encode()).hexdigest()
            if cid not in self.live:
                assert not self.live, "validation must run sequentially"
                self.live[cid] = {"spec": self.specs[name], "running": True}
            stdout = cid + "\n"
        elif command.startswith("docker inspect "):
            cid = command.rsplit(" ", 1)[1]
            container = self.live[cid]
            stdout = json.dumps([inspection(container["spec"], cid, running=container["running"],
                                           exit_code=int(self.probe_failure))])
        elif command.startswith("timeout 180 docker wait "):
            cid = command.rsplit(" ", 1)[1]
            if self.wait_failure:
                raise RuntimeError("synthetic wait timeout")
            self.live[cid]["running"] = False
            self.waited.append(cid)
            stdout = str(int(self.probe_failure)) + "\n"
        elif command.startswith("docker stop --time 5 "):
            cid = command.rsplit(" ", 1)[1]
            self.live[cid]["running"] = False
            stdout = cid + "\n"
        elif command.startswith("docker logs "):
            cid = command.rsplit(" ", 1)[1]
            result = probe(self.live[cid]["spec"])
            if self.probe_failure:
                result["passed"] = False
                result["error"] = "synthetic allocation or EGL failure"
            stdout = "library diagnostic\n" + val.PROBE_PREFIX + json.dumps(result) + "\n"
            stderr = "synthetic warning\n"
        elif command.startswith("docker rm "):
            cid = command.rsplit(" ", 1)[1]
            assert not self.live[cid]["running"]
            assert self.out is not None
            archived = json.loads(self.out.read_text())
            saved = next(item for item in archived["instances"] if item.get("container_id") == cid)
            assert saved["logs"]["stdout"].startswith("library diagnostic\n")
            assert saved["inspect"]["Id"] == cid
            assert val.PROBE_PREFIX in saved["logs"]["stdout"]
            del self.live[cid]
            self.removed.append(cid)
            stdout = cid + "\n"
        else:
            raise AssertionError("unexpected remote command: " + command[:200])
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)


def test_discovery_complete_filters_foreign_devices_and_warns_full_gpu():
    report = val.validate_discovery(evidence())
    assert report["passed"] and len(report["inventory"]["instances"]) == 16
    assert report["cdi"]["full_gpu_entries"] == sorted(PARENTS)
    assert report["cdi"]["index_names"] == ["0:0", "1:1"]
    assert report["warnings"]
    assert FOREIGN not in json.dumps(report) and mig_uuid(4, 0) not in json.dumps(report)


@pytest.mark.parametrize("failure, expected", [
    ("missing_instance", "3 usable"), ("missing_cdi_uuid", "CDI is missing"),
    ("stale", "root must regenerate CDI after creating the instances"),
    ("changed_cdi", "CDI spec changed"), ("reboot", "remote boot changed"),
    ("disabled", "Enabled/Enabled"), ("foreign_process", "3 usable"),
])
def test_discovery_failures(failure, expected):
    data = evidence()
    if failure == "missing_instance":
        data["listing"] = "\n".join(line for line in data["listing"].splitlines() if mig_uuid(0, 0) not in line)
    elif failure == "missing_cdi_uuid":
        data["cdi"] = data["cdi"].replace("name: " + mig_uuid(0, 0), "name: other-name")
    elif failure == "stale":
        data["cdi_mtime"] = data["cdi_mtime_after"] = "99"
    elif failure == "changed_cdi":
        data["cdi_mtime_after"] = "201"
    elif failure == "reboot":
        data["boot_id_after"] = "abcdefab-1234-5678-9abc-123456789abc"
    elif failure == "disabled":
        data["modes"] = data["modes"].replace("Enabled, Enabled", "Enabled, Disabled", 1)
    elif failure == "foreign_process":
        data["query"] = render_query(processes=[(3, 999)])
    report = val.validate_discovery(data)
    assert not report["passed"] and any(expected in error for error in report["errors"])
    if failure == "foreign_process":
        assert all("foreign processes" in item["reason"] for item in report["inventory"]["excluded"])


def test_allowlist_uuid_and_mapping_drift():
    data = evidence()
    old, new = mig_uuid(0, 0), "MIG-aaaaaaaa-aaaa-5000-8000-aaaaaaaaaaaa"
    for key in ("listing", "cdi"):
        data[key] = data[key].replace(old, new)
    result = val.validate_discovery(data, inv.validate_allowlist(allowed()))
    assert not result["passed"]
    assert result["allowlist_drift"]["missing"] == [old]
    assert result["allowlist_drift"]["new"] == [new]
    reviewed = copy.deepcopy(allowed())
    reviewed["instances"][0]["compute_instance_id"] = 12
    result = val.validate_discovery(evidence(), inv.validate_allowlist(reviewed))
    assert result["allowlist_drift"]["mapping_changed"] == [old]


@pytest.mark.parametrize("text", ["", "devices: []", "kind: other\ndevices: []", "kind: nvidia.com/gpu\ndevices: [null]",
                                 "kind: nvidia.com/gpu\ndevices: [{name: a}, {name: a}]"])
def test_cdi_invalid_or_duplicate_names(text):
    with pytest.raises(ValueError):
        val.parse_cdi(text)


def test_cdi_preserves_yaml_index_names():
    assert val.parse_cdi("kind: nvidia.com/gpu\ndevices: [{name: 1:1}, {name: '0:1'}]") == ["1:1", "0:1"]


def test_discovery_remote_commands_are_read_only_and_scoped(allowlist):
    client = FakeClient()
    report = val.discover(client, allowlist)
    assert report["passed"]
    commands = [cmd for cmd, _ in client.calls]
    assert commands == list(val.discovery_commands().values())
    assert all("docker" not in cmd and "sudo" not in cmd for cmd in commands)
    assert all(FOREIGN not in cmd for cmd in commands)
    assert all(",".join(PARENTS) in cmd for cmd in commands if "nvidia-smi" in cmd and "-L" not in cmd)
    assert all(idempotent for _, idempotent in client.calls)


def test_missing_allowlist_and_unreadable_processes_fail_closed(tmp_path, allowlist):
    client = FakeClient()
    assert not val.discover(client, tmp_path / "absent.yaml")["passed"]
    assert not client.calls
    client.evidence["query"] = render_query().replace("Processes : None", "Processes : Insufficient Permissions")
    assert not val.discover(client, allowlist)["passed"]


def test_container_command_contract_and_quoted_optional_mount():
    specification = spec(release_code=val.ROOT + "/releases/a b")
    args = shlex.split(val.container_command(specification))
    assert args[:3] == ["docker", "run", "-d"] and "--rm" not in args
    assert args.count("--device") == 1 and args[args.index("--device") + 1] == "nvidia.com/gpu=" + mig_uuid(0, 0)
    for flag, value in (("--network", "none"), ("--memory", "32g"), ("--cpus", "4")):
        assert args[args.index(flag) + 1] == value
    assert all(flag in args for flag in ("--read-only", "--init", "--pid=host"))
    assert "NVIDIA_VISIBLE_DEVICES=void" in args and "CUDA_VISIBLE_DEVICES=" + mig_uuid(0, 0) in args
    assert args[args.index("--mount") + 1] == (
        "type=bind,src=" + val.ROOT + "/releases/a b,dst=/workspace/splatter4d,readonly")
    assert "S4D_HOST_CONFIG=/workspace/splatter4d/configs/hosts/remote.yaml" in args
    assert args[-6:] == ["python", IMAGE, "-I", "-B", "-c", val.PROBE]
    assert "--mount" not in shlex.split(val.container_command(spec()))
    launch = val.launch_command(specification)
    assert launch.index("flock -x 9") < launch.index(val.check_shell()) < launch.index("docker run ")
    assert "s4d-migval-12345678-0" in launch and "s4d.migval.spec" in launch
    ast.parse(val.PROBE)  # Parse the embedded code, never execute/import it.
    assert "start_new_session=True" in val.PROBE and "sleep 600" in val.PROBE


@pytest.mark.parametrize("image", ["latest", "s4d:latest", "sha256:abc", IMAGE + ";false"])
def test_image_must_be_pinned(image):
    with pytest.raises(ValueError, match="pinned"):
        val.container_spec(BOOT, 0, allowed()["instances"][0], mig_uuid(0, 1), image)


@pytest.mark.parametrize("path", ["/tmp/release", val.ROOT + "/../foreign", val.ROOT + "/a,b", val.ROOT + "/x\ny"])
def test_release_path_is_bounded(path):
    with pytest.raises(ValueError, match="release code"):
        spec(release_code=path)


def test_exact_ids_and_self_excluding_pgrep():
    with pytest.raises(ValueError, match="exact container ID"):
        val.exact_command("remove", "s4d-migval-12345678-0")
    command = val.survivor_command(spec()["marker"])
    assert "[s]4d-migval-child-" in command and spec()["marker"] not in command
    assert val.parse_survivors("") == [] and val.parse_survivors("123\n456\n") == [123, 456]
    with pytest.raises(ValueError):
        val.parse_survivors("permission denied\n")


@pytest.mark.parametrize("failure", ["passed", "device_count", "parent_uuid", "extra_mig", "20gib", "24gib",
                                    "egl", "frame", "other", "child", "missing"])
def test_probe_failures(failure):
    specification = spec()
    result = probe(specification)
    checks = result["checks"]
    if failure == "passed":
        result["passed"] = False
    elif failure == "device_count":
        checks["device_count"] = 2
    elif failure == "parent_uuid":
        checks["cuda_uuid"] = FOREIGN[4:]
    elif failure == "extra_mig":
        checks["mig_uuids"].append(specification["other"])
    elif failure == "20gib":
        checks["allocation_20gib"] = False
    elif failure == "24gib":
        checks["allocation_24gib_rejected"] = False
    elif failure == "egl":
        checks["egl"]["cuda_ordinal"] = 1
    elif failure == "frame":
        checks["frame"]["mean"] = 0
    elif failure == "other":
        checks["other_instance"]["device_count"] = 1
    elif failure == "child":
        checks["detached_child"]["detached"] = False
    elif failure == "missing":
        del checks["egl"]
    with pytest.raises(ValueError, match="probe failed"):
        val.parse_probe(val.PROBE_PREFIX + json.dumps(result), specification)


def test_probe_success_and_missing_duplicate_records():
    specification = spec()
    raw = val.PROBE_PREFIX + json.dumps(probe(specification)) + "\n"
    assert val.parse_probe("library log\n" + raw, specification)["passed"]
    for bad in ("", "library log", raw + raw):
        with pytest.raises(ValueError, match="exactly one"):
            val.parse_probe(bad, specification)


@pytest.mark.parametrize("failure", ["foreign", "extra_device", "pid", "rw", "mount", "image", "command"])
def test_inspect_rejects_wrong_identity_or_confinement(failure):
    specification, cid = spec(), "f" * 64
    doc = inspection(specification, cid)
    if failure == "foreign":
        doc["Config"]["Labels"]["s4d.owner"] = "other-user"
    elif failure == "extra_device":
        doc["HostConfig"]["DeviceRequests"][0]["DeviceIDs"].append("nvidia.com/gpu=" + FOREIGN)
    elif failure == "pid":
        doc["HostConfig"]["PidMode"] = "private"
    elif failure == "rw":
        doc["HostConfig"]["ReadonlyRootfs"] = False
    elif failure == "mount":
        doc["Mounts"] = [{"Type": "bind", "Source": "/", "Destination": "/host", "RW": True}]
    elif failure == "image":
        doc["Config"]["Image"] = "latest"
    elif failure == "command":
        doc["Config"]["Cmd"] = ["-c", "pass"]
    with pytest.raises(ValueError):
        val.parse_inspect(json.dumps([doc]), cid, specification)


def test_hold_in_force_prevents_discovery_and_launch(allowlist, tmp_path):
    out = tmp_path / "report.json"
    client = FakeClient(out, held=True)
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert not report["passed"] and "hold in force" in report["errors"][0]
    assert len(client.calls) == 1 and not report["instances"]
    assert not json.loads(out.read_text())["passed"]


def test_instances_validate_all_sequentially_and_archive_before_exact_removal(allowlist, tmp_path):
    out = tmp_path / "report.json"
    client = FakeClient(out, release_code=val.ROOT + "/releases/abc")
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out,
                           release_code=val.ROOT + "/releases/abc")
    assert report["passed"], report["errors"]
    assert len(report["instances"]) == len(client.removed) == 16 and not client.live
    assert all(item["passed"] and item["framebuffer_samples"][-1]["used_mib"] <= 256 for item in report["instances"])
    assert all(item["logs"]["stderr"] == "synthetic warning\n" for item in report["instances"])
    assert all(item["spec"]["uuid"] in {i["uuid"] for i in inv.load_allowlist(allowlist)} for item in report["instances"])
    assert json.loads(out.read_text())["passed"]
    assert val.instances(client, allowlist=allowlist, image=IMAGE, out=out,
                         release_code=val.ROOT + "/releases/abc")["passed"]
    assert len(client.removed) == 32  # Safe repeat after completed cleanup.


@pytest.mark.parametrize("failure, expected", [("survivor", "surviving process after"),
                                              ("probe_failure", "probe failed"),
                                              ("wait_failure", "wait failed"),
                                              ("preflight_survivor", "previous validation"),
                                              ("foreign_process", "foreign processes")])
def test_instances_failures_stop_further_launches(allowlist, tmp_path, failure, expected):
    out = tmp_path / "report.json"
    client = FakeClient(out, **{failure: True})
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert not report["passed"] and len(report["instances"]) == 1
    assert expected in " ".join(report["instances"][0]["errors"])
    assert not client.live
    if failure in ("preflight_survivor", "foreign_process"):
        assert not client.removed
    else:
        assert len(client.removed) == 1
        assert report["instances"][0]["logs"] and report["instances"][0]["inspect"]


def test_instances_refuse_drift_before_any_container(allowlist, tmp_path):
    out = tmp_path / "report.json"
    client = FakeClient(out)
    client.evidence["listing"] = client.evidence["listing"].replace(mig_uuid(0, 0), mig_uuid(10, 0))
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert not report["passed"] and not report["instances"] and not client.live
    assert "discovery failed" in report["errors"][0]


def test_reconcile_completed_container_after_interruption(allowlist, tmp_path):
    out = tmp_path / "report.json"
    client = FakeClient(out)
    specification = spec()
    cid = hashlib.sha256(specification["name"].encode()).hexdigest()
    client.live[cid] = {"spec": specification, "running": False}
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert report["passed"] and report["instances"][0]["container_id"] == cid
    assert len(client.removed) == 16 and not client.live


def test_hold_rearmed_between_snapshot_and_launch(allowlist, tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    client = FakeClient(out)
    checked_ssh = client.checked_ssh

    def rearmed(command, **kwargs):
        if "\nelse\ndocker run " in command:
            assert val.check_shell() in command
            raise RuntimeError(f"Remote operation exited {val.EXIT_HELD}: launch held")
        return checked_ssh(command, **kwargs)

    monkeypatch.setattr(client, "checked_ssh", rearmed)
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert not report["passed"] and "launch held" in report["instances"][0]["errors"][0]
    assert not client.live and not client.removed


def test_foreign_container_never_stopped_or_removed(allowlist, tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    client = FakeClient(out)
    checked_ssh = client.checked_ssh

    def foreign(command, **kwargs):
        result = checked_ssh(command, **kwargs)
        if command.startswith("docker inspect "):
            doc = json.loads(result.stdout)
            doc[0]["Config"]["Labels"]["s4d.owner"] = "foreign-owner"
            result.stdout = json.dumps(doc)
        return result

    monkeypatch.setattr(client, "checked_ssh", foreign)
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=out)
    assert not report["passed"] and len(client.live) == 1 and not client.removed
    assert not any(cmd.startswith(("docker stop", "docker rm")) for cmd, _ in client.calls)


def test_allowlist_change_between_reads_is_refused(allowlist):
    reviewed = inv.load_allowlist(allowlist)
    document = allowed()
    document["instances"][0]["uuid"] = mig_uuid(10, 0)
    allowlist.write_text(yaml.safe_dump(document))
    client = FakeClient()
    report = val.discover(client, allowlist, reviewed=reviewed)
    assert not report["passed"] and "allowlist changed" in report["errors"][0] and not client.calls


def test_partial_allowlist_refused_without_remote_calls(allowlist, tmp_path):
    document = allowed()
    document["instances"].pop()
    allowlist.write_text(yaml.safe_dump(document))
    client = FakeClient()
    report = val.instances(client, allowlist=allowlist, image=IMAGE, out=tmp_path / "report.json")
    assert not report["passed"] and "all 16" in report["errors"][0] and not client.calls


def test_framebuffer_parsing_and_deadline(monkeypatch):
    instance = allowed()["instances"][0]
    assert val.parse_memory(render_query(), instance)["idle"]
    assert not val.parse_memory(render_query(used=257), instance)["idle"]
    assert not val.parse_memory(render_query(processes=[(3, 999)]), instance)["idle"]
    with pytest.raises(ValueError):
        val.parse_memory(render_query().replace("Used : 67 MiB", "Used : N/A"), instance)
    client = FakeClient(foreign_process=True)
    ticks = iter([0, 0, 1, 2, 31])
    monkeypatch.setattr(val.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(val.time, "sleep", lambda _: None)
    report = {}
    with pytest.raises(RuntimeError, match="within 30 s"):
        val.wait_idle(client, report, instance, BOOT)
    assert report["framebuffer_samples"][0]["idle"] is False


def test_cli_reports_failures_and_summary_without_real_client(tmp_path, capsys):
    out = tmp_path / "report.json"
    client = FakeClient()
    client.evidence["cdi_mtime"] = "99"
    assert val.main(["discover", "--out", str(out)], client_factory=lambda: client) == 1
    assert "FAIL discover" in capsys.readouterr().out
    assert val.main(["report", str(out)], client_factory=lambda: pytest.fail("report must stay local")) == 1
    assert "root must regenerate" in capsys.readouterr().out
    assert val.main(["discover", "--out", str(out)], client_factory=FakeClient) == 0
    assert val.main(["report", str(out)]) == 0


def test_client_failure_still_creates_atomic_report(tmp_path):
    out = tmp_path / "report.json"

    def unavailable():
        raise RuntimeError("synthetic transport unavailable")

    assert val.main(["discover", "--out", str(out)], client_factory=unavailable) == 1
    report = json.loads(out.read_text())
    assert not report["passed"] and "synthetic transport unavailable" in report["errors"][0]
    assert list(tmp_path.iterdir()) == [out]
