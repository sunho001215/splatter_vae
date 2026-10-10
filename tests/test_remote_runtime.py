"""Synthetic byte fixtures test packaging boundaries, not native execution or experiment acceptance."""

from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_runtime", REPO / "scripts/remote/package_runtime.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_record(site, info):
    record = site / info / "RECORD"
    rows = []
    for path in sorted(site.rglob("*")):
        if path.is_file() and path != record:
            data = path.read_bytes()
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest(data))).rstrip(b"=").decode()
            rows.append([str(path.relative_to(site)), f"sha256={encoded}", str(len(data))])
    rows.append([str(record.relative_to(site)), "", ""])
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    record.write_text(stream.getvalue())


@pytest.fixture
def installed_bytes(tmp_path):
    site = tmp_path / "site"
    spec = copy.deepcopy(TOOL.NATIVE["gsplat"])
    spec["shared_libraries"] = {"gsplat/csrc.so": digest(b"synthetic native bytes, never loaded")}
    info = "gsplat-1.5.3.dist-info"
    files = {
        "gsplat/__init__.py": b'__version__ = "1.5.3"\n',
        "gsplat/csrc.so": b"synthetic native bytes, never loaded",
        f"{info}/METADATA": b"Metadata-Version: 2.4\nName: gsplat\nVersion: 1.5.3\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: setuptools (84.0.0)\nRoot-Is-Purelib: false\n"
                           b"Tag: cp310-cp310-linux_x86_64\n",
        f"{info}/INSTALLER": b"uv",
        f"{info}/REQUESTED": b"",
        f"{info}/direct_url.json": json.dumps({
            "url": spec["url"],
            "vcs_info": {"vcs": "git", "commit_id": spec["revision"], "requested_revision": spec["revision"]},
        }).encode(),
    }
    for relative, data in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_record(site, info)
    return site, spec, info


def lock_text():
    return "\n".join(
        f'[[package]]\nname = "{name}"\nversion = "{spec["version"]}"\n'
        f'source = {{ git = "{spec["url"]}?rev={spec["revision"]}#{spec["revision"]}" }}\n'
        for name, spec in TOOL.NATIVE.items()
    )


def test_packager_has_only_standard_library_imports():
    source = (REPO / "scripts/remote/package_runtime.py").read_text()
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots <= {
        "__future__", "argparse", "base64", "csv", "hashlib", "importlib", "io", "json", "platform",
        "re", "shutil", "subprocess", "sys", "sysconfig", "tarfile", "email", "pathlib",
    }
    assert "torch" not in roots


def test_record_inspection_preserves_build_generator_and_git_origin(installed_bytes):
    site, spec, info = installed_bytes
    result = TOOL.inspect_distribution(site, "gsplat", spec)
    assert result["build_generator"] == "setuptools (84.0.0)"
    assert result["direct_url"]["vcs_info"]["commit_id"] == spec["revision"]
    assert result["files"][f"{info}/RECORD"]["sha256"] == TOOL.sha256_file(site / info / "RECORD")
    assert result["shared_libraries"] == spec["shared_libraries"]


def test_payload_is_deterministic_and_preserves_every_installed_byte(installed_bytes, tmp_path):
    site, spec, _ = installed_bytes
    package = TOOL.inspect_distribution(site, "gsplat", spec)
    first, second = tmp_path / "first.tar", tmp_path / "second.tar"
    TOOL.write_payload(site, {"gsplat": package}, first)
    for path in site.rglob("*"):
        if path.is_file():
            os.utime(path, (1700000000, 1700000000))
    TOOL.write_payload(site, {"gsplat": package}, second)
    assert first.read_bytes() == second.read_bytes()
    target = tmp_path / "installed"
    target.mkdir()
    TOOL.install_payload(first, {"gsplat": package}, target)
    assert TOOL.inspect_distribution(target, "gsplat", spec) == package
    for relative in package["files"]:
        assert (target / relative).read_bytes() == (site / relative).read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        TOOL.install_payload(first, {"gsplat": package}, target)


def test_inspection_rejects_changed_record_input(installed_bytes):
    site, spec, _ = installed_bytes
    (site / "gsplat/__init__.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="RECORD hash/size mismatch"):
        TOOL.inspect_distribution(site, "gsplat", spec)


def test_inspection_rejects_different_binary_even_with_consistent_record(installed_bytes):
    site, spec, info = installed_bytes
    (site / "gsplat/csrc.so").write_bytes(b"another internally consistent binary")
    write_record(site, info)
    with pytest.raises(ValueError, match="audited sm_120"):
        TOOL.inspect_distribution(site, "gsplat", spec)


def test_inspection_rejects_changed_git_provenance(installed_bytes):
    site, spec, info = installed_bytes
    path = site / info / "direct_url.json"
    origin = json.loads(path.read_text())
    origin["vcs_info"]["commit_id"] = "f" * 40
    path.write_text(json.dumps(origin))
    write_record(site, info)
    with pytest.raises(ValueError, match="Git provenance"):
        TOOL.inspect_distribution(site, "gsplat", spec)


def test_inspection_rejects_wrong_abi(installed_bytes):
    site, spec, info = installed_bytes
    wheel = site / info / "WHEEL"
    wheel.write_text(wheel.read_text().replace("cp310-cp310", "cp311-cp311"))
    write_record(site, info)
    with pytest.raises(ValueError, match="native ABI"):
        TOOL.inspect_distribution(site, "gsplat", spec)


@pytest.mark.parametrize("relative", ["", ".", "../secret", "/secret", "gsplat//file", "gsplat/../file", "gsplat\\file"])
def test_payload_names_reject_noncanonical_paths(relative):
    with pytest.raises(ValueError, match="unsafe payload path"):
        TOOL.safe_relative(relative)


@pytest.mark.parametrize("mutation", ["duplicate", "escape", "unhashed"])
def test_inspection_rejects_invalid_record_entries(installed_bytes, mutation):
    site, spec, info = installed_bytes
    path = site / info / "RECORD"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "escape":
        rows.append(["../secret", "sha256=x", "1"])
    else:
        rows[0][1] = ""
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    path.write_text(output.getvalue())
    with pytest.raises(ValueError):
        TOOL.inspect_distribution(site, "gsplat", spec)


def test_inspection_rejects_input_symlink(installed_bytes, tmp_path):
    site, spec, _ = installed_bytes
    target = site / "gsplat/csrc.so"
    outside = tmp_path / "external.so"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinked"):
        TOOL.inspect_distribution(site, "gsplat", spec)


def test_payload_rechecks_inputs_when_written(installed_bytes, tmp_path):
    site, spec, _ = installed_bytes
    package = TOOL.inspect_distribution(site, "gsplat", spec)
    (site / "gsplat/csrc.so").write_bytes(b"changed after inspection")
    with pytest.raises(ValueError, match="changed after RECORD verification"):
        TOOL.write_payload(site, {"gsplat": package}, tmp_path / "payload.tar")


@pytest.mark.parametrize("kind", ["duplicate", "symlink", "escape", "missing", "changed"])
def test_payload_verification_rejects_invalid_archive(installed_bytes, tmp_path, kind):
    site, spec, _ = installed_bytes
    packages = {"gsplat": TOOL.inspect_distribution(site, "gsplat", spec)}
    source = tmp_path / "source.tar"
    TOOL.write_payload(site, packages, source)
    bad = tmp_path / "bad.tar"
    with tarfile.open(source, "r:") as original, tarfile.open(bad, "w") as archive:
        members = original.getmembers()
        for index, member in enumerate(members):
            data = original.extractfile(member).read()
            if index == 0 and kind == "missing":
                continue
            if index == 0 and kind == "changed":
                data = b"x" * len(data)
            archive.addfile(member, io.BytesIO(data))
        if kind in ("duplicate", "symlink", "escape"):
            extra = tarfile.TarInfo(members[0].name if kind != "escape" else "../escape")
            if kind == "symlink":
                extra.type, extra.linkname = tarfile.SYMTYPE, "/outside"
            archive.addfile(extra)
    target = tmp_path / "destination"
    target.mkdir()
    with pytest.raises(ValueError):
        TOOL.install_payload(bad, packages, target)
    assert not list(target.iterdir())


def test_install_rejects_symlinked_parent_before_writes(installed_bytes, tmp_path):
    site, spec, _ = installed_bytes
    packages = {"gsplat": TOOL.inspect_distribution(site, "gsplat", spec)}
    source = tmp_path / "source.tar"
    TOOL.write_payload(site, packages, source)
    target, outside = tmp_path / "destination", tmp_path / "outside"
    target.mkdir()
    outside.mkdir()
    (target / "gsplat").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escape|symlinked"):
        TOOL.install_payload(source, packages, target)
    assert not list(outside.iterdir())


def test_lock_accepts_only_both_pinned_native_git_identities():
    TOOL.validate_lock(lock_text())
    with pytest.raises(ValueError, match="Git identity"):
        TOOL.validate_lock(lock_text().replace(TOOL.NATIVE["gsplat"]["revision"], "e" * 40))
    with pytest.raises(ValueError, match="exactly one"):
        TOOL.validate_lock(lock_text() + lock_text())
    with pytest.raises(ValueError, match="version differs"):
        TOOL.validate_lock(lock_text().replace('version = "1.5.3"', 'version = "1.5.4"'))


@pytest.mark.parametrize("base", ["", "ubuntu:22.04", "ubuntu:22.04@sha256:abc", "ubuntu:24.04@sha256:" + "a" * 64])
def test_base_refuses_mutable_or_wrong_ubuntu_image(base):
    with pytest.raises(ValueError, match="immutable"):
        TOOL.validate_base_image(base)


def test_base_accepts_an_explicit_ubuntu_digest():
    TOOL.validate_base_image("ubuntu:22.04@sha256:" + "a" * 64)
    TOOL.validate_base_image("docker.io/library/ubuntu:22.04@sha256:" + "b" * 64)


def test_runtime_versions_are_metadata_only_and_exact(installed_bytes):
    site, _, _ = installed_bytes
    TOOL.verify_versions(site, {"gsplat": "1.5.3"})
    with pytest.raises(ValueError, match="runtime version mismatch"):
        TOOL.verify_versions(site, {"gsplat": "1.5.4"})
    with pytest.raises(ValueError, match="runtime version mismatch"):
        TOOL.verify_versions(site, {"torch": "2.10.0+cu129"})


def test_package_output_cannot_escape_authorized_runtime_root(tmp_path):
    with pytest.raises(ValueError, match="output must stay inside"):
        TOOL.package_bundle(REPO, tmp_path, tmp_path / "unauthorized", "uv")
    assert not (tmp_path / "unauthorized").exists()


def test_bundle_detects_top_level_input_tampering(tmp_path):
    identities = {}
    for name in TOOL.BUNDLE_FILES:
        data = lock_text().encode() if name == "uv.lock" else b"input bytes"
        (tmp_path / name).write_bytes(data)
        identities[name] = {"sha256": digest(data), "size": len(data)}
    manifest = {"format": TOOL.FORMAT, "files": identities}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "pyproject.toml").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="runtime bundle hash/size mismatch"):
        TOOL.verify_bundle(tmp_path)


def test_docker_recipe_is_digest_gated_and_never_builds_native_sources():
    docker = (REPO / "scripts/remote/Dockerfile").read_text()
    assert docker.startswith("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    assert "ubuntu:22\\.04@sha256:[a-f0-9]{64}" in docker
    assert "uv python install 3.10.19" in docker
    assert "--no-build --no-install-package gsplat --no-install-package fused-ssim" in docker
    assert "--locked --extra dev" in docker
    assert "UV_NO_BUILD=1" in docker
    assert "6b52a47358deea1c5e173278bf46b2b489747a59ae31f2a4362ed5c6c1c269f7" in docker
    assert "USER compu" in docker and "WORKDIR /workspace/splatter4d" in docker
    assert "UV_PROJECT_ENVIRONMENT=/opt/s4d/.venv" in docker
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in docker
    assert "pip install" not in docker and "build-essential" not in docker
