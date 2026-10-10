"""Package verified installed native files without importing or building native code."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/home/ws/ws/splatter4d/runs/remote/runtime")
FORMAT = "s4d-native-runtime-v1"
BUNDLE_FILES = {
    "pyproject.toml", "uv.lock", "runtime_versions.json", "requirements.lock.txt",
    "package_runtime.py", "Dockerfile", "native_payload.tar",
}
NATIVE = {
    "gsplat": {
        "version": "1.5.3",
        "module": "gsplat",
        "url": "https://github.com/nerfstudio-project/gsplat.git",
        "revision": "d28ee0c42264a540b8b17388cf793ad31bc20b53",
        "shared_libraries": {
            "gsplat/csrc.so": "8cc4919482a99543528e3d0ab76ce7b5ed0bddb4e856b40f49dd74b877bad025",
            "gsplat/experimental/render/kernels/csrc.so": "81ca50f714d6de0d3f0677634bc2d8c79506aa6c5c3a6421cc2ec9fc8101c840",
        },
    },
    "fused-ssim": {
        "version": "1.0.0",
        "module": "fused_ssim",
        "url": "https://github.com/rahul-goel/fused-ssim",
        "revision": "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38",
        "shared_libraries": {
            "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so":
                "bb515dd65dfc547be96eb179d1e8e688b413bc1eb5fbc5e7fc93e7dac864fac2",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or str(path) != value or "\\" in value:
        raise ValueError(f"unsafe payload path: {value!r}")
    return path


def checked_file(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    if not path.is_file() or any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError(f"missing or symlinked input: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"input escapes its root: {relative}")
    return path


def validate_base_image(value: str) -> None:
    if not re.fullmatch(r"(?:docker\.io/library/)?ubuntu:22\.04@sha256:[a-f0-9]{64}", value):
        raise ValueError("BASE_IMAGE must be Ubuntu 22.04 with an immutable sha256 digest")


def validate_lock(text: str, specs: dict = NATIVE) -> None:
    blocks = re.split(r"(?m)^\[\[package\]\]\s*$", text)[1:]
    for name, spec in specs.items():
        matching = [block for block in blocks if re.search(rf'^name = "{re.escape(name)}"$', block, re.MULTILINE)]
        if len(matching) != 1:
            raise ValueError(f"lock must contain exactly one {name} package")
        expected = f'{spec["url"]}?rev={spec["revision"]}#{spec["revision"]}'
        if not re.search(rf'^version = "{re.escape(spec["version"])}"$', matching[0], re.MULTILINE):
            raise ValueError(f"locked {name} version differs")
        if not re.search(rf'^source = \{{ git = "{re.escape(expected)}" \}}$', matching[0], re.MULTILINE):
            raise ValueError(f"locked {name} Git identity differs")


def allowed_member(relative: str, spec: dict) -> bool:
    path = safe_relative(relative)
    info = f'{spec["module"]}-{spec["version"]}.dist-info'
    return path.parts[0] in (spec["module"], info) or relative in spec["shared_libraries"]


def inspect_distribution(site: Path, name: str, spec: dict) -> dict:
    info = f'{spec["module"]}-{spec["version"]}.dist-info'
    record_name = f"{info}/RECORD"
    record = checked_file(site, record_name).read_bytes()
    files = {}
    for row in csv.reader(io.StringIO(record.decode("utf-8"))):
        if len(row) != 3 or not allowed_member(row[0], spec) or row[0] in files:
            raise ValueError(f"invalid or duplicate {name} RECORD entry: {row}")
        relative, expected, size = row
        path = checked_file(site, relative)
        digest = sha256_file(path)
        if relative == record_name:
            if expected or size:
                raise ValueError("RECORD must have an unhashed self entry")
        else:
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
            if expected != f"sha256={encoded}" or size != str(path.stat().st_size):
                raise ValueError(f"RECORD hash/size mismatch: {relative}")
        files[relative] = {"sha256": digest, "size": path.stat().st_size}
    if record_name not in files:
        raise ValueError("RECORD omits itself")
    for required in ("METADATA", "WHEEL", "direct_url.json"):
        if f"{info}/{required}" not in files:
            raise ValueError(f"RECORD omits {required}")
    metadata = Parser().parsestr(checked_file(site, f"{info}/METADATA").read_text())
    if metadata["Name"] != name or metadata["Version"] != spec["version"]:
        raise ValueError(f"installed {name} identity differs")
    wheel = Parser().parsestr(checked_file(site, f"{info}/WHEEL").read_text())
    if wheel.get_all("Tag") != ["cp310-cp310-linux_x86_64"] or wheel["Root-Is-Purelib"] != "false":
        raise ValueError(f"unexpected {name} native ABI")
    origin = json.loads(checked_file(site, f"{info}/direct_url.json").read_text())
    expected_origin = {
        "url": spec["url"],
        "vcs_info": {"vcs": "git", "commit_id": spec["revision"], "requested_revision": spec["revision"]},
    }
    if origin != expected_origin:
        raise ValueError(f"installed {name} Git provenance differs")
    actual_libraries = {path: entry["sha256"] for path, entry in files.items() if path.endswith(".so")}
    if actual_libraries != spec["shared_libraries"]:
        raise ValueError(f"{name} native binaries differ from the audited sm_120 payload")
    return {
        "version": spec["version"], "direct_url": origin, "build_generator": wheel["Generator"],
        "shared_libraries": actual_libraries, "files": files,
    }


def verify_versions(site: Path, expected: dict) -> None:
    def canonical(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    installed = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = canonical(distribution.metadata["Name"])
        if name in installed:
            raise ValueError(f"duplicate installed distribution: {name}")
        installed[name] = distribution.version
    for name, version in expected.items():
        if installed.get(canonical(name)) != version:
            raise ValueError(
                f"runtime version mismatch for {name}: expected {version}, found {installed.get(canonical(name))}"
            )


def payload_entries(packages: dict) -> dict:
    entries = {}
    for name, package in packages.items():
        for relative, identity in package["files"].items():
            if not allowed_member(relative, NATIVE[name]) or relative in entries:
                raise ValueError(f"unexpected/duplicate native payload member: {relative}")
            entries[relative] = identity
    return entries


def write_payload(site: Path, packages: dict, output: Path) -> None:
    entries = payload_entries(packages)
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative, identity in sorted(entries.items()):
            data = checked_file(site, relative).read_bytes()
            if len(data) != identity["size"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError(f"input changed after RECORD verification: {relative}")
            member = tarfile.TarInfo(relative)
            member.size, member.mode = len(data), 0o644
            member.uid = member.gid = member.mtime = 0
            archive.addfile(member, io.BytesIO(data))


def verify_payload(path: Path, packages: dict) -> None:
    entries, seen = payload_entries(packages), set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile() or member.name not in entries or member.name in seen:
                raise ValueError(f"unexpected native archive member: {member.name}")
            seen.add(member.name)
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            expected = entries[member.name]
            if member.size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"native archive hash/size mismatch: {member.name}")
    if seen != entries.keys():
        raise ValueError("native archive omits manifest members")


def install_payload(path: Path, packages: dict, site: Path) -> None:
    verify_payload(path, packages)
    site = site.resolve(strict=True)
    entries = payload_entries(packages)
    for relative in entries:
        target = site / relative
        if target.exists() or target.is_symlink() or not target.resolve().is_relative_to(site):
            raise ValueError(f"native install would overwrite or escape: {relative}")
        if any(parent.is_symlink() for parent in target.parents if parent.is_relative_to(site)):
            raise ValueError(f"native install parent is symlinked: {relative}")
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != entries[member.name]["sha256"]:
                raise ValueError(f"archive changed during install: {member.name}")
            target = site / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(data)
            target.chmod(0o644)


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads(checked_file(bundle, "manifest.json").read_text())
    if manifest.get("format") != FORMAT or set(manifest.get("files", {})) != BUNDLE_FILES:
        raise ValueError("unexpected runtime bundle format/files")
    for relative, identity in manifest["files"].items():
        path = checked_file(bundle, relative)
        if path.stat().st_size != identity["size"] or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"runtime bundle hash/size mismatch: {relative}")
    validate_lock((bundle / "uv.lock").read_text())
    runtime = json.loads((bundle / "runtime_versions.json").read_text())
    if manifest["runtime_versions"] != runtime or set(manifest["packages"]) != set(NATIVE):
        raise ValueError("runtime manifest identity differs")
    for name, spec in NATIVE.items():
        package = manifest["packages"][name]
        if package["version"] != spec["version"] or package["shared_libraries"] != spec["shared_libraries"]:
            raise ValueError(f"unexpected {name} audited binary identity")
        if package["direct_url"] != {
            "url": spec["url"],
            "vcs_info": {"vcs": "git", "commit_id": spec["revision"], "requested_revision": spec["revision"]},
        }:
            raise ValueError(f"unexpected {name} source identity")
        actual = {relative: entry["sha256"] for relative, entry in package["files"].items() if relative.endswith(".so")}
        if actual != spec["shared_libraries"]:
            raise ValueError(f"unexpected {name} native members")
    verify_payload(bundle / "native_payload.tar", manifest["packages"])
    return manifest


def package_bundle(repo: Path, site: Path, output: Path, uv: str) -> dict:
    output = output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"package output must stay inside {OUTPUT_ROOT}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a nonempty runtime bundle")
    runtime_path = repo / "docs/runtime_versions.json"
    runtime = json.loads(runtime_path.read_text())
    if platform.python_version() != runtime["python"]:
        raise ValueError("packaging interpreter differs from the recorded Python version")
    actual_uv = subprocess.run([uv, "--version"], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    if actual_uv != runtime["uv"]:
        raise ValueError("packaging uv differs from the recorded version")
    verify_versions(site, runtime["versions"])
    lock_hash = sha256_file(repo / "uv.lock")
    validate_lock((repo / "uv.lock").read_text())
    packages = {name: inspect_distribution(site, name, spec) for name, spec in NATIVE.items()}
    exported = subprocess.run(
        [uv, "export", "--project", str(repo), "--locked", "--extra", "dev", "--format", "requirements.txt",
         "--no-header", "--no-annotate", "--no-emit-project", "--no-emit-package", "gsplat",
         "--no-emit-package", "fused-ssim"],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    if sha256_file(repo / "uv.lock") != lock_hash:
        raise ValueError("dependency export changed the pinned lock")
    if "git+" in exported or re.search(r"(?m)^(?:-e |gsplat\b|fused[-_]ssim\b)", exported):
        raise ValueError("lock export contains source/editable native requirements")
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in (
        (repo / "pyproject.toml", "pyproject.toml"), (repo / "uv.lock", "uv.lock"),
        (runtime_path, "runtime_versions.json"), (repo / "scripts/remote/package_runtime.py", "package_runtime.py"),
        (repo / "scripts/remote/Dockerfile", "Dockerfile"),
    ):
        digest = sha256_file(source)
        shutil.copyfile(source, output / relative)
        if sha256_file(output / relative) != digest:
            raise ValueError(f"bundle input changed while copied: {relative}")
    (output / "requirements.lock.txt").write_text(exported)
    write_payload(site, packages, output / "native_payload.tar")
    manifest = {
        "format": FORMAT, "runtime_versions": runtime, "packages": packages,
        "native_compilation_allowed": False, "cuda_architecture": "sm_120",
        "native_architecture_evidence": "Audited installed .so SHA256 identities; no new compilation",
        "runtime_setuptools": runtime["versions"]["setuptools"],
        "files": {
            relative: {"sha256": sha256_file(output / relative), "size": (output / relative).stat().st_size}
            for relative in sorted(BUNDLE_FILES)
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_bundle(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("package")
    package.add_argument(
        "--site-packages", type=Path, default=Path("/home/ws/ws/splatter4d/.venv/lib/python3.10/site-packages")
    )
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--uv", default="uv")
    for command in ("verify", "install"):
        sub.add_parser(command).add_argument("--bundle", type=Path, required=True)
    sub.add_parser("check-base").add_argument("--base-image", required=True)
    args = parser.parse_args()
    if args.command == "check-base":
        validate_base_image(args.base_image)
    elif args.command == "package":
        package_bundle(REPO, args.site_packages, args.output, args.uv)
        print(f"Verified native runtime bundle: {args.output}")
    else:
        manifest = verify_bundle(args.bundle)
        if args.command == "install":
            if Path(sys.prefix) != Path("/opt/s4d/.venv"):
                raise ValueError("native install is restricted to the immutable image venv /opt/s4d/.venv")
            runtime = manifest["runtime_versions"]
            if platform.python_version() != runtime["python"]:
                raise ValueError("image Python version differs")
            site = Path(sysconfig.get_path("purelib"))
            install_payload(args.bundle / "native_payload.tar", manifest["packages"], site)
            verify_versions(site, runtime["versions"])
            for name, spec in NATIVE.items():
                if inspect_distribution(site, name, spec) != manifest["packages"][name]:
                    raise ValueError(f"installed {name} differs from the packaged payload")
        print(f"Verified runtime manifest SHA256: {sha256_file(args.bundle / 'manifest.json')}")


if __name__ == "__main__":
    main()
