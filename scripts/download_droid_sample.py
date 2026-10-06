"""Pinned, minimal, data-only download. Invoke with python -I outside downloads.

Exactly one selected episode is saved. Flow/depth archive streams stop after
that regular tar member. No archive path extraction or downloaded code runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import guard_gpus

GPU_MAPPING = guard_gpus()

from s4d.data import POINTWORLD_ROOT, writable_path  # noqa: E402

REVISION = "dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f"
EPISODE = "AUTOLab+0d4edc83+2023-10-21-19h-07m-04s"
BASE = f"https://huggingface.co/datasets/nvidia/PointWorld-DROID/resolve/{REVISION}"
HASHES = {
    "flow": "a07623861b8443841509af8189dd30b291669236c721fae44dfc4c02f3a385be",
    "depth": "c1c7e10ce6c546f3d5672ef423ffccb4837a910457824ecd1a0522a799f62aca",
    "camera_archive": "2640e62bf700e92312b5e328daf0e2e8a2f0cfa4b8aa3662b029d949b2893d7c",
    "license": "4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def curl_arguments(url):
    return [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--proto",
        "=https",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
        url,
    ]


def copy_member(stream, expected: str, destination: Path) -> dict:
    """The output filename is caller-owned, never taken from archive paths."""
    scanned = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            scanned += member.size
            if member.name != expected:
                continue
            if not member.isfile() or member.size <= 0 or member.size > 100_000_000:
                raise ValueError("selected member is not a bounded regular data file")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("selected member cannot be read")
            with destination.open("xb") as out:
                shutil.copyfileobj(source, out, length=1 << 20)
            if destination.stat().st_size != member.size:
                raise ValueError("truncated selected member")
            return {
                "member": expected,
                "size_bytes": member.size,
                "sha256": sha256(destination),
                "scanned_raw_bytes": scanned,
            }
    raise FileNotFoundError(f"selected episode absent: {expected}")


def stop_process(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def stream_episode(url, expected, destination):
    curl = subprocess.Popen(curl_arguments(url), stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE)
    curl.stdout.close()
    try:
        record = copy_member(zstd.stdout, expected, destination)
        record["url"] = url
        return record
    finally:
        zstd.stdout.close()
        stop_process(zstd)
        stop_process(curl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="new empty sample root, not current working directory")
    args = parser.parse_args()
    root = writable_path(args.root, POINTWORLD_ROOT)
    if root == Path.cwd().resolve():
        raise ValueError("downloads must not be the current working directory")
    if not root.parent.is_dir():
        raise FileNotFoundError("create/check the approved parent first")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("download root must be new or empty")
    if shutil.disk_usage(root.parent).free < 2_000_000_000:
        raise OSError("less than 2 GB free for selected sample")
    root.mkdir(exist_ok=True)
    # Each downloaded archive has its own fresh directory. Small metadata files
    # use separate new leaf directories, then are copied to compatibility paths.
    for name in ("metadata_download", "camera_archive_download", "camera_episode", "flow_episode", "depth_episode"):
        (root / name).mkdir()
    small = [
        ("card", "README.md"),
        ("license", "LICENSE.pdf"),
        ("manifest", "droid/flows-fs-optimized/_shards_manifest.json"),
    ]
    for name, remote in small:
        leaf = root / "metadata_download" / name
        leaf.mkdir()
        destination = leaf / Path(remote).name
        subprocess.run([*curl_arguments(f"{BASE}/{remote}"), "--output", str(destination)], check=True)
        if name == "license" and sha256(destination) != HASHES["license"]:
            raise ValueError("pinned license checksum mismatch")
        if name == "license":
            shutil.copyfile(destination, root / "metadata_download/LICENSE.pdf")
    archive = root / "camera_archive_download/package.tar.zst.part-0000"
    subprocess.run(
        [*curl_arguments(f"{BASE}/droid/cameras/package.tar.zst.part-0000"), "--output", str(archive)], check=True
    )
    if sha256(archive) != HASHES["camera_archive"]:
        raise ValueError("camera archive checksum mismatch")
    zstd = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    try:
        record = copy_member(
            zstd.stdout, f"droid/cameras/{EPISODE}_cameras.json", root / "camera_episode" / f"{EPISODE}_cameras.json"
        )
        (root / "camera_episode/download_record.json").write_text(json.dumps(record, indent=2))
    finally:
        zstd.stdout.close()
        stop_process(zstd)
    for kind, remote, folder in (
        ("flow", "droid/flows-fs-optimized/shard-000000/package.tar.zst.part-0000", "flows-fs-optimized"),
        ("depth", "droid/depth_320x180/package.tar.zst.part-0000", "depth_320x180"),
    ):
        basename = f"{EPISODE}_{'flows' if kind == 'flow' else 'depth'}.h5"
        record = stream_episode(f"{BASE}/{remote}", f"droid/{folder}/{basename}", root / f"{kind}_episode" / basename)
        if record["sha256"] != HASHES[kind]:
            raise ValueError(f"{kind} episode checksum mismatch")
        (root / f"{kind}_episode/download_record.json").write_text(json.dumps(record, indent=2))
    # An ordinal is merely a search hint. Converter independently rereads RLDS,
    # checks both paths and all per-clip robot states before accepting it.
    hint = {
        "episode_id": EPISODE,
        "rlds_ordinal": 21512,
        "rlds_split": "train",
        "camera_mapping": {"22008760": "exterior_image_1_left", "24400334": "exterior_image_2_left"},
        "source": "verified selected sample canonical ordinal; independently rechecked by converter",
    }
    (root / "rlds_match_hint.json").write_text(json.dumps(hint, indent=2))
    print(json.dumps({"sample": str(root), "revision": REVISION, "episode": EPISODE}, indent=2))


if __name__ == "__main__":
    main()
