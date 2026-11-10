"""Pinned official sources of the large-scale DROID data path and exactly which files the pipeline needs from each.

What the pipeline reads (docs/droid_largescale.md, s4d/data/droid/largescale/):

- DROID RLDS 1.0.1 (``gs://gresearch/robotics/droid/1.0.1``, public, not requester-pays): all 2048 TFRecord shards
  plus ``dataset_info.json`` (``shardLengths``) and ``features.json``. Used in place by ``rlds_index.py``; never
  extracted. Every shard holds at least one PointWorld episode that joins DROID-3D (41,599 of 42,936 by identity), so
  no shard subset exists. Not taken: ``droid/1.0.0`` and ``droid_100``.
- PointWorld-DROID (``nvidia/PointWorld-DROID`` @ ``dd9aaeec``, NVIDIA licence, non-commercial): flow shards
  (``flows-fs-optimized/shard-*/package.tar.zst.part-0000``, one self-contained ``tar.zst`` per shard; members and
  ``raw_member_bytes`` in ``_shards_manifest.json``), the depth package (``depth_320x180``, 252 parts of ONE ``tar.zst``
  stream without an index: sequential only) and the cameras package (one part). ``pointworld_source.py`` reads the
  extracted per-episode ``*_flows.h5``, ``*_depth.h5`` and ``*_cameras.json``. Not taken: ``droid/confidence``
  (unused), ``recover_dataset_from_parts.sh`` (a concatenation helper), ``.gitattributes``. The licence and card are
  kept next to the data (licence section 3.1).
- DROID-3D calibration (``jaibrdhn/droid_3d_extrinsics`` @ ``69db9ccf``, gated: needs a Hugging Face token of an
  approved account): ``extrinsics.jsonl``, ``camera_intrinsics.jsonl``, the card. Not taken: ``pseudolabels/``
  (CC BY-NC, unused), ``extrinsics_missing.json``, the figure.
- Hand-eye assets read by ``calibration.load_hand_eye``: DepthWorld ``depth_extras/assets/gripper2wrist_transforms.json``
  @ ``6823c13d`` and PointWorld ``real/gripper2wrist_transforms.json`` @ ``3872ec6`` (identity check).

Episode identity across the three sources: PointWorld member name = flow ``uuid`` = DROID-3D ``episode_uuid``; RLDS by
the ``<lab>/<outcome>/<date>/<folder>`` suffix of ``file_path`` (docs/droid_largescale.md section 4.1).
"""

from __future__ import annotations

import json
import re
import urllib.parse

from ops.datasets.transfer import fetch_bytes, fetch_json, gcs_md5_hex

GCS_LIST = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_OBJECT = "https://storage.googleapis.com/{bucket}/{name}?generation={generation}"
HF_TREE = "https://huggingface.co/api/datasets/{repo}/tree/{revision}{path}"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
GITHUB_CONTENTS = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
HF_HOST = "huggingface.co"

RLDS_BUCKET, RLDS_PREFIX, RLDS_DEST = "gresearch", "robotics/droid/1.0.1/", "droid/1.0.1"
RLDS_SHARDS = 2048
POINTWORLD_REPO = "nvidia/PointWorld-DROID"
POINTWORLD_REVISION = "dd9aaeec94bb14e27ab6b16b6e4aa0dbcf3ef56f"
POINTWORLD_DEST = "pointworld-droid"  # the extracted tree lives in <POINTWORLD_DEST>/droid (data.pointworld_root)
FLOW_MANIFEST = "droid/flows-fs-optimized/_shards_manifest.json"
DROID3D_REPO = "jaibrdhn/droid_3d_extrinsics"
DROID3D_REVISION = "69db9ccf9a51aab00c17b9d61c981079aff4c2ad"
DROID3D_DEST = "droid3d"
DROID3D_FILES = ("README.md", "extrinsics.jsonl", "camera_intrinsics.jsonl")
HAND_EYE = (
    ("Jai2500/depthworld", "6823c13ddcd96484de04604da7d52c9e12f5c86c",
     "depth_extras/assets/gripper2wrist_transforms.json", "droid3d/hand_eye/depthworld_gripper2wrist_transforms.json"),
    ("NVlabs/PointWorld", "3872ec6ee73146aa671192ef79b5dfbedc0246e3",
     "real/gripper2wrist_transforms.json", "droid3d/hand_eye/pointworld_gripper2wrist_transforms.json"),
)

# Measured on released bytes (2026-11-10, LOCAL copies whose sha256 equal the LFS ids): the cameras package
# (sha256 2640e62b...) holds 42,935 JSON members of 91,524,697 bytes; depth part-0000 (sha256 e76d8eab...)
# decompresses 4,900,000,000 -> 4,921,473,046 bytes (its 458 members are already gzip-chunked HDF5). The depth
# expansion of the other 251 parts is an estimate: part-0000's ratio with a 2 % safety factor.
CAMERAS_MEMBERS, CAMERAS_EXPANDED = 42935, 91524697
DEPTH_RATIO, DEPTH_SAFETY = 4921473046 / 4900000000, 1.02

GROUPS = ("pw_meta", "droid3d", "hand_eye", "pw_cameras", "rlds", "pw_depth", "pw_flows")
DEPTH_PART = re.compile(r"droid/depth_320x180/package\.tar\.zst\.part-(\d{4})")
FLOW_SHARD = re.compile(r"droid/flows-fs-optimized/(shard-\d{6})/package\.tar\.zst\.part-0000")


def pointworld_group(path: str) -> str | None:
    if path in ("LICENSE.pdf", "README.md", FLOW_MANIFEST):
        return "pw_meta"
    if path == "droid/cameras/package.tar.zst.part-0000":
        return "pw_cameras"
    if DEPTH_PART.fullmatch(path):
        return "pw_depth"
    if FLOW_SHARD.fullmatch(path):
        return "pw_flows"
    return None


def hf_checksum(entry: dict) -> dict:
    lfs = entry.get("lfs")
    if not lfs:
        return {"git_sha1": entry["oid"]}
    if re.fullmatch(r"[0-9a-f]{64}", lfs.get("oid", "")):
        return {"sha256": lfs["oid"]}
    return {"lfs_pointer_sha1": entry["oid"]}  # gated repositories mask the sha256 for anonymous listings


def gcs_listing(bucket: str, prefix: str, get=fetch_json) -> list[dict]:
    items, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": "1000",
                 "fields": "items(name,size,md5Hash,crc32c,generation),nextPageToken"}
        if token:
            query["pageToken"] = token
        payload, _ = get(GCS_LIST.format(bucket=bucket) + "?" + urllib.parse.urlencode(query))
        items += payload.get("items", [])
        token = payload.get("nextPageToken")
        if not token:
            return items


def hf_listing(repo: str, revision: str, path: str = "", recursive: bool = True, get=fetch_json) -> list[dict]:
    url = HF_TREE.format(repo=repo, revision=revision, path="/" + path if path else "")
    url += "?" + urllib.parse.urlencode({"recursive": str(recursive).lower(), "expand": "false"})
    entries = []
    while url:
        payload, headers = get(url)
        entries += [e for e in payload if e.get("type") == "file"]
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
        url = match.group(1) if match else None
    return entries


def rlds_items(objects: list[dict]) -> list[dict]:
    items = []
    for obj in sorted(objects, key=lambda o: o["name"]):
        name = obj["name"][len(RLDS_PREFIX):]
        if "/" in name or not (name.endswith(".json") or ".tfrecord-" in name):
            continue
        if "md5Hash" not in obj:
            raise ValueError(f"{obj['name']}: no md5 published (composite object)")
        items.append({
            "id": f"rlds/{name}", "group": "rlds", "size": int(obj["size"]),
            "url": GCS_OBJECT.format(bucket=RLDS_BUCKET, name=urllib.parse.quote(obj["name"]),
                                     generation=obj["generation"]),
            "generation": str(obj["generation"]), "checksum": {"md5": gcs_md5_hex(obj["md5Hash"])},
            "dest": f"{RLDS_DEST}/{name}",
        })
    shards = sum(".tfrecord-" in i["id"] for i in items)
    if shards != RLDS_SHARDS or {"rlds/dataset_info.json", "rlds/features.json"} - {i["id"] for i in items}:
        raise ValueError(f"unexpected RLDS listing: {shards} shards")
    return items


def pointworld_items(entries: list[dict], manifest: dict) -> list[dict]:
    shards = {s["shard_id"]: s for s in manifest["shards"]}
    items = []
    for entry in sorted(entries, key=lambda e: e["path"]):
        path, group = entry["path"], pointworld_group(entry["path"])
        if group is None:
            continue
        item = {
            "id": f"pw/{path}", "group": group, "size": int(entry["size"]), "checksum": hf_checksum(entry),
            "url": HF_RESOLVE.format(repo=POINTWORLD_REPO, revision=POINTWORLD_REVISION, path=path),
        }
        if group == "pw_meta":
            item["dest"] = f"{POINTWORLD_DEST}/{path}"
        else:
            item["dest"] = f".downloads/{POINTWORLD_DEST}/{path}"
        if group == "pw_cameras":
            item["archive"] = {"kind": "single", "kinds": ["cameras"], "expanded": CAMERAS_EXPANDED, "exact": True,
                               "members": CAMERAS_MEMBERS}
        elif group == "pw_depth":
            index = int(DEPTH_PART.fullmatch(path).group(1))
            item["archive"] = {"kind": "stream", "kinds": ["depth_320x180"], "index": index,
                               "expanded": round(item["size"] * DEPTH_RATIO * DEPTH_SAFETY), "exact": False}
        elif group == "pw_flows":
            shard = shards[FLOW_SHARD.fullmatch(path).group(1)]
            (part,) = shard["parts"]
            if part["size_bytes"] != item["size"] or item["checksum"] != {"sha256": part["sha256"]}:
                raise ValueError(f"{path}: HF listing and _shards_manifest.json disagree")
            item["archive"] = {"kind": "single", "kinds": ["flows-fs-optimized"], "shard": shard["shard_id"],
                               "expanded": int(shard["raw_member_bytes"]), "exact": True,
                               "members": int(shard["member_count"])}
        items.append(item)
    depth = [i for i in items if i["group"] == "pw_depth"]
    if [i["archive"]["index"] for i in depth] != list(range(len(depth))) or not depth:
        raise ValueError("depth parts are not a contiguous sequence")
    if sum(i["group"] == "pw_flows" for i in items) != len(shards):
        raise ValueError("flow shards in the listing != _shards_manifest.json")
    return items


def droid3d_items(entries: list[dict]) -> list[dict]:
    found = {e["path"]: e for e in entries}
    missing = [f for f in DROID3D_FILES if f not in found]
    if missing:
        raise ValueError(f"DROID-3D listing lacks {missing}")
    return [{
        "id": f"droid3d/{name}", "group": "droid3d", "size": int(found[name]["size"]),
        "checksum": hf_checksum(found[name]), "auth": "hf", "dest": f"{DROID3D_DEST}/{name}",
        "url": HF_RESOLVE.format(repo=DROID3D_REPO, revision=DROID3D_REVISION, path=name),
    } for name in DROID3D_FILES]


def hand_eye_items(get=fetch_json) -> list[dict]:
    items = []
    for repo, ref, path, dest in HAND_EYE:
        meta, _ = get(GITHUB_CONTENTS.format(repo=repo, path=path, ref=ref))
        items.append({"id": f"hand_eye/{dest.rsplit('/', 1)[1]}", "group": "hand_eye", "size": int(meta["size"]),
                      "checksum": {"git_sha1": meta["sha"]}, "dest": dest,
                      "url": GITHUB_RAW.format(repo=repo, ref=ref, path=path)})
    return items


def build_plan(get=fetch_json, get_bytes=fetch_bytes) -> dict:
    """Every needed file with its official size and checksum, from the sources' own listings (pinned revisions)."""
    rlds = rlds_items(gcs_listing(RLDS_BUCKET, RLDS_PREFIX, get=get))
    pw_entries = hf_listing(POINTWORLD_REPO, POINTWORLD_REVISION, get=get)
    manifest_entry = next(e for e in pw_entries if e["path"] == FLOW_MANIFEST)
    manifest = json.loads(get_bytes(HF_RESOLVE.format(repo=POINTWORLD_REPO, revision=POINTWORLD_REVISION,
                                                      path=FLOW_MANIFEST),
                                    hf_checksum(manifest_entry), int(manifest_entry["size"])))
    items = (pointworld_items(pw_entries, manifest) + droid3d_items(hf_listing(DROID3D_REPO, DROID3D_REVISION,
                                                                                recursive=False, get=get))
             + hand_eye_items(get=get) + rlds)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids")
    return {
        "format": "s4d_dataset_download_plan_v1",
        "sources": {
            "rlds": f"gs://{RLDS_BUCKET}/{RLDS_PREFIX}",
            "pointworld": f"{POINTWORLD_REPO}@{POINTWORLD_REVISION}",
            "droid3d": f"{DROID3D_REPO}@{DROID3D_REVISION}",
            "hand_eye": [f"{repo}@{ref}:{path}" for repo, ref, path, _ in HAND_EYE],
        },
        "flow_members": {s["shard_id"]: sorted(m.rsplit("/", 1)[1][: -len("_flows.h5")] for m in s["members"])
                         for s in manifest["shards"]},
        "items": items,
    }
