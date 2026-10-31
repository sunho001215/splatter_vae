"""Random access to individual steps of the *released* DROID RLDS TFRecord shards, in place (no copy, no TF).

``build_shard_index`` streams one shard once (one episode record in memory at a time) and stores, per episode, the
absolute byte span of every per-step JPEG and of each packed numeric payload, plus the episode identity
(``episode_metadata/file_path`` and ``recording_folderpath``). ``RldsStepReader`` then preads only the frames a window
needs through a bounded pool of open shard descriptors.

Verified RLDS 1.0.1 schema (``features.json``, see docs/droid_largescale.md): ``steps/observation/{exterior_image_1_left,
exterior_image_2_left, wrist_image_left}`` JPEG 180x320x3; ``steps/observation/cartesian_position`` (6,),
``joint_position`` (7,), ``gripper_position`` (1,) float64 tensors; ``steps/action`` (7,); no per-step timestamps
and no camera calibration in the RLDS release.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

from s4d.data.droid.largescale.handles import ByteLRU, HandlePool
from s4d.data.droid.largescale.tfrecord import TFRecordError, decode_numeric, iter_record_spans, walk_example

INDEX_FORMAT = "s4d_droid_rlds_step_index_v1"
IMAGE_KEYS = {
    "ext1": "steps/observation/exterior_image_1_left",
    "ext2": "steps/observation/exterior_image_2_left",
    "wrist": "steps/observation/wrist_image_left",
}
NUMERIC_KEYS = {  # name -> (feature key, per-step width)
    "cartesian_position": ("steps/observation/cartesian_position", 6),
    "gripper_position": ("steps/observation/gripper_position", 1),
    "joint_position": ("steps/observation/joint_position", 7),
    "action": ("steps/action", 7),
}
META_KEYS = ("episode_metadata/file_path", "episode_metadata/recording_folderpath")
IMAGE_HW = (180, 320)


def _npz_key(*parts: str) -> str:
    return "__".join(parts)


def build_shard_index(shard_path: str | Path, out_dir: str | Path, *, shard_id: int, verify_payload: bool = False) -> dict:
    """Index one shard; writes ``<out_dir>/<shard name>.npz`` and returns its episode rows (identity only)."""
    shard_path = Path(shard_path).resolve(strict=True)
    out_dir = Path(out_dir)
    wanted = set(IMAGE_KEYS.values()) | {key for key, _ in NUMERIC_KEYS.values()} | set(META_KEYS)
    rec_off, rec_len, steps = [], [], []
    img_off = {name: [] for name in IMAGE_KEYS}
    img_len = {name: [] for name in IMAGE_KEYS}
    num_off = {name: [] for name in NUMERIC_KEYS}
    num_len = {name: [] for name in NUMERIC_KEYS}
    rows = []
    fd = os.open(shard_path, os.O_RDONLY)
    try:
        stat = os.fstat(fd)
        for record, (offset, length) in enumerate(iter_record_spans(fd, verify_payload=verify_payload)):
            payload = os.pread(fd, length, offset)
            spans = walk_example(payload, wanted)
            missing = sorted(wanted - set(spans))
            if missing:
                raise TFRecordError(f"{shard_path.name} record {record} misses RLDS features {missing}")
            T = len(spans[IMAGE_KEYS["ext1"]].elements)
            if T == 0:
                raise TFRecordError(f"{shard_path.name} record {record} has no steps")
            for name, key in IMAGE_KEYS.items():
                elements = spans[key].elements
                if len(elements) != T:
                    raise TFRecordError(f"{key} has {len(elements)} frames, expected {T}")
                img_off[name].extend(offset + o for o, _ in elements)
                img_len[name].extend(n for _, n in elements)
            for name, (key, width) in NUMERIC_KEYS.items():
                span = spans[key]
                if span.kind != "float" or span.packed is None:
                    raise TFRecordError(f"{key} must be a packed float_list (RLDS encoding 'none')")
                if span.packed[1] != 4 * width * T:
                    raise TFRecordError(f"{key} holds {span.packed[1]} bytes, expected {4 * width * T}")
                num_off[name].append(offset + span.packed[0])
                num_len[name].append(span.packed[1])
            meta = {}
            for key in META_KEYS:
                elements = spans[key].elements
                if len(elements) != 1:
                    raise TFRecordError(f"{key} must hold exactly one string")
                o, n = elements[0]
                meta[key.split("/", 1)[1]] = payload[o : o + n].decode()
            rec_off.append(offset)
            rec_len.append(length)
            steps.append(T)
            rows.append({"shard": shard_id, "row": record, "num_steps": T, **meta})
            del payload
    finally:
        os.close(fd)
    arrays = {
        "record_offset": np.asarray(rec_off, np.int64),
        "record_length": np.asarray(rec_len, np.int64),
        "step_ptr": np.concatenate([[0], np.cumsum(steps)]).astype(np.int64),
    }
    for name in IMAGE_KEYS:
        arrays[_npz_key("img", name, "offset")] = np.asarray(img_off[name], np.int64)
        arrays[_npz_key("img", name, "length")] = np.asarray(img_len[name], np.int64)
    for name in NUMERIC_KEYS:
        arrays[_npz_key("num", name, "offset")] = np.asarray(num_off[name], np.int64)
        arrays[_npz_key("num", name, "length")] = np.asarray(num_len[name], np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{shard_path.name}.npz", **arrays)
    return {"shard": shard_path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "episodes": rows}


def build_index(
    shard_paths, out_dir: str | Path, *, only: range | None = None, verify_payload: bool = False, finalize: bool = True
) -> Path:
    """Index shards (resumable, parallelizable by ``only``); global ordinals follow the given sorted shard order.

    Each shard writes ``<shard>.npz`` and ``<shard>.rows.json``; a shard whose rows file matches the current file
    size and mtime is skipped. ``finalize`` writes ``index.json`` + ``episodes.jsonl`` once every shard is indexed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(p) for p in shard_paths]
    for shard_id, path in enumerate(shard_paths):
        if only is not None and shard_id not in only:
            continue
        rows_path = out_dir / f"{path.name}.rows.json"
        stat = path.stat()
        if rows_path.is_file() and (out_dir / f"{path.name}.npz").is_file():
            done = json.loads(rows_path.read_text())
            if (done["size"], done["mtime_ns"], done["shard_id"]) == (stat.st_size, stat.st_mtime_ns, shard_id):
                continue
        info = build_shard_index(path, out_dir, shard_id=shard_id, verify_payload=verify_payload)
        tmp = rows_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**info, "shard_id": shard_id}))
        os.replace(tmp, rows_path)
    if finalize:
        finalize_index(shard_paths, out_dir)
    return out_dir


def finalize_index(shard_paths, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    shards, ordinal = [], 0
    tmp = out_dir / "episodes.jsonl.tmp"
    with tmp.open("w") as stream:
        for shard_id, path in enumerate(Path(p) for p in shard_paths):
            info = json.loads((out_dir / f"{path.name}.rows.json").read_text())
            if info["shard_id"] != shard_id:
                raise ValueError(f"{path.name} was indexed as shard {info['shard_id']}, expected {shard_id}")
            for row in info.pop("episodes"):
                row["ordinal"] = ordinal
                ordinal += 1
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            info.pop("shard_id")
            shards.append(info)
    os.replace(tmp, out_dir / "episodes.jsonl")
    index = {"format": INDEX_FORMAT, "shards": shards, "episodes": ordinal, "image_hw": list(IMAGE_HW)}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return out_dir


def load_episode_rows(index_dir: str | Path) -> list[dict]:
    with (Path(index_dir) / "episodes.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"))
    if array.shape != (*IMAGE_HW, 3):
        raise ValueError(f"RLDS frame has shape {array.shape}, expected {(*IMAGE_HW, 3)}")
    return array


class RldsStepReader:
    """pread-based access to selected steps of an indexed episode. Bounded open shards and cached shard indices."""

    def __init__(
        self, index_dir: str | Path, data_root: str | Path, *, max_open_files: int = 8, max_index_bytes: int = 64 << 20
    ):
        self.index_dir = Path(index_dir)
        self.data_root = Path(data_root)
        self.meta = json.loads((self.index_dir / "index.json").read_text())
        if self.meta.get("format") != INDEX_FORMAT:
            raise ValueError(f"not an RLDS step index: {self.meta.get('format')!r}")
        self._shards = [row["shard"] for row in self.meta["shards"]]
        self._expected = {row["shard"]: (row["size"], row["mtime_ns"]) for row in self.meta["shards"]}
        self.files = HandlePool(self._open_shard, os.close, max_open=max_open_files)
        self._indices = ByteLRU(max_index_bytes)

    def _open_shard(self, name: str) -> int:
        fd = os.open(self.data_root / name, os.O_RDONLY)
        stat = os.fstat(fd)
        if stat.st_size != self._expected[name][0]:
            os.close(fd)
            raise ValueError(f"RLDS shard {name} changed size since indexing")
        return fd

    def _shard_index(self, shard: int) -> dict:
        def load():
            with np.load(self.index_dir / f"{self._shards[shard]}.npz") as data:
                arrays = {k: data[k] for k in data.files}
            return arrays, sum(a.nbytes for a in arrays.values())

        return self._indices.get(shard, load)

    def read(
        self, shard: int, row: int, steps, *, cameras=("ext1", "ext2"), numeric=("cartesian_position", "gripper_position")
    ) -> dict:
        index = self._shard_index(shard)
        fd = self.files.get(self._shards[shard])
        start, end = index["step_ptr"][row], index["step_ptr"][row + 1]
        steps = np.asarray(steps, np.int64)
        if steps.min() < 0 or steps.max() >= end - start:
            raise IndexError(f"steps {steps.tolist()} outside episode of {end - start} steps")
        out = {"num_steps": int(end - start)}
        for name in cameras:
            offs = index[_npz_key("img", name, "offset")][start + steps]
            lens = index[_npz_key("img", name, "length")][start + steps]
            out[name] = np.stack([decode_jpeg(os.pread(fd, int(n), int(o))) for o, n in zip(offs, lens)])
        for name in numeric:
            _, width = NUMERIC_KEYS[name]
            offset = int(index[_npz_key("num", name, "offset")][row])
            length = int(index[_npz_key("num", name, "length")][row])
            values = decode_numeric(os.pread(fd, length, offset), "float").reshape(-1, width)
            out[name] = values[steps].astype(np.float64)
        return out

    def close(self) -> None:
        self.files.close()
        self._indices.clear()
