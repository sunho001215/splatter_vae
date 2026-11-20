"""Completeness / integrity validation of one ``panda_teleop_episode`` (format_version 2) HDF5 recording.

Read-only: the file is opened with h5py in mode "r" without HDF5 file locking and is never modified.

Result ``status``:
  waiting     modified less than ``grace_s`` ago (max of mtime/ctime), or changed while it was being checked: the
              upload may still be running; nothing is concluded and no failure is counted.
  incomplete  the file does not look finished: HDF5 cannot open it (truncated superblock EOF), or the recorder's
              completion signal is missing (root attrs ``status != "complete"`` or no ``end_unix``).
  failed      it looks finished but violates an integrity check (missing stream, inconsistent row counts, corrupt
              image, non-monotonic or implausible timestamps, ...). The worker retries and only declares such a
              file ``invalid`` after it stayed failed and unchanged for a long time.
  valid       every integrity check passed. ``warnings`` lists content properties that do not make the upload
              incomplete or corrupt but matter downstream (e.g. one dropped stereo record, so colour and stereo
              counts differ by one; recorder task_name label differs from the folder). ``adapter`` reports whether
              the repository's PandaTeleopH5Adapter (include_ir) + contract validation accept the episode.

The required layout is the full format_version 2 schema as recorded (every group/dataset below), which contains
everything s4d/realworld/adapters/panda_teleop.py reads (colour/IR/depth + their timestamps, robot, gripper,
segments, calibration_json) plus the commanded actions (action/*), the haptic device (omega/*) and events.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # never take (or wait for) HDF5 file locks on raw data

import h5py  # noqa: E402
import numpy as np  # noqa: E402

VALIDATOR_VERSION = "rw-ingest-validator/1"
REPO = Path("/home/ws/ws/splatter4d")
FORMAT = "panda_teleop_episode"
FORMAT_VERSION = 2
EXPECTED_CAMERAS = 3
FRAME_PERIOD_S = 1.0 / 15.0          # recorder camera.fps preference 15 (stats fps ~14.99)
ROBOT_PERIOD_S = 0.01                # recording.robot_rate_hz 100
MIN_FRAMES = 30                      # 2 s of camera frames
PAIR_TOLERANCE_S = 0.001             # adapter PAIR_TOLERANCE_NS (colour/depth pairing bound)
SEGMENT_SLACK_S = 0.5                # camera/robot rows must lie inside their segment's [t_start, t_end] +- slack
MAX_CAMERA_SPREAD = 5                # colour frame counts of the three cameras may differ by this much (seen: <= 3)
MAX_UNPAIRED_COLOUR = 3              # colour frames without a stereo record tolerated (as a warning) per camera
NAME_RE = re.compile(r"^episode_(\d{6})\.h5$")
JPEG_SOI, JPEG_EOI = b"\xff\xd8", b"\xff\xd9"
PNG_SIG, PNG_IEND = b"\x89PNG\r\n\x1a\n", b"\x00\x00\x00\x00IEND\xaeB`\x82"

# dataset -> (dtype kinds accepted, trailing shape)
_F, _I, _O = "f", "iu", "O"
CAMERA_DATASETS = {
    "color": (_O, ()), "frame_number": (_I, ()), "frame_timestamp_us": (_F, ()), "hw_timestamp_ms": (_F, ()),
    "segment": (_I, ()), "sensor_timestamp_us": (_F, ()), "t": (_F, ()), "t_active": (_F, ()), "t_recv": (_F, ()),
    "t_recv_unix": (_F, ()), "t_valid": (_I, ()), "timestamp_domain": (_I, ()),
}
STEREO_DATASETS = {
    "color_index": (_I, ()), "color_offset_ms": (_F, ()), "depth": (_O, ()), "depth_frame_number": (_I, ()),
    "depth_hw_timestamp_ms": (_F, ()), "depth_ok": (_I, ()), "depth_timestamp_domain": (_I, ()),
    "emitter_mode": (_I, ()), "frame_number": (_I, ()), "frame_timestamp_us": (_F, ()), "hw_timestamp_ms": (_F, ()),
    "left": (_O, ()), "pair_ok": (_I, ()), "right": (_O, ()), "right_frame_number": (_I, ()),
    "right_hw_timestamp_ms": (_F, ()), "right_timestamp_domain": (_I, ()), "segment": (_I, ()),
    "sensor_timestamp_us": (_F, ()), "t": (_F, ()), "t_active": (_F, ()), "t_recv": (_F, ()),
    "t_recv_unix": (_F, ()), "t_valid": (_I, ()), "timestamp_domain": (_I, ()),
}
GROUP_DATASETS = {
    "robot": {"dq": (_F, (7,)), "flange_pose": (_F, (7,)), "joint_frames": (_F, (7, 7)), "q": (_F, (7,)),
              "seq": (_I, ()), "t": (_F, ()), "t_active": (_F, ()), "t_recv": (_F, ()), "t_send_unix": (_F, ()),
              "tau_ext": (_F, (7,)), "wrench_ext": (_F, (6,))},
    "action": {"engaged": (_I, ()), "flange_pose": (_F, (7,)), "flange_target": (_F, (7,)), "gripper": (_F, ()),
               "lead_limited": (_I, ()), "segment": (_I, ()), "sent": (_I, ())},
    "omega": {"buttons": (_I, ()), "gripper": (_F, ()), "gripper_raw": (_F, ()), "pos": (_F, (3,)),
              "quat": (_F, (4,)), "t": (_F, ())},
    "gripper": {k: (_F, ()) for k in ("activated", "cmd_sent", "current", "fault", "object", "pos_request_echo",
                                       "position", "status", "t", "t_active")} | {"segment": (_I, ())},
    "events": {"t": (_F, ()), "text": (_O, ())},
    "segments": {"segment": (_I, ()), "t_active_end": (_F, ()), "t_active_start": (_F, ()), "t_end": (_F, ()),
                 "t_start": (_F, ())},
}
REQUIRED_ROOT_ATTRS = ("format", "format_version", "status", "end_unix", "created_unix", "episode_index",
                       "task_name", "stats_json", "conventions_json", "T_flange_controller_ee", "T_flange_tcp",
                       "software_version", "hostname", "camera_calibration", "config_yaml")
CAL_EXTRINSICS = ("color_from_depth", "infrared_2_from_infrared_1", "color_from_infrared_1", "depth_from_infrared_1")
# streams decoded in full at first/middle/last frame; every frame is checked for the codec's start/end markers
IMAGE_STREAMS = (("color", "jpeg", "color"), ("stereo/left", "png8", "infrared_1"),
                 ("stereo/right", "png8", "infrared_2"), ("stereo/depth", "png16", "depth"))
EXPECTED_ADAPTER_WARNINGS = ("gripper model unknown", "no URDF supplied")  # no robot config is passed here
# the reader's refusals of colour frames without a stereo record (its 1:1 check; at f9e8332 the shape mismatch raises
# numpy's broadcast ValueError before its own message) and of recorder-flagged frames
ADAPTER_UNPAIRED_ERRORS = ["depth frames are not paired 1:1 with colour frames", "could not be broadcast together"]
ADAPTER_FLAGGED_ERRORS = ["flagged by the recorder"]
# short machine-readable codes for the (non-fatal) warnings of a valid episode
WARNING_CODES = (("task_name attribute", "task_name_mismatch"), ("episode_index attribute", "index_name_mismatch"),
                 ("have no stereo record", "colour_stereo_unpaired"), ("dropped frame", "dropped_frames"),
                 ("flagged invalid by the recorder", "recorder_flagged_frames"),
                 ("repository adapter/contract refuses", "adapter_refused"), ("unknown members", "unknown_datasets"),
                 ("repository adapter unavailable", "adapter_unavailable"),
                 ("frame counts differ across cameras", "camera_count_spread"))


def warning_codes(warnings: list[str]) -> list[str]:
    return sorted({code for w in warnings for needle, code in WARNING_CODES if needle in w}
                  | ({"other_warning"} if any(not any(n in w for n, _ in WARNING_CODES) for w in warnings) else set()))


class _Failed(Exception):
    """Integrity failure: the file looks finished but is not usable as recorded."""


class _Incomplete(Exception):
    """The file does not (yet) look like a finished recording."""


def stat_fingerprint(path: Path) -> dict:
    st = os.stat(path, follow_symlinks=False)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns, "ino": st.st_ino,
            "dev": st.st_dev}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _code_digest() -> str:
    h = hashlib.sha256(Path(__file__).read_bytes())
    adapter = REPO / "s4d/realworld/adapters/panda_teleop.py"
    contract = REPO / "s4d/realworld/contract.py"
    for p in (adapter, contract):
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()[:16]


CODE_DIGEST = _code_digest()


def _attr_str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _as_bytes(x) -> bytes:
    return x.tobytes() if isinstance(x, np.ndarray) else bytes(x)


def _check_dataset(g, name: str, spec, errors: list, where: str):
    kinds, trailing = spec
    if name not in g:
        errors.append(f"{where}/{name}: missing")
        return None
    ds = g[name]
    if not isinstance(ds, h5py.Dataset):
        errors.append(f"{where}/{name}: not a dataset")
        return None
    if ds.dtype.kind not in kinds:
        errors.append(f"{where}/{name}: dtype {ds.dtype} (kind {ds.dtype.kind!r}) not in {kinds!r}")
    if tuple(ds.shape[1:]) != tuple(trailing) or len(ds.shape) != 1 + len(trailing):
        errors.append(f"{where}/{name}: shape {ds.shape} != (N, {', '.join(map(str, trailing))})")
    return ds


def _group_rows(g, specs: dict, where: str, errors: list, warnings: list) -> int | None:
    lengths = {}
    for name, spec in specs.items():
        ds = _check_dataset(g, name, spec, errors, where)
        if ds is not None:
            lengths[name] = len(ds)
    extra = sorted(set(g.keys()) - set(specs) - {"stereo"})
    if extra:
        warnings.append(f"{where}: unknown members {extra}")
    if len(set(lengths.values())) > 1:
        errors.append(f"{where}: datasets have different row counts {sorted(set(lengths.values()))}")
        return None
    return next(iter(lengths.values()), None)


def _strictly_increasing(a: np.ndarray) -> bool:
    return bool(len(a) < 2 or np.all(np.diff(a) > 0))


def _within_segment_steps(t: np.ndarray, seg: np.ndarray) -> np.ndarray:
    same = seg[1:] == seg[:-1]
    return np.diff(t)[same]


def _check_steps(name: str, t: np.ndarray, seg: np.ndarray, period: float, errors: list, warnings: list,
                 max_gap_s: float) -> dict:
    d = _within_segment_steps(t, seg)
    out = {"steps": int(len(d))}
    if len(d) == 0:
        return out
    med = float(np.median(d))
    out.update(median_ms=round(med * 1e3, 3), max_ms=round(float(d.max()) * 1e3, 1))
    if not 0.9 * period <= med <= 1.1 * period:
        errors.append(f"{name}: median within-segment interval {med * 1e3:.2f} ms, expected ~{period * 1e3:.1f} ms")
    if d.min() < 0.5 * period:
        errors.append(f"{name}: within-segment interval {d.min() * 1e3:.2f} ms < half the period")
    if d.max() > max_gap_s:
        errors.append(f"{name}: within-segment gap {d.max():.3f} s > {max_gap_s} s")
    dropped = int(np.sum(np.rint(d / period).astype(int) - 1).clip(min=0)) if len(d) else 0
    if dropped:
        out["dropped_frames"] = dropped
        warnings.append(f"{name}: {dropped} dropped frame(s) within segments (max interval {d.max() * 1e3:.1f} ms)")
    return out


def _check_in_segments(name: str, t: np.ndarray, seg: np.ndarray, segments: dict, errors: list) -> None:
    for k in np.unique(seg):
        if int(k) not in segments:
            errors.append(f"{name}: rows reference segment {int(k)} not in segments/segment")
            continue
        t0, t1 = segments[int(k)]
        tk = t[seg == k]
        if tk.min() < t0 - SEGMENT_SLACK_S or tk.max() > t1 + SEGMENT_SLACK_S:
            errors.append(f"{name}: segment {int(k)} rows span [{tk.min():.3f}, {tk.max():.3f}] outside "
                          f"[{t0:.3f}, {t1:.3f}] +- {SEGMENT_SLACK_S} s")


def _scan_images(ds, codec: str, expected_wh: tuple[int, int] | None, name: str, errors: list,
                 block: int = 32) -> dict:
    """Every frame: non-empty with the codec's start/end markers. First/middle/last: full decode + size/mode."""
    from PIL import Image

    n = len(ds)
    total = 0
    bad = []
    for i in range(0, n, block):
        for j, x in enumerate(ds[i:i + block]):
            b = _as_bytes(x)
            total += len(b)
            if codec == "jpeg":
                ok = len(b) > 4 and b[:2] == JPEG_SOI and b[-2:] == JPEG_EOI
            else:
                ok = len(b) > 20 and b[:8] == PNG_SIG and b[-12:] == PNG_IEND
            if not ok:
                bad.append(i + j)
    if bad:
        errors.append(f"{name}: {len(bad)} frame(s) empty or without {codec} start/end markers (first {bad[:5]})")
    decoded = {}
    for i in sorted({0, n // 2, n - 1}):
        if i < 0:
            continue
        try:
            img = Image.open(io.BytesIO(_as_bytes(ds[i])))
            img.load()
        except Exception as exc:  # noqa: BLE001 - any decoder failure is an integrity failure
            errors.append(f"{name}[{i}]: decode failed ({type(exc).__name__}: {exc})")
            continue
        want_mode = {"jpeg": ("RGB",), "png8": ("L",), "png16": ("I;16", "I;16B", "I;16L")}[codec]
        if img.mode not in want_mode:
            errors.append(f"{name}[{i}]: mode {img.mode} not in {want_mode}")
        if expected_wh and tuple(img.size) != tuple(expected_wh):
            errors.append(f"{name}[{i}]: size {img.size} != calibration {expected_wh}")
        if codec == "png16":
            arr = np.asarray(img)
            decoded[i] = round(float(np.count_nonzero(arr)) / arr.size, 4)
    return {"frames": n, "bytes": total, "valid_depth_fraction": decoded} if codec == "png16" else \
        {"frames": n, "bytes": total}


DISTORTION_MODELS = ("brown_conrady", "inverse_brown_conrady", "modified_brown_conrady", "none")


def _calibration_problems(cal: dict, serial: str) -> list[str]:
    """Everything PandaTeleopH5Adapter reads from calibration_json (colour, depth, IR pair), checked for presence,
    type and finiteness, so a malformed calibration fails here instead of only in the reader."""
    out = []

    def finite(x, shape) -> bool:
        try:
            a = np.asarray(x, float)
        except (TypeError, ValueError):
            return False
        return a.shape == shape and bool(np.isfinite(a).all())

    dev = cal.get("device") if isinstance(cal, dict) else None
    if not isinstance(dev, dict) or any(k not in dev for k in ("serial_number", "sdk_version", "name",
                                                               "firmware_version")):
        return ["device block incomplete"]
    if str(dev["serial_number"]) != serial:
        out.append(f"belongs to {dev['serial_number']}")
    for k in ("calibration_id", "extrinsics_convention", "depth_scale", "streams", "extrinsics", "stereo"):
        if k not in cal:
            out.append(f"missing {k}")
    if out:
        return out
    if not finite(cal["depth_scale"], ()) or not float(cal["depth_scale"]) > 0:
        out.append("depth_scale not a finite positive number")
    for name in ("color", "depth", "infrared_1", "infrared_2"):
        st = cal["streams"].get(name)
        if not isinstance(st, dict) or any(k not in st for k in ("format", "width", "height", "fps", "intrinsics")):
            out.append(f"stream {name} incomplete")
            continue
        k = st["intrinsics"]
        need = ("width", "height", "fx", "fy", "ppx", "ppy", "distortion_model", "distortion_coeffs")
        if not isinstance(k, dict) or any(x not in k for x in need):
            out.append(f"stream {name} intrinsics incomplete")
            continue
        if k["distortion_model"] not in DISTORTION_MODELS:
            out.append(f"stream {name} distortion model {k['distortion_model']!r} unknown")
        if not all(finite(k[x], ()) for x in ("fx", "fy", "ppx", "ppy")) or not (float(k["fx"]) > 0 < float(k["fy"])):
            out.append(f"stream {name} focal/principal point invalid")
        if (int(k["width"]), int(k["height"])) != (int(st["width"]), int(st["height"])):
            out.append(f"stream {name} intrinsics size differs from the stream size")
        n_coeffs = 0 if k["distortion_model"] == "none" else 5
        coeffs = k["distortion_coeffs"]
        if not isinstance(coeffs, list) or (n_coeffs and not finite(coeffs, (n_coeffs,))) or \
                (not n_coeffs and coeffs and not finite(coeffs, (len(coeffs),))):
            out.append(f"stream {name} distortion_coeffs not {n_coeffs or 'any'} finite numbers")
    for name, src, dst in (("color_from_depth", "depth", "color"),
                           ("infrared_2_from_infrared_1", "infrared_1", "infrared_2"),
                           ("color_from_infrared_1", "infrared_1", "color"),
                           ("depth_from_infrared_1", "infrared_1", "depth")):
        e = cal["extrinsics"].get(name)
        if not isinstance(e, dict) or (e.get("source"), e.get("target")) != (src, dst):
            out.append(f"extrinsic {name} missing or not {src}->{dst}")
        elif not (finite(e.get("R_target_from_source"), (3, 3)) and finite(e.get("t_target_from_source_m"), (3,))):
            out.append(f"extrinsic {name} R/t not a finite 3x3/3-vector")
    stereo = cal["stereo"]
    if not isinstance(stereo, dict) or (stereo.get("left"), stereo.get("right")) != ("infrared_1", "infrared_2"):
        out.append("stereo roles are not infrared_1/infrared_2")
    elif not finite(stereo.get("baseline_m"), ()) or not float(stereo["baseline_m"]) > 0:
        out.append("stereo baseline_m not positive")
    return out


def _camera(f, cam: str, stats: dict, segments: dict, errors: list, warnings: list) -> dict:
    serial = cam[len("cam_"):]
    g = f["cameras"][cam]
    where = f"cameras/{cam}"
    summary: dict = {"serial": serial}
    n_c = _group_rows(g, CAMERA_DATASETS, where, errors, warnings)
    st = g.get("stereo")
    if not isinstance(st, h5py.Group):
        errors.append(f"{where}/stereo: missing")
        return summary
    n_s = _group_rows(st, STEREO_DATASETS, where + "/stereo", errors, warnings)
    summary.update(colour_frames=n_c, stereo_frames=n_s)
    cal = None
    try:
        cal = json.loads(_attr_str(g.attrs["calibration_json"]))
        problems = _calibration_problems(cal, serial)
        if problems:
            errors.extend(f"{where}: calibration_json {p}" for p in problems)
            cal = None
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{where}: calibration_json missing or unreadable ({type(exc).__name__}: {exc})")
        cal = None
    if "info_json" not in g.attrs:
        errors.append(f"{where}: info_json attribute missing")
    if n_c is None or n_s is None:
        return summary
    rows = stats.get("rows", {})
    if rows.get(where) != n_c or rows.get(where + "/stereo") != n_s:
        errors.append(f"{where}: rows (colour {n_c}, stereo {n_s}) != recorder stats_json rows "
                      f"({rows.get(where)}, {rows.get(where + '/stereo')})")
    # (stats_json cameras.*.recorded counts are cumulative over the recorder session, not per file)
    if n_c < MIN_FRAMES or n_s < MIN_FRAMES:
        errors.append(f"{where}: only {n_c} colour / {n_s} stereo frames (< {MIN_FRAMES})")
        return summary
    # timestamps: monotonic, plausible intervals inside segments, inside the segment spans
    t_c, seg_c = g["t"][()], g["segment"][()]
    t_s, seg_s = st["t"][()], st["segment"][()]
    for name, arr in (("t", t_c), ("hw_timestamp_ms", g["hw_timestamp_ms"][()]),
                      ("frame_timestamp_us", g["frame_timestamp_us"][()]),
                      ("sensor_timestamp_us", g["sensor_timestamp_us"][()]),
                      ("stereo/t", t_s), ("stereo/hw_timestamp_ms", st["hw_timestamp_ms"][()]),
                      ("stereo/frame_timestamp_us", st["frame_timestamp_us"][()]),
                      ("stereo/sensor_timestamp_us", st["sensor_timestamp_us"][()])):
        if not np.isfinite(arr).all():
            errors.append(f"{where}/{name}: non-finite timestamps")
        elif not _strictly_increasing(arr):
            errors.append(f"{where}/{name}: not strictly increasing")
    summary["colour_steps"] = _check_steps(f"{where} colour", t_c, seg_c, FRAME_PERIOD_S, errors, warnings, 1.0)
    summary["stereo_steps"] = _check_steps(f"{where} stereo", t_s, seg_s, FRAME_PERIOD_S, errors, warnings, 1.0)
    _check_in_segments(f"{where} colour", t_c, seg_c, segments, errors)
    _check_in_segments(f"{where} stereo", t_s, seg_s, segments, errors)
    ids, cnt = np.unique(seg_c, return_counts=True)  # (no bincount: a corrupt huge id must not allocate)
    summary["frames_per_segment"] = {str(int(i)): int(c) for i, c in zip(ids, cnt)}
    # colour <-> stereo (IR left/right + depth share the stereo record) pairing
    ci = st["color_index"][()].astype(np.int64)
    if ci.min() < 0 or ci.max() >= n_c or not _strictly_increasing(ci):
        errors.append(f"{where}/stereo/color_index: not a strictly increasing index into the {n_c} colour frames")
    else:
        offset = np.abs(t_c[ci] - t_s)
        summary["max_pair_offset_ms"] = round(float(offset.max()) * 1e3, 4)
        if offset.max() > PAIR_TOLERANCE_S:
            errors.append(f"{where}: colour/stereo pair offset {offset.max() * 1e3:.3f} ms > "
                          f"{PAIR_TOLERANCE_S * 1e3} ms")
        unpaired = n_c - n_s
        if unpaired > MAX_UNPAIRED_COLOUR:
            errors.append(f"{where}: {unpaired} colour frames have no stereo (IR/depth) record "
                          f"(> {MAX_UNPAIRED_COLOUR})")
        elif unpaired:
            missing = sorted(set(range(n_c)) - set(ci.tolist()))
            summary["unpaired_colour_frames"] = missing
            warnings.append(f"{where}: colour {n_c} vs stereo (IR/depth) {n_s} frames: colour frame(s) {missing} "
                            "have no stereo record (dropped by the recorder; stereo/color_index maps the rest)")
    hw = {k: st[k][()] for k in ("hw_timestamp_ms", "right_hw_timestamp_ms", "depth_hw_timestamp_ms")}
    if not (np.array_equal(hw["hw_timestamp_ms"], hw["right_hw_timestamp_ms"])
            and np.array_equal(hw["hw_timestamp_ms"], hw["depth_hw_timestamp_ms"])):
        errors.append(f"{where}/stereo: left/right/depth hardware timestamps differ")
    for name, arr in (("t_valid", g["t_valid"][()]), ("stereo/t_valid", st["t_valid"][()]),
                      ("stereo/depth_ok", st["depth_ok"][()]), ("stereo/pair_ok", st["pair_ok"][()])):
        bad = int(np.count_nonzero(arr == 0))
        if bad:
            warnings.append(f"{where}/{name}: {bad} frame(s) flagged invalid by the recorder")
            summary.setdefault("recorder_flagged", {})[name] = bad
    # readable content
    images = {}
    for path, codec, stream in IMAGE_STREAMS:
        wh = None
        if cal is not None:
            s = cal["streams"][stream]
            wh = (int(s["width"]), int(s["height"]))
        ds = g[path]
        images[path] = _scan_images(ds, codec, wh, f"{where}/{path}", errors)
    summary["images"] = images
    return summary


def _robot(f, segments: dict, errors: list, warnings: list) -> dict:
    rows = {}
    for grp, specs in GROUP_DATASETS.items():
        if grp not in f or not isinstance(f[grp], h5py.Group):
            errors.append(f"{grp}: group missing")
            continue
        rows[grp] = _group_rows(f[grp], specs, grp, errors, warnings)
    summary = {"rows": rows}
    T = rows.get("robot")
    if T is None:
        return summary
    if rows.get("action") != T or rows.get("omega") != T:
        errors.append(f"action/omega rows {rows.get('action')}/{rows.get('omega')} != robot rows {T}")
        return summary
    if T < 100:
        errors.append(f"robot: only {T} rows")
        return summary
    t = f["robot/t"][()]
    seg = f["action/segment"][()]
    if not np.isfinite(t).all() or not _strictly_increasing(t):
        errors.append("robot/t: not finite and strictly increasing")
    else:
        summary["robot_steps"] = _check_steps("robot", t, seg, ROBOT_PERIOD_S, errors, warnings, 0.25)
        _check_in_segments("robot", t, seg, segments, errors)
    for name in ("robot/q", "robot/dq", "robot/flange_pose", "robot/tau_ext", "robot/joint_frames",
                 "action/flange_target", "action/flange_pose", "action/gripper"):
        if not np.isfinite(f[name][()]).all():
            errors.append(f"{name}: non-finite values")
    for name in ("robot/flange_pose", "action/flange_target"):
        qn = np.linalg.norm(f[name][:, 3:7], axis=1)
        if np.abs(qn - 1).max() > 1e-3:
            errors.append(f"{name}: quaternion norm deviates from 1 by {np.abs(qn - 1).max():.2e}")
    sent = f["action/sent"][()]
    summary["actions_sent_fraction"] = round(float(np.mean(sent != 0)), 4)
    if rows.get("gripper"):
        gt = f["gripper/t"][()]
        if not np.isfinite(gt).all() or not _strictly_increasing(gt):
            errors.append("gripper/t: not finite and strictly increasing")
    else:
        errors.append("gripper: no rows")
    ot = f["omega/t"][()]
    if not np.isfinite(ot).all() or not _strictly_increasing(ot):
        errors.append("omega/t: not finite and strictly increasing")
    if rows.get("events"):
        et = f["events/t"][()]
        if not np.isfinite(et).all() or np.any(np.diff(et) < 0):
            errors.append("events/t: not finite and non-decreasing")
        summary["events"] = int(rows["events"])
    else:
        errors.append("events: no rows")
    return summary


def _segments(f, stats: dict, errors: list) -> tuple[dict, list]:
    if "segments" not in f:
        return {}, []
    g = f["segments"]
    try:
        ids = g["segment"][()].astype(int)
        t0, t1 = g["t_start"][()], g["t_end"][()]
        a0, a1 = g["t_active_start"][()], g["t_active_end"][()]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"segments: unreadable ({exc})")
        return {}, []
    if len(ids) == 0:
        errors.append("segments: empty")
        return {}, []
    if not (np.isfinite(t0).all() and np.isfinite(t1).all() and np.all(t1 > t0)):
        errors.append("segments: t_end must exceed t_start")
    if len(ids) > 1 and np.any(t0[1:] < t1[:-1]):
        errors.append("segments: overlap")
    if ids.tolist() != list(range(len(ids))):
        errors.append(f"segments/segment {ids.tolist()} != 0..{len(ids) - 1}")
    if stats.get("segments") != len(ids) or stats.get("rows", {}).get("segments") != len(ids):
        errors.append(f"segments: {len(ids)} rows != stats_json segments {stats.get('segments')}")
    seg = {int(k): (float(a), float(b)) for k, a, b in zip(ids, t0, t1)}
    detail = [{"segment": int(k), "duration_s": round(float(b - a), 3),
               "active_s": [round(float(x), 3), round(float(y), 3)]} for k, a, b, x, y in zip(ids, t0, t1, a0, a1)]
    for k in range(1, len(detail)):
        detail[k]["pause_before_s"] = round(float(t0[k] - t1[k - 1]), 3)
    return seg, detail


def _adapter_check(path: Path) -> dict:
    """The repository's own reader + contract validation, exactly as preprocessing will run them."""
    out = {"adapter": "panda_teleop_h5", "include_ir": True, "available": True}
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter
        from s4d.realworld.contract import validate_episode
    except Exception as exc:  # noqa: BLE001 - a broken checkout must not fail every episode
        out.update(available=False, ok=None, errors=[f"{type(exc).__name__}: {str(exc)[:300]}"], warnings=[])
        return out
    try:
        out["adapter_version"] = PandaTeleopH5Adapter.version
        meta = PandaTeleopH5Adapter({"episodes": {"episode": str(path)}, "include_ir": True}).read_meta("episode")
        report = validate_episode(meta)
        out["errors"] = list(report.errors)
        out["warnings"] = [w for w in report.warnings if not w.startswith(EXPECTED_ADAPTER_WARNINGS)]
        out["ok"] = report.ok
    except Exception as exc:  # noqa: BLE001 - adapter refusal is reported, not raised
        out.update(ok=False, errors=[f"{type(exc).__name__}: {str(exc)[:300]}"], warnings=[])
    return out


def validate(path: str | Path, *, grace_s: float = 120.0, now: float | None = None, task_names=None,
             run_adapter: bool = True, compute_sha: bool = True) -> dict:
    """Validate one episode file; never raises for data problems (see module docstring for ``status``)."""
    path = Path(path)
    started = time.time()
    now = started if now is None else now
    res: dict = {"validator_version": VALIDATOR_VERSION, "code_digest": CODE_DIGEST, "path": str(path),
                 "checked_at": started, "reasons": [], "warnings": [], "summary": {}}
    m = NAME_RE.match(path.name)
    res["file_index"] = int(m.group(1)) if m else None
    try:
        before = stat_fingerprint(path)
    except FileNotFoundError:
        res.update(status="missing", reasons=["file not found"])
        return res
    res["stat"] = before
    if not os.path.isfile(path) or os.path.islink(path):
        res.update(status="failed", reasons=["not a regular file"])
        return res
    age = now - max(before["mtime_ns"], before["ctime_ns"]) / 1e9
    res["age_s"] = round(age, 1)
    if age < grace_s:
        res.update(status="waiting", reasons=[f"modified {age:.0f} s ago (< grace {grace_s:.0f} s)"])
        return res
    if before["size"] == 0:
        res.update(status="incomplete", reasons=["empty file"])
        return res
    errors, warnings = res["reasons"], res["warnings"]
    try:
        try:
            f = h5py.File(path, "r", locking=False)
        except Exception as exc:  # noqa: BLE001
            raise _Incomplete(f"HDF5 open failed: {type(exc).__name__}: {str(exc)[:300]}") from None
        with f:
            a = f.attrs
            missing = [k for k in ("status", "end_unix") if k not in a]
            status_attr = _attr_str(a["status"]) if "status" in a else None
            res["summary"]["recorder_status"] = status_attr
            if missing or status_attr != "complete":
                raise _Incomplete(f"recorder completion signal absent (status={status_attr!r}, missing {missing})")
            missing = [k for k in REQUIRED_ROOT_ATTRS if k not in a]
            if missing:
                raise _Failed(f"root attributes missing: {missing}")
            if _attr_str(a["format"]) != FORMAT or int(a["format_version"]) != FORMAT_VERSION:
                raise _Failed(f"format {_attr_str(a['format'])!r} v{a['format_version']} is not {FORMAT} "
                              f"v{FORMAT_VERSION}")
            end_unix, created_unix = float(a["end_unix"]), float(a["created_unix"])
            if not (math.isfinite(end_unix) and math.isfinite(created_unix) and end_unix > created_unix):
                raise _Failed(f"end_unix {end_unix} not after created_unix {created_unix}")
            try:
                stats = json.loads(_attr_str(a["stats_json"]))
                json.loads(_attr_str(a["conventions_json"]))
            except Exception as exc:  # noqa: BLE001
                raise _Failed(f"stats_json/conventions_json unreadable: {exc}") from None
            for k in ("T_flange_controller_ee", "T_flange_tcp"):
                if np.asarray(a[k]).shape != (4, 4) or not np.isfinite(np.asarray(a[k], float)).all():
                    errors.append(f"root attribute {k} is not a finite 4x4 matrix")
            episode_index = int(a["episode_index"])
            task_name = _attr_str(a["task_name"])
            res["episode_index"] = episode_index
            res["summary"].update(
                format=FORMAT, format_version=int(a["format_version"]), episode_index=episode_index,
                task_name=task_name, software_version=_attr_str(a["software_version"]),
                hostname=_attr_str(a["hostname"]), mode=_attr_str(a.get("mode", "")),
                created_unix=created_unix, end_unix=end_unix, wall_duration_s=round(end_unix - created_unix, 3),
                duration_s=stats.get("duration_s"), active_duration_s=stats.get("active_duration_s"),
                dropped_batches=stats.get("dropped_batches"), late_frames=stats.get("late_frames"))
            if res["file_index"] is not None and episode_index != res["file_index"]:
                warnings.append(f"episode_index attribute {episode_index} != filename index {res['file_index']}")
            if task_names and task_name not in task_names:
                warnings.append(f"task_name attribute {task_name!r} not in {sorted(task_names)} for this folder")
            if "cameras" not in f:
                raise _Failed("cameras group missing")
            cams = sorted(k for k in f["cameras"] if k.startswith("cam_"))
            res["summary"]["cameras"] = len(cams)
            if len(cams) != EXPECTED_CAMERAS:
                errors.append(f"{len(cams)} cameras ({cams}), expected {EXPECTED_CAMERAS}")
            if set(stats.get("cameras", {})) != set(cams):
                errors.append(f"stats_json cameras {sorted(stats.get('cameras', {}))} != groups {cams}")
            segments, seg_detail = _segments(f, stats, errors)
            res["summary"]["segments"] = seg_detail
            res["summary"]["n_segments"] = len(seg_detail)
            res["summary"]["pauses_s"] = [d["pause_before_s"] for d in seg_detail[1:]]
            robot = _robot(f, segments, errors, warnings)
            res["summary"]["robot"] = robot
            rows = stats.get("rows", {})
            for grp in ("robot", "gripper"):
                if robot["rows"].get(grp) is not None and rows.get(grp) != robot["rows"][grp]:
                    errors.append(f"{grp}: {robot['rows'][grp]} rows != stats_json rows {rows.get(grp)}")
            res["summary"]["camera"] = {c: _camera(f, c, stats, segments, errors, warnings) for c in cams}
            counts = [v.get("colour_frames") for v in res["summary"]["camera"].values()]
            if all(isinstance(c, int) for c in counts) and counts and \
                    max(counts) - min(counts) > max(MAX_CAMERA_SPREAD, 0.03 * max(counts)):
                errors.append(f"colour frame counts differ across cameras: {counts} (a camera stopped/started late)")
    except _Incomplete as exc:
        errors.insert(0, str(exc))
        res["status"] = "incomplete"
    except _Failed as exc:  # fatal structural failure: later checks are meaningless
        errors.insert(0, str(exc))
        res["status"] = "failed"
    except Exception as exc:  # noqa: BLE001 - unexpected reader errors on a finished-looking file
        errors.insert(0, f"unexpected {type(exc).__name__}: {str(exc)[:300]}")
        res["status"] = "failed"
    if errors and "status" not in res:
        res["status"] = "failed"
    if "status" in res:
        res["elapsed_s"] = round(time.time() - started, 2)
        return res
    if run_adapter:
        res["adapter"] = _adapter_check(path)
        ad = res["adapter"]
        if not ad.get("available", True):
            warnings.append("repository adapter unavailable (not checked): " + "; ".join(ad["errors"]))
        elif not ad["ok"]:
            # acceptable only when explained by a recorder-content condition found above (one dropped stereo record,
            # recorder-flagged frames): the reader's documented 1:1/flag requirements. Anything else is integrity.
            cams = list(res["summary"].get("camera", {}).values())
            known = []  # adapter messages attributable to a condition this validator detected itself
            if any("unpaired_colour_frames" in c for c in cams):
                known += ADAPTER_UNPAIRED_ERRORS
            if any("recorder_flagged" in c for c in cams):
                known += ADAPTER_FLAGGED_ERRORS
            explained = bool(ad["errors"]) and all(any(k in err for k in known) for err in ad["errors"])
            if explained:
                warnings.append("repository adapter/contract refuses this episode: " + "; ".join(ad["errors"]))
            else:
                errors.append("repository adapter/contract refuses this episode (not explained by a known recorder "
                              "condition): " + "; ".join(ad["errors"]))
                res["status"] = "failed"
                res["elapsed_s"] = round(time.time() - started, 2)
                return res
    if compute_sha:
        res["sha256"] = sha256_file(path)
    after = stat_fingerprint(path)
    if after != before:
        res.update(status="waiting", reasons=["file changed while it was being validated"], stat=after)
        res.pop("sha256", None)
    else:
        res["status"] = "valid"
    res["warning_codes"] = warning_codes(warnings)
    res["elapsed_s"] = round(time.time() - started, 2)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--grace", type=float, default=120.0)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--no-sha", action="store_true")
    args = ap.parse_args()
    for p in args.paths:
        r = validate(p, grace_s=args.grace, run_adapter=not args.no_adapter, compute_sha=not args.no_sha)
        print(json.dumps({k: r.get(k) for k in ("path", "status", "reasons", "warnings", "sha256", "elapsed_s",
                                                "episode_index")}, indent=None))
