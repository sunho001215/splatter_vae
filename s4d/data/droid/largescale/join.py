"""Join DROID RLDS, PointWorld-DROID and DROID-3D by verified identity; write a small per-episode manifest.

Join keys (never unrelated numeric indices such as DepthWorld ``traj_id`` or RLDS ordinals):

* episode: PointWorld ``uuid`` == DROID-3D ``episode_uuid`` (exact string); PointWorld ``scene_path`` suffix
  ``<lab>/<success|failure>/<date>/<folder>`` == RLDS ``episode_metadata/file_path`` *and* ``recording_folderpath``
  suffixes; DROID-3D ``scene_path`` suffix, when present, must agree too. Ambiguous (duplicate) keys are rejected.
* camera: by serial. PointWorld flow attrs ``ext{1,2}_cam_serial`` == ``camera_<serial>_ext`` groups ==
  ``<serial>+ext`` depth groups == DROID-3D intrinsics ``cameras.ext{1,2}.serial``; wrist: depth ``<serial>+wrist`` ==
  flow attr ``wrist_cam_serial`` (when present) == DROID-3D ``cameras.wrist.serial``. RLDS ``exterior_image_{1,2}_left``
  / ``wrist_image_left`` follow the same ext1/ext2/wrist roles (DROID metadata assigns ext1/ext2 by sorted serial).
* frame: canonical PointWorld index ``i`` -> raw/RLDS step ``2 i`` (``time_skip_ratio`` 2 of the ext1 timeline),
  validated per window by robot-state agreement (PointWorld ``gripper_pose`` vs RLDS ``cartesian_position``) and per
  camera by nearest depth timestamp (the selected depth frame must be raw step ``2 i`` within half a frame period).

The manifest stores only small values (matrices, serials, timeline, track counts) plus provenance; sources stay
immutable and are re-read lazily at training time.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from s4d.data.droid.largescale import calibration as C
from s4d.data.droid.largescale.geometry import check_rigid, invert, to_continuous_centres
from s4d.data.droid.largescale.pointworld_source import DEFAULT_LAYOUT, depth_mm_to_m, summarize_episode

MANIFEST_FORMAT = "s4d_droid_pointworld_droid3d_manifest_v1"
CANONICAL_STRIDE = 2  # raw steps per canonical PointWorld step


def scene_suffix(value: str | bytes) -> str:
    """``<lab>/<success|failure>/<date>/<folder>`` of any raw/gs:///local/RLDS episode path."""
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.replace("\\", "/").rstrip("/")
    for tail in ("/trajectory.h5", "/recordings/MP4", "/recordings/SVO", "/recordings"):
        if text.endswith(tail):
            text = text[: -len(tail)]
    parts = [p for p in text.split("/") if p]
    for marker in ("success", "failure"):
        if marker in parts:
            i = len(parts) - 1 - parts[::-1].index(marker)
            if i >= 1 and len(parts) - i == 3:
                return "/".join(parts[i - 1 :])
    raise ValueError(f"not a DROID episode path: {value!r}")


_UUID = re.compile(
    r"^(?P<lab>[^+]+)\+(?P<user>[0-9A-Za-z]+)\+(?P<date>\d{4}-\d{2}-\d{2})-(?P<h>\d{2})h-(?P<m>\d{2})m-(?P<s>\d{2})s$"
)
# DROID names the episode folder ``time.asctime().replace(" ", "_")``: days 1-9 are space padded, so the folder reads
# ``Fri_Oct__6_20:35:03_2023`` (two underscores; 21,880 of 73,217 DROID-3D raw_dir values on real data), and some early
# IRIS folders also replace the colons (``Thu_Mar__2_14_57_27_2023``).
_FOLDER = re.compile(
    r"^\w{3}_(?P<mon>\w{3})_{1,2}(?P<day>\d{1,2})_(?P<h>\d{2})[:_](?P<m>\d{2})[:_](?P<s>\d{2})_(?P<year>\d{4})$"
)
_MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def uuid_matches_suffix(uuid: str, suffix: str) -> bool:
    """DROID builds ``uuid = f"{lab}+{user_id}+{%Y-%m-%d-%Hh-%Mm-%Ss}"`` and the folder from ``time.asctime()`` at the
    same instant, so lab and date-time must agree. The ``<date>`` directory is *not* compared: on real data it is the
    collection day and differs from the folder/uuid date for sessions past midnight (2,772 DROID-3D records, e.g.
    ``2023-11-29/Thu_Nov_30_00:02:20_2023`` for uuid ``...+2023-11-30-00h-02m-20s``)."""
    u, parts = _UUID.match(uuid), suffix.split("/")
    f = _FOLDER.match(parts[-1]) if len(parts) == 4 else None
    if not u or not f or u["lab"].lower() != parts[0].lower():  # TRI writes "tri+..." uuids under "TRI/" (30 records)
        return False
    year, month, day = map(int, u["date"].split("-"))
    return (year, month, day, u["h"], u["m"], u["s"]) == (
        int(f["year"]),
        _MONTHS.get(f["mon"], 0),
        int(f["day"]),
        f["h"],
        f["m"],
        f["s"],
    )


class JoinError(ValueError):
    pass


def _unique_map(rows, key_fn, what: str) -> tuple[dict, set]:
    counts = Counter(key_fn(r) for r in rows)
    duplicates = {k for k, n in counts.items() if n > 1}
    return {key_fn(r): r for r in rows if counts[key_fn(r)] == 1}, duplicates


def index_rlds_rows(rows: list[dict]) -> tuple[dict, dict]:
    """scene suffix -> RLDS row, requiring ``file_path`` and ``recording_folderpath`` to agree; returns rejects too."""
    good, rejects = [], {}
    for row in rows:
        try:
            a, b = scene_suffix(row["file_path"]), scene_suffix(row["recording_folderpath"])
        except ValueError as error:
            rejects[f"rlds:{row['shard']}:{row['row']}"] = str(error)
            continue
        if a != b:
            rejects[f"rlds:{row['shard']}:{row['row']}"] = f"file_path {a!r} != recording_folderpath {b!r}"
            continue
        good.append({**row, "suffix": a})
    by_suffix, duplicates = _unique_map(good, lambda r: r["suffix"], "rlds")
    for key in duplicates:
        rejects[f"rlds-suffix:{key}"] = "duplicate RLDS episodes share one scene suffix"
    return by_suffix, rejects


def _ext_camera(summary, flow_group, cameras_json, serial: str) -> dict:
    K_native = np.asarray(flow_group[f"camera_{serial}_ext"]["intrinsic"][:], np.float64)
    w2c = check_rigid(np.asarray(flow_group[f"camera_{serial}_ext"]["extrinsic"][:], np.float64), f"pw w2c {serial}")
    json_w2c = np.asarray(cameras_json[serial]["optimized_extrinsics"], np.float64)
    if not np.allclose(json_w2c, w2c, atol=1e-5):
        raise JoinError(f"{summary.episode}: camera JSON optimized_extrinsics != flow extrinsic for {serial}")
    return {"serial": serial, "K_pw_native": K_native, "w2c_pw": w2c}


def build_episode_entry(
    pw_root: str | Path,
    episode: str,
    rlds_by_suffix: dict,
    *,
    calibration: str = "droid3d",
    droid3d_extrinsics: C.JsonlIndex | None = None,
    droid3d_intrinsics: C.JsonlIndex | None = None,
    hand_eye: dict | None = None,
    layout: dict = DEFAULT_LAYOUT,
    rlds_reader=None,
    max_dt_ms: float = 34.0,
    max_xyz_m: float = 2e-3,
    max_rot_rad: float = 5e-3,
    wrist_max_residual_m: float = 0.03,
) -> dict:
    """One manifest row, or ``JoinError`` with the reason. ``calibration`` selects the extrinsics used for training
    geometry: ``"pointworld"`` (released optimized w2c) or ``"droid3d"`` (DROID-3D w2c; PointWorld tracks rebased).
    ``rlds_reader`` (an ``RldsStepReader``) is required: every clip's robot state must match RLDS at step ``2 i``."""
    if rlds_reader is None:
        raise ValueError("an RLDS step reader is required to validate the canonical -> RLDS frame mapping")
    import h5py

    if calibration not in ("pointworld", "droid3d"):
        raise ValueError(f"unknown calibration {calibration!r}")
    summary = summarize_episode(pw_root, episode, layout)
    if summary.uuid != episode:
        raise JoinError(f"{episode}: file name and flow uuid {summary.uuid!r} disagree")
    if not summary.optimization_success:
        raise JoinError(f"{episode}: PointWorld camera optimization_success is false")
    suffix = scene_suffix(summary.scene_path)
    if summary.hdf5_path:  # "success/<date>/<folder>/trajectory.h5" (no lab component)
        tail = [p for p in summary.hdf5_path.replace("\\", "/").split("/") if p]
        tail = tail[:-1] if tail and tail[-1] == "trajectory.h5" else tail
        if "/".join(tail[-3:]) != suffix.split("/", 1)[1]:
            raise JoinError(f"{episode}: flow hdf5_path and scene_path disagree")
    rlds = rlds_by_suffix.get(suffix)
    if rlds is None:
        raise JoinError(f"{episode}: no unique RLDS episode with scene suffix {suffix!r}")
    needed_steps = CANONICAL_STRIDE * (max(int(k.split(":")[1]) for k in summary.clips) - 1) + 1
    if rlds["num_steps"] < needed_steps:
        raise JoinError(f"{episode}: RLDS has {rlds['num_steps']} steps, canonical timeline needs {needed_steps}")
    root = Path(pw_root)
    cameras_json = json.loads((root / layout["cameras"].format(episode=episode)).read_text())
    with h5py.File(root / layout["flows"].format(episode=episode), "r") as flow:
        first = flow[summary.clips[0]]
        ext = {role: _ext_camera(summary, first, cameras_json, s) for role, s in zip(("ext1", "ext2"), summary.ext_serials)}
        for key in summary.clips[1:]:
            for role, cam in ext.items():
                group = flow[key][f"camera_{cam['serial']}_ext"]
                if not (
                    np.allclose(group["intrinsic"][:], cam["K_pw_native"], atol=1e-5)
                    and np.allclose(group["extrinsic"][:], cam["w2c_pw"], atol=1e-5)
                ):
                    raise JoinError(f"{episode}: static {role} calibration changes between clips")
        attrs = dict(flow.attrs)
    flow_wrist = str(attrs["wrist_cam_serial"]) if "wrist_cam_serial" in attrs else None
    robot_serial = str(attrs["robot_serial"]) if "robot_serial" in attrs else None
    if len(summary.wrist_serials) > 1:
        raise JoinError(f"{episode}: several wrist depth groups {summary.wrist_serials}")
    depth_wrist = summary.wrist_serials[0] if summary.wrist_serials else None
    if flow_wrist is not None and depth_wrist is not None and flow_wrist != depth_wrist:
        raise JoinError(f"{episode}: wrist serial {flow_wrist} (flow attrs) != {depth_wrist} (depth group)")
    wrist_serial = depth_wrist or flow_wrist

    entry: dict = {
        "episode": episode,
        "scene_suffix": suffix,
        "uuid_matches_folder": uuid_matches_suffix(episode, suffix),
        "rlds": {k: rlds[k] for k in ("shard", "row", "ordinal", "num_steps", "file_path", "recording_folderpath")},
        "timeline": {"canonical_timestamps": list(summary.canonical_timestamps), "stride": CANONICAL_STRIDE},
        "clips": list(summary.clips),
        "track_counts": summary.track_counts,
        "depth_groups": {
            "ext1": f"{ext['ext1']['serial']}+ext",
            "ext2": f"{ext['ext2']['serial']}+ext",
            "wrist": f"{wrist_serial}+wrist" if depth_wrist else None,
        },
        "calibration": calibration,
        "cameras": {},
        "provenance": {
            "pointworld": {
                "dataset": "nvidia/PointWorld-DROID",
                "flows": layout["flows"].format(episode=episode),
                "depth": layout["depth"].format(episode=episode),
                "cameras": layout["cameras"].format(episode=episode),
            },
            "rlds": {"dataset": "droid 1.0.1 (RLDS)", "shard": rlds["shard"], "row": rlds["row"]},
        },
    }
    d3e = droid3d_record(droid3d_extrinsics, episode, suffix, "scene_path")
    d3i = droid3d_record(droid3d_intrinsics, episode, suffix, "raw_dir")
    if calibration == "droid3d" and d3e is None:
        raise JoinError(f"{episode}: not in DROID-3D extrinsics.jsonl")
    parsed = C.parse_droid3d_extrinsics(d3e) if d3e is not None else None
    if d3i is not None:
        for role in ("ext1", "ext2"):
            serial = d3i["cameras"][role].get("serial")
            if serial is not None and str(serial) != ext[role]["serial"]:
                raise JoinError(f"{episode}: DROID-3D {role} serial {serial} != PointWorld {ext[role]['serial']}")
        wserial = d3i["cameras"].get("wrist", {}).get("serial")
        if wserial is not None and wrist_serial is not None and str(wserial) != wrist_serial:
            raise JoinError(f"{episode}: DROID-3D wrist serial {wserial} != PointWorld wrist depth {wrist_serial}")
    robot_serial = (parsed["robot_serial"] if parsed else None) or robot_serial

    for role, cam in ext.items():
        # PointWorld K: ZED K x 0.25 written without a half-pixel shift; its tracks/depth are defined on that grid with
        # integer pixel centres, so it is kept for both calibrations (only the extrinsic changes).
        row = {
            "serial": cam["serial"],
            "K": to_continuous_centres(cam["K_pw_native"], integer_centres=True).tolist(),
            "K_source": "pointworld flow intrinsic (+0.5 px to continuous centres)",
            "w2c_pointworld": cam["w2c_pw"].tolist(),
        }
        if calibration == "droid3d":
            row["w2c"] = parsed["w2c"][role].tolist()
            row["w2c_source"] = (
                f"droid3d extrinsics.jsonl {'T0_ext1_in_world' if role == 'ext1' else 'T1_ext2_in_world'} (w2c)"
            )
            row["tracks"] = "rebased exactly: X' = inv(w2c_droid3d) @ w2c_pointworld @ X (static camera, same K)"
            if d3i is not None:
                factory = C.parse_droid3d_intrinsics(d3i, role)["K"]
                row["K_droid3d_minus_pw_max_px"] = float(np.abs(factory - np.asarray(row["K"])).max())
        else:
            row["w2c"] = cam["w2c_pw"].tolist()
            row["w2c_source"] = "pointworld optimized_extrinsics (w2c)"
            row["tracks"] = "as released"
        entry["cameras"][role] = row

    wrist: dict | None = None
    reasons = []
    if depth_wrist is None:
        reasons.append("no <serial>+wrist depth group")
    if hand_eye is None:
        reasons.append("no hand-eye asset given")
    T_gw = None
    if hand_eye is not None and robot_serial is not None:
        key = robot_serial if robot_serial in hand_eye else next((k for k in hand_eye if k.lower() == robot_serial.lower()), None)
        T_gw = None if key is None else hand_eye[key]
        asset_wrist = getattr(hand_eye, "wrist_serials", {}).get(key) if key is not None else None
        if T_gw is None:
            reasons.append(f"robot {robot_serial} has no hand-eye entry")
        elif asset_wrist is not None and wrist_serial is not None and asset_wrist != wrist_serial:
            reasons.append(f"hand-eye entry {key} belongs to wrist camera {asset_wrist}, episode has {wrist_serial}")
        elif not C.identifies_robot(robot_serial) and asset_wrist is None:
            # a pooled placeholder serial ("unknown", "fr3-") cannot select a per-robot calibration on its own
            reasons.append(f"robot_serial {robot_serial!r} does not identify a robot (no wrist-camera check possible)")
    elif robot_serial is None:
        reasons.append("robot_serial unknown")
    if d3i is None or "wrist" not in d3i.get("cameras", {}):
        reasons.append("no wrist intrinsics (DROID-3D camera_intrinsics.jsonl)")
    if calibration == "droid3d" and parsed is None:
        reasons.append("no DROID-3D corrections")
    if not reasons:
        intr = C.parse_droid3d_intrinsics(d3i, "wrist")
        wrist = {
            "serial": wrist_serial,
            "robot_serial": robot_serial,
            "K": intr["K"].tolist(),
            "K_source": "droid3d camera_intrinsics.jsonl cameras.wrist.left "
            f"({intr['native_wh'][0]}x{intr['native_wh'][1]} -> 320x180, +0.5 px)",
            "T_gw_static": np.asarray(T_gw).tolist(),
            "pose_model": calibration,
        }
        if calibration == "droid3d":
            wrist.update(
                dT_gw=parsed["dT_gw"].tolist(),
                dq=parsed["dq"].tolist(),
                image_rotation_deg=180,
                pose_source="w2c(t) = inv(inv(FLIP @ T_gw_static @ dT_gw @ inv(FK(q(t)+dq))) @ Rz(180))",
            )
        else:
            wrist.update(pose_source="w2c(t) = T_gw_static @ inv(T_base_ee(cartesian_position(t))) (PointWorld recipe)")
    entry["cameras"]["wrist"] = wrist
    entry["wrist_unavailable"] = reasons
    entry["frames"] = validate_frames(
        root, layout, episode, summary, entry, rlds_reader, max_dt_ms=max_dt_ms, max_xyz_m=max_xyz_m, max_rot_rad=max_rot_rad
    )
    if entry["cameras"]["wrist"] is not None:
        qa = wrist_qa(root, layout, episode, summary, entry, rlds_reader, max_residual_m=wrist_max_residual_m)
        entry["wrist_qa"] = qa
        if qa["decision"] != "ok":
            entry["wrist_unavailable"].append(f"wrist QA: {qa['decision']}")
            entry["cameras"]["wrist"] = None
    if parsed is not None:
        entry["droid3d"] = {
            "robot_serial": parsed["robot_serial"],
            "dq": parsed["dq"].tolist(),
            "dT_gw": parsed["dT_gw"].tolist(),
            "meta": parsed["meta"],
        }
        entry["provenance"]["droid3d"] = {"dataset": "jaibrdhn/droid_3d_extrinsics", "key": "episode_uuid"}
    return entry


def droid3d_record(index: C.JsonlIndex | None, episode: str, suffix: str, path_key: str) -> dict | None:
    """The DROID-3D record of ``episode`` whose ``path_key`` (``scene_path`` / ``raw_dir``) suffix is ``suffix``.

    The uuid alone is not unique on real data (``camera_intrinsics.jsonl`` repeats 100 uuids under ``success/`` and
    ``failure/``), so the path suffix decides; a record whose path parses to another suffix is an identity mismatch.
    """
    if index is None:
        return None
    records = index.get_all(episode)
    matching, unparsed = [], []
    for record in records:
        try:
            other = scene_suffix(record.get(path_key, ""))
        except ValueError:  # a path without the lab/outcome/date/folder tail carries no identity
            unparsed.append(record)
            continue
        if other == suffix:
            matching.append(record)
        elif len(records) == 1:
            raise JoinError(f"{episode}: DROID-3D {index.path.name} {path_key} suffix {other!r} != PointWorld {suffix!r}")
    if len(matching) == 1:
        return matching[0]
    if not matching and len(unparsed) == 1 and len(records) == 1:
        return unparsed[0]
    if not records:
        return None
    raise JoinError(
        f"{episode}: {len(records)} DROID-3D {index.path.name} records, {len(matching)} with {path_key} suffix {suffix!r}"
    )


def nearest_index(stamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    stamps = np.asarray(stamps, np.int64)
    if stamps.ndim != 1 or not len(stamps) or np.any(np.diff(stamps) < 0):
        raise JoinError("depth timestamps must be a nonempty sorted vector")
    right = np.searchsorted(stamps, targets).clip(0, len(stamps) - 1)
    left = (right - 1).clip(0, len(stamps) - 1)
    return np.where(np.abs(stamps[left] - targets) <= np.abs(stamps[right] - targets), left, right)


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2
    return np.arccos(np.clip(cos, -1, 1))


def validate_frames(root, layout, episode, summary, entry, rlds_reader, *, max_dt_ms, max_xyz_m, max_rot_rad) -> dict:
    """Validated frame mapping: canonical i -> RLDS step 2i (state agreement on every clip frame) and, per camera,
    the depth frame nearest to the canonical timestamp must *be* raw step 2i within ``max_dt_ms``."""
    import h5py

    from s4d.data.droid.largescale.geometry import pose6_to_matrix, pose7_to_matrix

    stamps = np.asarray(summary.canonical_timestamps, np.int64)
    raw_steps = CANONICAL_STRIDE * np.arange(len(stamps))
    usable = raw_steps < entry["rlds"]["num_steps"]
    rl = entry["rlds"]
    state = rlds_reader.read(
        rl["shard"], rl["row"], raw_steps[usable], cameras=(), numeric=("cartesian_position", "joint_position")
    )
    # convention diagnostic: controller EE frame relative to the Franka flange (FK of the same joint reading)
    flange_ee = invert(C.franka_fk(state["joint_position"])) @ pose6_to_matrix(state["cartesian_position"])
    flange_offset = np.linalg.norm(flange_ee[:, :3, 3], axis=-1)
    xyz_err, rot_err = [], []
    with h5py.File(Path(root) / layout["flows"].format(episode=episode), "r") as flow:
        for key in summary.clips:
            start, end = map(int, key.split(":"))
            if end > usable.sum():
                raise JoinError(f"{episode}: clip {key} exceeds the RLDS episode")
            pw = pose7_to_matrix(flow[key]["gripper_pose"][:])
            rlds = pose6_to_matrix(state["cartesian_position"][start:end])
            xyz_err.append(np.linalg.norm(pw[:, :3, 3] - rlds[:, :3, 3], axis=-1))
            rot_err.append(rotation_angle(pw[:, :3, :3], rlds[:, :3, :3]))
    xyz_err, rot_err = np.concatenate(xyz_err), np.concatenate(rot_err)
    if xyz_err.max() > max_xyz_m or rot_err.max() > max_rot_rad:
        raise JoinError(
            f"{episode}: PointWorld gripper_pose != RLDS cartesian_position[2i] "
            f"(max {xyz_err.max():.4g} m, {rot_err.max():.4g} rad); frame mapping rejected"
        )
    out = {
        "state_max_xyz_m": float(xyz_err.max()),
        "state_max_rot_rad": float(rot_err.max()),
        "flange_to_cartesian_m": float(np.median(flange_offset)),
        "flange_to_cartesian_spread_m": float(flange_offset.max() - flange_offset.min()),
        "depth_index": {},
        "valid": {},
        "max_dt_ms": {},
    }
    with h5py.File(Path(root) / layout["depth"].format(episode=episode), "r") as depth:
        for role, group in entry["depth_groups"].items():
            if group is None:
                continue
            ts = depth[group]["timestamps"][:]
            idx = nearest_index(ts, stamps)
            dt = np.abs(ts[idx] - stamps)
            ok = (idx == raw_steps) & (dt <= max_dt_ms) & usable
            out["depth_index"][role] = idx.tolist()
            out["valid"][role] = ok.tolist()
            out["max_dt_ms"][role] = float(dt[ok].max()) if ok.any() else None
    return out


def wrist_pose_w2c(wrist: dict, state: dict) -> np.ndarray:
    """Per-frame w2c of the stored wrist images under the manifest's pose model (DROID-3D or PointWorld recipe)."""
    from s4d.data.droid.largescale.geometry import wrist_camera_c2w

    T_gw = np.asarray(wrist["T_gw_static"], np.float64)
    if wrist["pose_model"] == "droid3d":
        return C.droid3d_wrist_w2c(
            state["joint_position"], wrist["dq"], T_gw, wrist["dT_gw"], image_rotation_deg=int(wrist["image_rotation_deg"])
        )
    return invert(wrist_camera_c2w(state["cartesian_position"], invert(T_gw)))  # mean_mat = camera <- controller EE


def wrist_qa(root, layout, episode, summary, entry, rlds_reader, *, max_residual_m: float, max_frames: int = 6) -> dict:
    """Data check of the wrist pose model *and* the wrist depth orientation, per episode.

    External-camera tracks (already in the training world frame) that their own camera sees are projected into the
    wrist camera at the same canonical times and compared with the released wrist depth, read as stored and rotated
    by 180 degrees. The orientation with the clearly lower median residual is recorded; a high residual for both
    rejects the wrist calibration for this episode (pose model, hand-eye asset or orientation wrong).
    """
    import h5py

    from s4d.data.droid.largescale.geometry import project, rebase_static_camera_points

    wrist = entry["cameras"]["wrist"]
    valid = np.asarray(entry["frames"]["valid"].get("wrist", []), bool)
    clip = summary.clips[0]
    start, end = map(int, clip.split(":"))
    frames = [i for i in range(start, end) if i < len(valid) and valid[i]][:max_frames]
    if not frames:
        return {"decision": "no valid wrist frames", "depth_rotation_deg": None}
    raw = CANONICAL_STRIDE * np.asarray(frames)
    rl = entry["rlds"]
    state = rlds_reader.read(rl["shard"], rl["row"], raw, cameras=(), numeric=("cartesian_position", "joint_position"))
    w2c = wrist_pose_w2c(wrist, state)
    K = np.asarray(wrist["K"], np.float64)
    residual = {"as_stored": [], "rotated180": []}
    with (
        h5py.File(Path(root) / layout["flows"].format(episode=episode), "r") as flow,
        h5py.File(Path(root) / layout["depth"].format(episode=episode), "r") as depth_file,
    ):
        group = depth_file[entry["depth_groups"]["wrist"]]["depth"]
        for k, i in enumerate(frames):
            depth = depth_mm_to_m(group[entry["frames"]["depth_index"]["wrist"][i]], np.float64)
            for role in ("ext1", "ext2"):
                cam = entry["cameras"][role]
                g = flow[clip][f"camera_{cam['serial']}_ext"]
                pts = g["scene_flows"][i - start].astype(np.float64)
                seen = g["scene_visibility"][i - start] & g["scene_depth_valid_mask"][i - start]
                pts = pts[seen]
                if entry["calibration"] == "droid3d":
                    pts = rebase_static_camera_points(pts, np.asarray(cam["w2c_pointworld"]), np.asarray(cam["w2c"]))
                uv, z = project(pts, K, w2c[k])
                h, w = depth.shape
                px = np.floor(uv).astype(np.int64)
                ok = (z > 0) & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
                for name, d in (("as_stored", depth), ("rotated180", depth[::-1, ::-1])):
                    m = d[px[ok, 1], px[ok, 0]]
                    residual[name].append(np.abs(z[ok] - m)[m > 0])
    med = {k: (float(np.median(np.concatenate(v))) if sum(map(len, v)) else None) for k, v in residual.items()}
    count = int(sum(map(len, residual["as_stored"])))
    out = {"median_residual_m": med, "count": count, "frames": frames, "max_residual_m": max_residual_m}
    a, b = med["as_stored"], med["rotated180"]
    if count < 50 or a is None or b is None:
        out.update(decision="too few wrist observations of external tracks", depth_rotation_deg=None)
    elif min(a, b) > max_residual_m:
        out.update(decision=f"wrist pose/depth inconsistent (median {min(a, b):.3f} m)", depth_rotation_deg=None)
    elif a <= 0.5 * b:
        out.update(decision="ok", depth_rotation_deg=0)
    elif b <= 0.5 * a:
        out.update(decision="ok", depth_rotation_deg=180)
    else:
        out.update(decision="wrist depth orientation ambiguous", depth_rotation_deg=None)
    return out


def build_manifest(
    pw_root: str | Path,
    episodes: list[str],
    rlds_rows: list[dict],
    out_path: str | Path,
    **kwargs,
) -> dict:
    """Join every listed PointWorld episode; write accepted rows to ``out_path`` (JSONL) and a report next to it."""
    by_suffix, rejects = index_rlds_rows(rlds_rows)
    accepted = 0
    seen = set()
    with Path(out_path).open("w") as stream:
        for episode in sorted(episodes):
            if episode in seen:
                rejects[episode] = "duplicate PointWorld episode"
                continue
            seen.add(episode)
            try:
                entry = build_episode_entry(pw_root, episode, by_suffix, **kwargs)
            except (JoinError, KeyError, ValueError, OSError) as error:
                rejects[episode] = f"{type(error).__name__}: {error}"
                continue
            stream.write(json.dumps({"format": MANIFEST_FORMAT, **entry}, sort_keys=True) + "\n")
            accepted += 1
    report = {"format": MANIFEST_FORMAT, "accepted": accepted, "rejected": rejects, "candidates": len(episodes)}
    Path(str(out_path) + ".report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    return report


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("format") != MANIFEST_FORMAT:
                raise ValueError(f"not a {MANIFEST_FORMAT} manifest row")
            rows.append(row)
    names = [r["episode"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("manifest repeats an episode")
    return rows


def rigid(T) -> np.ndarray:
    return check_rigid(np.asarray(T, np.float64), "manifest transform")


__all__ = ["build_manifest", "build_episode_entry", "load_manifest", "scene_suffix", "uuid_matches_suffix", "invert"]
