"""Reproducible native-resolution schema and alignment evidence, not a fixture."""

from __future__ import annotations

import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from s4d.data import DROID_CACHE_ROOT, writable_path
from s4d.data.droid.convert import EPISODE, episode_geometry, sample_paths, sha256
from s4d.data.droid.gripper import gripper_depth_residuals, gripper_points
from s4d.data.droid.pointworld import depth_test, project
from s4d.data.droid.rlds import IMAGE_KEYS, verify_path, verify_states

SCENE_COLOR = "#2a78d6"
GRIPPER_COLOR = "#eb6834"


def inspection_report(sample_root: Path, cache_root: Path) -> dict:
    paths = sample_paths(sample_root)
    with h5py.File(paths["flow"], "r") as flow, h5py.File(paths["depth"], "r") as depth:
        geom = episode_geometry(flow, depth, json.loads(paths["cameras"].read_text()))
        with np.load(cache_root / "matched_raw_external.npz", allow_pickle=False) as saved:
            raw = {key: saved[key] for key in saved.files}
        metadata = json.loads((cache_root / "matched_raw_metadata.json").read_text())
        verify_path(metadata, str(flow.attrs["scene_path"]))
        states = verify_states(raw, flow)
        native_errors, cross_errors, cross_unfiltered, schema, rgb_alignment = [], [], [], {}, []
        for key in geom["clips"]:
            start, end = map(int, key.split(":"))
            schema[key] = {}
            for source, serial in enumerate(geom["serials"]):
                camera = flow[key][f"camera_{serial}_ext"]
                points = camera["scene_flows"][:].astype(np.float32)
                mask = camera["scene_visibility"][:] & camera["scene_depth_valid_mask"][:]
                decoded = np.asarray(Image.open(io.BytesIO(bytes(camera["initial_rgb"][0]))).convert("RGB"))
                schema[key][serial] = {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in camera.items()
                }
                schema[key][serial]["decoded_initial_rgb"] = list(decoded.shape)
                raw_start = 2 * start
                candidates = range(max(0, raw_start - 2), min(len(raw[IMAGE_KEYS[source]]), raw_start + 3))
                image_errors = {
                    index: float(
                        np.abs(decoded.astype(np.float32) - raw[IMAGE_KEYS[source]][index].astype(np.float32)).mean()
                    )
                    for index in candidates
                }
                best_index = min(image_errors, key=image_errors.get)
                rgb_alignment.append(
                    {
                        "clip": key,
                        "serial": serial,
                        "expected_raw_index": raw_start,
                        "best_local_raw_index": best_index,
                        "expected_mean_absolute_rgb_error_255": image_errors[raw_start],
                        "local_raw_index_errors": image_errors,
                    }
                )
                target = 1 - source
                for t, canonical in enumerate(range(start, end)):
                    # For the native metric do not truncate residuals to the
                    # source supervision tolerance or the occlusion threshold.
                    uv, z = project(points[t], geom["K"][source], geom["w2c"][source])
                    pix = np.rint(uv).astype(np.int64)
                    inside = (pix[:, 0] >= 0) & (pix[:, 0] < 320) & (pix[:, 1] >= 0) & (pix[:, 1] < 180) & (z > 0)
                    valid = mask[t] & inside
                    measured = np.zeros(len(points[t]), np.float32)
                    good = np.flatnonzero(valid)
                    measured[good] = geom["depths"][canonical, source, pix[good, 1], pix[good, 0]]
                    valid &= measured > 0
                    native_errors.append(np.abs(z[valid] - measured[valid]))
                    uv, z = project(points[t], geom["K"][target], geom["w2c"][target])
                    pix = np.rint(uv).astype(np.int64)
                    valid = mask[t] & (z > 0) & (pix[:, 0] >= 0) & (pix[:, 0] < 320)
                    valid &= (pix[:, 1] >= 0) & (pix[:, 1] < 180)
                    measured = np.zeros(len(points[t]), np.float32)
                    good = np.flatnonzero(valid)
                    measured[good] = geom["depths"][canonical, target, pix[good, 1], pix[good, 0]]
                    valid &= measured > 0
                    relative = np.abs(z - measured) / np.maximum(measured, 1e-8)
                    cross_unfiltered.append(relative[valid])
                    valid &= z <= measured + 0.02
                    cross_errors.append(relative[valid])
        native = np.concatenate(native_errors)
        cross = np.concatenate(cross_errors)
        unfiltered = np.concatenate(cross_unfiltered)
        manifest = json.loads((cache_root / "manifest.json").read_text())
        calibration = manifest["gripper_calibration"]
        points = gripper_points(raw["cartesian_position"][::2], raw["gripper_position"][::2], calibration["offset_m"])
        all_gripper = gripper_depth_residuals(points, geom["K"], geom["w2c"], geom["depths"])
        dense, _ = dense_depth_alignment(geom["depths"], geom["K"], geom["w2c"])
        return {
            "dense_cross_camera": dense,
            "episode": EPISODE,
            "real_sample": True,
            "flow_root_attributes": {
                key: value.tolist()
                if isinstance(value, np.ndarray)
                else value.item()
                if isinstance(value, np.generic)
                else value
                for key, value in flow.attrs.items()
            },
            "camera_serials": geom["serials"],
            "clips": schema,
            "source_sha256": {key: sha256(path) for key, path in paths.items()},
            "matching": manifest["matching"],
            "state_alignment": states,
            "initial_rgb_alignment": rgb_alignment,
            "timestamp_median_ms": np.median(geom["timestamp_errors_ms"], axis=0).tolist(),
            "timestamp_max_ms": geom["timestamp_errors_ms"].max(0).tolist(),
            "canonical_depth_indices": geom["depth_indices"].tolist(),
            "scene_depth": {
                "median_residual_m": float(np.median(native)),
                "observations": len(native),
                "selection": "source visibility+depth-valid,positive depth,in-image; no residual threshold",
            },
            "cross_camera": {
                "median_relative_error": float(np.median(cross)),
                "observations": len(cross),
                "selection": "source visibility+depth-valid,target in-image+positive depth,projected z<=target depth+.02 m",
                "unfiltered_median_relative_error": float(np.median(unfiltered)),
                "unfiltered_observations": len(unfiltered),
                "limitation": "target-depth occlusion filtering is not independent calibration evidence",
            },
            "gripper": {key: value for key, value in calibration.items() if key != "candidates"}
            | {
                "all_timeline_unoccluded_median_m": float(np.median(all_gripper)),
                "all_timeline_observations": len(all_gripper),
            },
            "acceptance": {
                "selected_episode_match_95pct": manifest["matching"]["rate"] >= 0.95,
                "cross_camera_3pct": dense["passed"],
                "cross_camera_tracks_3pct": float(np.median(cross)) <= 0.03,
                "visible_scene_track_1cm": float(np.median(native)) <= 0.01,
                "heldout_gripper_1p5cm": calibration["heldout_median_residual_m"] is not None
                and calibration["heldout_median_residual_m"] <= 0.015,
            },
        }


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def alignment_overlays(sample_root: Path, cache_root: Path) -> list[Path]:
    cache_root = writable_path(cache_root, DROID_CACHE_ROOT)
    paths = sample_paths(sample_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    with np.load(cache_root / "matched_raw_external.npz", allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    output = cache_root / "alignment_overlays"
    output.mkdir(exist_ok=True)
    files = []
    with h5py.File(paths["flow"], "r") as flow, h5py.File(paths["depth"], "r") as depth:
        geom = episode_geometry(flow, depth, json.loads(paths["cameras"].read_text()))
        for window in (manifest["windows"][0], manifest["windows"][-5]):
            canonical = np.array(window["canonical_indices"])
            start = int(window["clip"].split(":")[0])
            gripper = gripper_points(
                raw["cartesian_position"][2 * canonical],
                raw["gripper_position"][2 * canonical],
                manifest["gripper_calibration"]["offset_m"],
            )
            canvas = Image.new("RGB", (1920, 1770), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (20, 10),
                f"{window['split']} / clip {window['clip']} / matched raw RGB and timestamp-nearest depth",
                fill="#252a32",
                font=font(24),
            )
            draw.ellipse((22, 50, 32, 60), fill=SCENE_COLOR)
            draw.text((40, 43), "Scene tracks: circles", fill="#252a32", font=font(20))
            draw.rectangle((360, 50, 370, 60), fill=GRIPPER_COLOR)
            draw.text((380, 43), "Matched EE/finger points: squares", fill="#252a32", font=font(20))
            draw.text(
                (920, 43), "Only unoccluded points shown; next-frame projections linked", fill="#252a32", font=font(20)
            )
            for view, serial in enumerate(geom["serials"]):
                camera = flow[window["clip"]][f"camera_{serial}_ext"]
                tracks = camera["scene_flows"][canonical - start].astype(np.float32)
                masks = camera["scene_visibility"][canonical - start] & camera["scene_depth_valid_mask"][canonical - start]
                # Keep identities fixed over time; native first-time visibility determines selected circles.
                eligible = np.flatnonzero(masks[0])
                selected = eligible[:: max(1, len(eligible) // 75)][:75]
                for time, index in enumerate(canonical):
                    rgb = raw[IMAGE_KEYS[view]][2 * index]
                    dep = geom["depths"][index, view]
                    scale_lo, scale_hi = 0.25, 1.5
                    gray = (255 * (1 - np.clip((dep - scale_lo) / (scale_hi - scale_lo), 0, 1))).astype(np.uint8)
                    gray[dep == 0] = 0
                    for mode, frame in enumerate((rgb, np.repeat(gray[..., None], 3, -1))):
                        x, y = time * 640, 95 + (view * 2 + mode) * 412
                        canvas.paste(Image.fromarray(frame).resize((640, 360), Image.Resampling.NEAREST), (x, y + 45))
                        title = f"cam {serial} {'RGB' if mode == 0 else 'depth'} / canonical {index} / raw {2 * index}"
                        draw.text((x + 12, y), title, fill="#252a32", font=font(18))
                        stamp = int(geom["timestamps"][index])
                        delta = int(geom["timestamp_errors_ms"][index, view])
                        draw.text(
                            (x + 12, y + 22),
                            f"canonical ms {stamp}; nearest depth error {delta} ms",
                            fill="#525a65",
                            font=font(16),
                        )
                        for points, candidates, color, square in (
                            (tracks, selected, SCENE_COLOR, False),
                            (gripper, np.arange(32), GRIPPER_COLOR, True),
                        ):
                            valid, _, _ = depth_test(points[time], geom["K"][view], geom["w2c"][view], dep, surface=False)
                            if not square:
                                valid &= masks[time]
                            uv, _ = project(points[time], geom["K"][view], geom["w2c"][view])
                            if time < 2:
                                next_uv, _ = project(points[time + 1], geom["K"][view], geom["w2c"][view])
                            for j in candidates[valid[candidates]]:
                                px, py = x + float(uv[j, 0]) * 2, y + 45 + float(uv[j, 1]) * 2
                                if time < 2 and np.isfinite(next_uv[j]).all():
                                    nx, ny = x + float(next_uv[j, 0]) * 2, y + 45 + float(next_uv[j, 1]) * 2
                                    if x <= nx < x + 640 and y + 45 <= ny < y + 405:
                                        draw.line((px, py, nx, ny), fill=color, width=2)
                                box = (px - 4, py - 4, px + 4, py + 4)
                                if square:
                                    draw.rectangle(box, fill=color, outline="white", width=2)
                                else:
                                    draw.ellipse(box, fill=color, outline="white", width=2)
            destination = output / f"{window['split']}_{window['clip'].replace(':', '_')}.png"
            canvas.save(destination)
            files.append(destination)
    return files


def cached_coordinate_overlays(cache_root: Path) -> list[Path]:
    """Inspect actual resized cache K, source-grid displacement and gripper points."""
    cache_root = writable_path(cache_root, DROID_CACHE_ROOT)
    from s4d.data.droid.convert import RESOLUTIONS

    manifest = json.loads((cache_root / "manifest.json").read_text())
    index = next(i for i, row in enumerate(manifest["windows"]) if row["split"] == "validation")
    window = manifest["windows"][index]
    with np.load(cache_root / "matched_raw_external.npz", allow_pickle=False) as saved:
        poses = saved["cartesian_position"][window["raw_indices"]]
        closure = saved["gripper_position"][window["raw_indices"]]
    gripper = gripper_points(poses, closure, manifest["gripper_calibration"]["offset_m"])
    files = []
    for backbone, (height, width) in RESOLUTIONS.items():
        with h5py.File(cache_root / f"{backbone}.h5", "r") as cache:
            group = cache[f"windows/{index:05d}"]
            arrays = {key: group[key][:] for key in ("images", "depth", "K", "w2c", "c2w", "motion3d", "motion_weight")}
        cell_width, cell_height = width * 2, height * 2 + 58
        canvas = Image.new("RGB", (cell_width * 3, cell_height * 4 + 105), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            f"{backbone}: actual cache / continuous centers / validation {window['clip']}",
            fill="#252a32",
            font=font(24),
        )
        draw.ellipse((14, 51, 24, 61), fill=SCENE_COLOR)
        draw.text((32, 43), "Sparse source-grid motion: circles + lines", fill="#252a32", font=font(18))
        draw.rectangle((520, 51, 530, 61), fill=GRIPPER_COLOR)
        draw.text((538, 43), "Matched gripper: squares", fill="#252a32", font=font(18))
        draw.text(
            (12, 74),
            "Lines end at the next-time projection. Only source-supervised motion and unoccluded gripper points are shown.",
            fill="#525a65",
            font=font(17),
        )
        for view, serial in enumerate(manifest["camera_serials"]):
            K, w2c, c2w = arrays["K"][view], arrays["w2c"][view], arrays["c2w"][view]
            for time in range(3):
                depth = arrays["depth"][time, view, 0]
                rgb = arrays["images"][time, view].transpose(1, 2, 0)
                gray = (255 * (1 - np.clip((depth - 0.25) / 1.25, 0, 1))).astype(np.uint8)
                gray[depth == 0] = 0
                valid_grip, _, _ = depth_test(gripper[time], K, w2c, depth, surface=False, continuous=True)
                uv_grip, _ = project(gripper[time], K, w2c)
                if time < 2:
                    pair = time
                    vectors = arrays["motion3d"][pair, view]
                    supported = arrays["motion_weight"][pair, view, 0] > 0
                    dynamic = supported & (np.linalg.norm(vectors, axis=0) > 0.003)
                    yy, xx = np.nonzero(dynamic)
                    selection = np.arange(len(xx))[:: max(1, len(xx) // 50)][:50]
                    xx, yy = xx[selection], yy[selection]
                    camera = np.c_[xx + 0.5, yy + 0.5, np.ones(len(xx))] @ np.linalg.inv(K).T
                    camera *= depth[yy, xx, None]
                    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
                    displaced = world + vectors[:, yy, xx].T
                    endpoints, _ = project(displaced, K, w2c)
                for mode, frame in enumerate((rgb, np.repeat(gray[..., None], 3, -1))):
                    x, y = time * cell_width, 105 + (view * 2 + mode) * cell_height
                    draw.text(
                        (x + 8, y + 2),
                        f"cam {serial} {'RGB' if mode == 0 else 'depth'} / raw {window['raw_indices'][time]}",
                        fill="#252a32",
                        font=font(18),
                    )
                    draw.text(
                        (x + 8, y + 26),
                        "Pair 01 source" if time == 0 else "Pair 12 source" if time == 1 else "Window endpoint",
                        fill="#525a65",
                        font=font(16),
                    )
                    canvas.paste(
                        Image.fromarray(frame).resize((width * 2, height * 2), Image.Resampling.NEAREST), (x, y + 55)
                    )
                    if time < 2:
                        for px, py, end in zip(xx, yy, endpoints):
                            sx, sy = x + (float(px) + 0.5) * 2, y + 55 + (float(py) + 0.5) * 2
                            ex, ey = x + float(end[0]) * 2, y + 55 + float(end[1]) * 2
                            if x <= ex < x + cell_width and y + 55 <= ey < y + cell_height:
                                draw.line((sx, sy, ex, ey), fill=SCENE_COLOR, width=2)
                            draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=SCENE_COLOR, outline="white", width=2)
                    for uv in uv_grip[valid_grip]:
                        sx, sy = x + float(uv[0]) * 2, y + 55 + float(uv[1]) * 2
                        draw.rectangle((sx - 4, sy - 4, sx + 4, sy + 4), fill=GRIPPER_COLOR, outline="white", width=2)
        destination = cache_root / "alignment_overlays" / f"cached_coordinates_{backbone}.png"
        canvas.save(destination)
        files.append(destination)
    return files


def dense_depth_warp(source_depth, source_K, source_w2c, target_K, target_w2c, target_shape):
    """Lift every native integer-centred source pixel and nearest-z splat into the target."""
    from s4d.data.droid.pointworld import nearest_z_winners

    rows, cols = np.nonzero(np.isfinite(source_depth) & (source_depth > 0))
    pixels = np.c_[cols, rows, np.ones(len(rows))].astype(np.float32)
    camera = (pixels @ np.linalg.inv(source_K).T) * source_depth[rows, cols, None]
    c2w = np.linalg.inv(source_w2c)
    world = camera @ c2w[:3, :3].T + c2w[:3, 3]
    uv, z = project(world, target_K, target_w2c)
    finite = np.isfinite(uv).all(1) & np.isfinite(z)
    pix = np.rint(np.where(np.isfinite(uv), uv, -1)).astype(np.int64)
    height, width = target_shape
    valid = finite & (z > 0) & (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    winners = nearest_z_winners(pix, z, valid, width)
    warped = np.zeros(target_shape, np.float32)
    warped[pix[winners, 1], pix[winners, 0]] = z[winners]
    return warped


def dense_depth_alignment(depths, K, w2c):
    """Dense depth reprojection, distinct from projecting the provided scene tracks.

    Retain an unfiltered measurement beside the one-sided target-depth occlusion
    selection. Never discard in-front error or threshold error to the 3% gate.
    """
    errors, unfiltered, per_frame, snapshots = [], [], [], {}
    for time in range(len(depths)):
        for source, target in ((0, 1), (1, 0)):
            truth = depths[time, target]
            warped = dense_depth_warp(depths[time, source], K[source], w2c[source], K[target], w2c[target], truth.shape)
            overlap = (truth > 0) & (warped > 0)
            visible = overlap & (warped <= truth + 0.02)
            relative = np.abs(warped - truth) / np.maximum(truth, 1e-8)
            unfiltered.append(relative[overlap])
            errors.append(relative[visible])
            per_frame.append(
                {
                    "time": time,
                    "source": source,
                    "target": target,
                    "source_depth_valid_pixels": int((depths[time, source] > 0).sum()),
                    "target_total_pixels": int(truth.size),
                    "target_depth_valid_pixels": int((truth > 0).sum()),
                    "warped_target_pixels": int((warped > 0).sum()),
                    "excluded_target_depth_invalid_pixels": int(((warped > 0) & (truth <= 0)).sum()),
                    "excluded_occluded_pixels": int((overlap & ~visible).sum()),
                    "overlap_pixels": int(overlap.sum()),
                    "selected_pixels": int(visible.sum()),
                }
            )
            if time in (0, 50):
                snapshots[(time, source, target)] = (warped, visible, relative)
    selected, all_errors = np.concatenate(errors), np.concatenate(unfiltered)
    report = {
        "median_relative_error": float(np.median(selected)) if len(selected) else None,
        "unfiltered_median_relative_error": float(np.median(all_errors)) if len(all_errors) else None,
        "selected_target_pixels": len(selected),
        "unfiltered_target_pixels": len(all_errors),
        "threshold": 0.03,
        "passed": bool(len(selected) and np.median(selected) <= 0.03),
        "frames": len(depths),
        "directions": 2,
        "per_frame": per_frame,
        "method": "all valid source depth pixels; native integer centres; nearest target pixel; nearest-z collisions",
        "occlusion_selection": "warped_z <= target_depth + 0.02 m, retains all in-front residuals",
        "limitation": (
            "Visibility uses target teacher depth, not an independent occlusion oracle. Unfiltered median is also reported."
        ),
    }
    return report, snapshots


def dense_sanity_overlays(sample_root: Path, cache_root: Path) -> list[Path]:
    """Actual dense depth warps, fused world cloud, and sparse motion-score maps."""
    import torch

    from s4d.diag import panels as panels
    from s4d.diag.pointclouds import cloud_panel
    from s4d.geometry import lift_depth

    cache_root = writable_path(cache_root, DROID_CACHE_ROOT)
    paths = sample_paths(sample_root)
    output = cache_root / "dense_sanity"
    output.mkdir(exist_ok=True)
    with h5py.File(paths["flow"], "r") as flow, h5py.File(paths["depth"], "r") as depth:
        geom = episode_geometry(flow, depth, json.loads(paths["cameras"].read_text()))
    report, snapshots = dense_depth_alignment(geom["depths"], geom["K"], geom["w2c"])
    (output / "dense_depth_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    files = []
    for (time, source, target), (warped, visible, relative) in snapshots.items():
        maps = [
            panels.colorize_depth(torch.from_numpy(d), 0.25, 1.5, torch.from_numpy(d > 0))
            for d in (geom["depths"][time, source], geom["depths"][time, target], warped)
        ]
        error_map = panels.colorize_error(torch.from_numpy(relative), 0.1)
        error_map[~visible] = 0
        maps.append(error_map)
        panel = Image.fromarray(panels.grid(maps, ncol=4))
        labeled = Image.new("RGB", (panel.width, panel.height + 32), "white")
        labeled.paste(panel, (0, 32))
        ImageDraw.Draw(labeled).text(
            (5, 6),
            f"canonical {time}; {source} to {target}: source depth | target | dense warp | rel error 0-10%",
            fill="black",
            font=font(18),
        )
        destination = output / f"dense_warp_t{time}_src{source}.png"
        labeled.save(destination)
        files.append(destination)
    with h5py.File(cache_root / "scratch.h5", "r") as cache:
        window = cache["windows/00000"]
        dep = torch.from_numpy(window["depth"][0, :, 0])
        xyz = lift_depth(dep, torch.from_numpy(window["K"][:]), torch.from_numpy(window["c2w"][:]))
        rgb = torch.from_numpy(window["images"][0]).permute(0, 2, 3, 1).float()
        points = torch.cat((xyz[dep > 0], rgb[dep > 0]), -1).numpy()
        destination = output / "fused_gt_cloud.png"
        Image.fromarray(cloud_panel(points)).save(destination)
        files.append(destination)
        np.save(output / "fused_gt_cloud.npy", points)
        scores = window["motion_score"][:]
        maps = [panels.colorize_gray(torch.from_numpy(scores[t, v, 0])) for v in range(2) for t in range(3)]
        destination = output / "motion_scores.png"
        Image.fromarray(panels.grid(maps, ncol=3)).save(destination)
        files.append(destination)
    return files
