"""Validate collected Meta-World windows, rigid motion, and multi-camera depth.

D2 retains moving source pixels with body-matched target interpolation support.
Occlusion is tested against an independently body-warped source point-cloud
z-buffer, never by thresholding the depth residual being measured. D3 computes
nearest-neighbor distance into other-view, mutually visible, same-body clouds.
The fixed acceptance medians remain 3 mm for D2 and 5 mm for D3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from s4d.gpu_guard import enforce_allowed_gpus  # noqa: E402

GPU_MAPPING = enforce_allowed_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from s4d.data.contract import PAIRS, collate, validate_batch  # noqa: E402
from s4d.data.metaworld.dataset import TRAIN_STRIDES, MetaworldWindowDataset, list_episodes  # noqa: E402
from s4d.geometry import lift_depth, quat_wxyz_to_matrix  # noqa: E402

MOVING_M = 0.005
D2_THRESHOLD_MM = 3.0
D3_THRESHOLD_MM = 5.0
# This only resolves approximate point-cloud occlusion. It is not a residual
# acceptance cutoff, and no observed target-depth residual is filtered by it.
OCCLUSION_SLACK_M = 0.01
BODY_COLORS = {"world": "#2a78d6", "moving": "#eb6834", "other": "#1baf7a"}
BLUE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
EPISODE_FIELDS = ("rgb", "depth", "body_id", "xpos", "xquat", "obs", "qpos", "qvel", "action", "reward", "success")


def project_points(points, K, w2c):
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    uv = np.zeros((len(points), 2), dtype=np.float64)
    np.divide(camera[:, 0] * K[0, 0], z, out=uv[:, 0], where=np.abs(z) > 1e-12)
    np.divide(camera[:, 1] * K[1, 1], z, out=uv[:, 1], where=np.abs(z) > 1e-12)
    uv += K[:2, 2]
    uv[np.abs(z) <= 1e-12] = np.nan
    return uv, z


def cloud_zbuffer(points, K, w2c, height, width):
    """A source-reconstruction z-buffer independent of the measured target depth."""
    result = np.full(height * width, np.inf, dtype=np.float64)
    if len(points) == 0:
        return result.reshape(height, width)
    uv, z = project_points(points, K, w2c)
    keep = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
    keep &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    pixel = np.floor(uv[keep]).astype(np.int64)
    np.minimum.at(result, pixel[:, 1] * width + pixel[:, 0], z[keep])
    return result.reshape(height, width)


def visible_surface(points, bodies, K, w2c, target_depth, target_body, independent_zbuffer):
    """Visibility and bilinear support without filtering the measured residual.

    Four depth neighbors must be valid and have the same body ID. Pixel centers
    are at .5, so interpolation indices are floor(u-.5), floor(v-.5). Exclusion
    buckets are exclusive and partition every projected source comparison.
    """
    H, W = target_depth.shape
    uv, z = project_points(points, K, w2c)
    keep = np.ones(len(points), dtype=bool)
    counts = {"comparisons": int(len(points))}

    def retain(condition, reason):
        nonlocal keep
        counts[reason] = int(np.count_nonzero(keep & ~condition))
        keep &= condition

    retain(np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0), "nonfinite_or_behind")
    retain(
        (uv[:, 0] >= 0.5) & (uv[:, 0] <= W - 0.5) & (uv[:, 1] >= 0.5) & (uv[:, 1] <= H - 0.5),
        "outside_interpolation_support",
    )
    safe_uv = np.nan_to_num(uv, nan=0.5, posinf=0.5, neginf=0.5)
    continuous = np.clip(safe_uv - 0.5, [0, 0], [W - 1, H - 1])
    x0, y0 = np.floor(continuous).astype(np.int64).T
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    depths = np.stack((target_depth[y0, x0], target_depth[y0, x1], target_depth[y1, x0], target_depth[y1, x1]), 1)
    ids = np.stack((target_body[y0, x0], target_body[y0, x1], target_body[y1, x0], target_body[y1, x1]), 1)
    retain((np.isfinite(depths) & (depths > 0)).all(1), "invalid_target_depth_support")
    retain((ids == bodies[:, None]).all(1), "target_body_or_boundary_mismatch")
    nearest_x = np.clip(np.floor(safe_uv[:, 0]).astype(np.int64), 0, W - 1)
    nearest_y = np.clip(np.floor(safe_uv[:, 1]).astype(np.int64), 0, H - 1)
    front = independent_zbuffer[nearest_y, nearest_x]
    retain(~np.isfinite(front) | (z <= front + OCCLUSION_SLACK_M), "independent_cloud_occlusion")
    fx, fy = continuous[:, 0] - x0, continuous[:, 1] - y0
    interpolated = (
        depths[:, 0] * (1 - fx) * (1 - fy)
        + depths[:, 1] * fx * (1 - fy)
        + depths[:, 2] * (1 - fx) * fy
        + depths[:, 3] * fx * fy
    )
    counts["retained"] = int(np.count_nonzero(keep))
    return keep, interpolated, z, counts


def _body_warp(xyz, body, xpos_a, quat_a, xpos_b, quat_b, background):
    ra = quat_wxyz_to_matrix(torch.as_tensor(quat_a, dtype=torch.float64)).numpy()
    rb = quat_wxyz_to_matrix(torch.as_tensor(quat_b, dtype=torch.float64)).numpy()
    relative = rb @ np.swapaxes(ra, -1, -2)
    translation = xpos_b - np.einsum("bij,bj->bi", relative, xpos_a)
    safe = np.where(body == background, 0, body).astype(np.int64)
    result = np.einsum("...ij,...j->...i", relative[safe], xyz) + translation[safe]
    # Background has no body and never enters the validation cloud.
    return result


def _select(mask, limit, rng):
    indices = np.flatnonzero(mask)
    if len(indices) > limit:
        indices = np.sort(rng.choice(indices, limit, replace=False))
    return indices


def metric_summary(values, threshold_mm):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "median_mm": None,
            "p90_mm": None,
            "threshold_mm": threshold_mm,
            "passed": False,
            "status": "insufficient_support",
        }
    median = float(np.median(values) * 1000)
    return {
        "count": int(len(values)),
        "median_mm": median,
        "p90_mm": float(np.percentile(values, 90) * 1000),
        "threshold_mm": threshold_mm,
        "passed": median <= threshold_mm,
        "status": "pass" if median <= threshold_mm else "fail",
    }


def check_window(sample, ep, train_cams, background, limit, rng):
    t_indices = sample["meta"]["t_indices"]
    depth = sample["depth"][:, :, 0].numpy().astype(np.float64)
    body = ep["body_id"][t_indices].astype(np.int64)
    xpos, quat = ep["xpos"][t_indices].astype(np.float64), ep["xquat"][t_indices].astype(np.float64)
    K, c2w, w2c = (sample[key].numpy().astype(np.float64) for key in ("K", "c2w", "w2c"))
    V, H, W = depth.shape[1:]
    if V != len(train_cams):
        raise ValueError("body-ID cameras and training-camera table disagree")
    valid = (depth > 0) & (body != background)
    xyz = lift_depth(
        torch.from_numpy(depth), torch.from_numpy(K[None].repeat(3, 0)), torch.from_numpy(c2w[None].repeat(3, 0))
    ).numpy()
    record = {
        "episode": sample["meta"]["episode"],
        "t_indices": t_indices,
        "stride": sample["meta"]["stride"],
        "d2_comparisons": [],
        "d3_pairs": [],
        "d3_views": [],
    }
    d2_errors, d3_errors = [], []
    world_count, world_nonzero, equation_error = 0, 0, 0.0
    for pair_index, (a, b, _) in enumerate(PAIRS):
        warped = _body_warp(xyz[a], body[a], xpos[a], quat[a], xpos[b], quat[b], background)
        expected = warped - xyz[a]
        target = sample["motion3d"][pair_index].permute(0, 2, 3, 1).numpy().astype(np.float64)
        world = valid[a] & (body[a] == 0)
        world_count += int(np.count_nonzero(world))
        world_nonzero += int(np.count_nonzero(np.any(target[world] != 0, axis=-1)))
        if valid[a].any():
            equation_error = max(equation_error, float(np.abs(target[valid[a]] - expected[valid[a]]).max()))
        cloud = warped[valid[a]]
        zbuffers = [cloud_zbuffer(cloud, K[j], w2c[j], H, W) for j in range(V)]
        for source in range(V):
            moving = valid[a, source] & (np.linalg.norm(expected[source], axis=-1) > MOVING_M)
            chosen = _select(moving.ravel(), limit, rng)
            source_xyz = xyz[a, source].reshape(-1, 3)[chosen]
            predicted = source_xyz + target[source].reshape(-1, 3)[chosen]
            bodies = body[a, source].ravel()[chosen]
            for other in range(V):
                if other == source:
                    continue
                keep, observed, z, counts = visible_surface(
                    predicted, bodies, K[other], w2c[other], depth[b, other], body[b, other], zbuffers[other]
                )
                errors = np.abs(observed[keep] - z[keep])
                d2_errors.extend(errors.tolist())
                counts.update(
                    {
                        "pair": f"{a}{b}",
                        "source_camera": int(train_cams[source]),
                        "target_camera": int(train_cams[other]),
                        "moving_source_pixels": int(moving.sum()),
                        "not_sampled": int(moving.sum()) - len(chosen),
                        "all_source_pixels": int(H * W),
                        "valid_source_pixels": int(valid[a, source].sum()),
                        "static_source_pixels": int(valid[a, source].sum() - moving.sum()),
                        "invalid_or_unassigned_source_pixels": int(H * W - valid[a, source].sum()),
                        "metric": metric_summary(errors, D2_THRESHOLD_MM),
                    }
                )
                record["d2_comparisons"].append(counts)
    for t in range(3):
        # Every z-buffer excludes the camera whose actual depth will be tested.
        zbuffers = []
        for camera in range(V):
            independent = [xyz[t, other][valid[t, other]] for other in range(V) if other != camera]
            cloud = np.concatenate(independent) if independent else np.empty((0, 3))
            zbuffers.append(cloud_zbuffer(cloud, K[camera], w2c[camera], H, W))
        for source in range(V):
            chosen = _select(valid[t, source].ravel(), limit, rng)
            points = xyz[t, source].reshape(-1, 3)[chosen]
            bodies = body[t, source].ravel()[chosen]
            nearest = np.full(len(chosen), np.inf)
            for other in range(V):
                if other == source:
                    continue
                keep, _, _, forward_counts = visible_surface(
                    points, bodies, K[other], w2c[other], depth[t, other], body[t, other], zbuffers[other]
                )
                targets = xyz[t, other][valid[t, other]]
                target_bodies = body[t, other][valid[t, other]]
                reciprocal, _, _, reverse_counts = visible_surface(
                    targets, target_bodies, K[source], w2c[source], depth[t, source], body[t, source], zbuffers[source]
                )
                distances, no_support = [], 0
                for body_id in np.unique(bodies[keep]):
                    query = np.flatnonzero(keep & (bodies == body_id))
                    support = targets[reciprocal & (target_bodies == body_id)]
                    if len(support) == 0:
                        no_support += len(query)
                        continue
                    distance = cKDTree(support).query(points[query], k=1, workers=1)[0]
                    nearest[query] = np.minimum(nearest[query], distance)
                    distances.extend(distance.tolist())
                record["d3_pairs"].append(
                    {
                        "time": int(t_indices[t]),
                        "source_camera": int(train_cams[source]),
                        "target_camera": int(train_cams[other]),
                        "source_visibility": forward_counts,
                        "target_reciprocal_visibility": reverse_counts,
                        "no_same_body_mutual_target_support": int(no_support),
                        "metric": metric_summary(distances, D3_THRESHOLD_MM),
                    }
                )
            retained = np.isfinite(nearest)
            d3_errors.extend(nearest[retained].tolist())
            record["d3_views"].append(
                {
                    "time": int(t_indices[t]),
                    "source_camera": int(train_cams[source]),
                    "all_source_pixels": int(H * W),
                    "valid_source_pixels": int(valid[t, source].sum()),
                    "sampled": len(chosen),
                    "not_sampled": int(valid[t, source].sum()) - len(chosen),
                    "no_other_view_mutual_support": int((~retained).sum()),
                    "metric": metric_summary(nearest[retained], D3_THRESHOLD_MM),
                }
            )
    record["world_body_pixels"] = world_count
    record["nonzero_world_body_pixels"] = world_nonzero
    record["motion_equation_max_error_m"] = equation_error
    record["d2"] = metric_summary(d2_errors, D2_THRESHOLD_MM)
    record["d3"] = metric_summary(d3_errors, D3_THRESHOLD_MM)
    return record, d2_errors, d3_errors, body, xpos


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _sequential(values):
    ramp = np.asarray([_rgb(color) for color in BLUE_RAMP], dtype=np.float64)
    x = np.nan_to_num(values).clip(0, 1) * (len(ramp) - 1)
    lower = np.floor(x).astype(np.int64)
    upper = np.minimum(lower + 1, len(ramp) - 1)
    fraction = (x - lower)[..., None]
    return ((1 - fraction) * ramp[lower] + fraction * ramp[upper]).round().astype(np.uint8)


def save_panel(sample, body, xpos, background, out):
    """Small multiples use one sequential hue and three labeled body categories."""
    depth = sample["depth"][:, :, 0].numpy()
    images = sample["images"].permute(0, 1, 3, 4, 2).numpy()
    score = sample["motion_score"][:, :, 0].numpy()
    T, V, H, W = depth.shape
    valid_depth = depth[depth > 0]
    near, far = np.quantile(valid_depth, [0.01, 0.99]) if len(valid_depth) else (0.0, 1.0)
    far = max(float(far), float(near) + 1e-6)
    movement = np.linalg.norm(xpos[-1] - xpos[0], axis=-1) > MOVING_M
    # Rotation-only motion is included using the actual loader displacement maps.
    motion = sample["motion3d"].norm(dim=2).numpy()
    for view in range(V):
        ids = body[0, view][motion[2, view] > MOVING_M]
        ids = ids[(ids != background) & (ids < len(movement))]
        movement[ids] = True
    padding, label, top, footer = 8, 124, 36, 76
    canvas = Image.new("RGB", (label + V * (W + padding) + padding, top + T * 4 * (H + padding) + footer), "#fcfcfb")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for view in range(V):
        draw.text((label + view * (W + padding), 12), f"train camera {view}", fill="#0b0b0b", font=font)
    for t in range(T):
        for view in range(V):
            ids = body[t, view]
            mask = np.full((H, W, 3), 225, dtype=np.uint8)
            mask[ids == 0] = _rgb(BODY_COLORS["world"])
            mask[(ids != 0) & (ids != background)] = _rgb(BODY_COLORS["other"])
            moving_ids = np.where(movement)[0]
            mask[np.isin(ids, moving_ids) & (ids != 0)] = _rgb(BODY_COLORS["moving"])
            depth_panel = _sequential((depth[t, view] - near) / (far - near))
            depth_panel[depth[t, view] <= 0] = 225
            motion_panel = _sequential(score[t, view])
            motion_panel[depth[t, view] <= 0] = 225
            panels = (images[t, view], depth_panel, mask, motion_panel)
            for kind, array in enumerate(panels):
                row = t * 4 + kind
                x, y = label + view * (W + padding), top + row * (H + padding)
                cell = Image.fromarray(array)
                if kind == 2:
                    annotate = ImageDraw.Draw(cell)
                    keys, counts = np.unique(ids[ids != background], return_counts=True)
                    occupied = []
                    for body_id in keys[np.argsort(-counts)[:10]]:
                        yy, xx = np.where(ids == body_id)
                        middle = np.array([np.median(xx), np.median(yy)])
                        closest = np.argmin((xx - middle[0]) ** 2 + (yy - middle[1]) ** 2)
                        position = (max(0, min(W - 22, int(xx[closest]))), max(0, min(H - 14, int(yy[closest]))))
                        text = str(body_id)
                        box = annotate.textbbox(position, text, font=font)
                        if any(
                            box[0] < other[2] + 2
                            and box[2] + 2 > other[0]
                            and box[1] < other[3] + 2
                            and box[3] + 2 > other[1]
                            for other in occupied
                        ):
                            continue
                        occupied.append(box)
                        annotate.rectangle(box, fill="#fcfcfb")
                        annotate.text(position, text, fill="#0b0b0b", font=font)
                    Image.fromarray(ids.astype(np.uint16)).save(out / f"body_id_t{t}_camera{view}.png")
                canvas.paste(cell, (x, y))
                if view == 0:
                    title = ("RGB", "depth", "body ID", "motion score")[kind]
                    draw.text((8, y + 4), f"t={sample['meta']['t_indices'][t]} {title}", fill="#0b0b0b", font=font)
    y = top + T * 4 * (H + padding) + 4
    draw.text(
        (8, y),
        f"Depth: {near:.3f} to {far:.3f} m. Motion: 0 to 3 cm max displacement. Invalid: gray.",
        fill="#0b0b0b",
        font=font,
    )
    for index, (name, color) in enumerate(BODY_COLORS.items()):
        x = 8 + index * 180
        draw.rectangle((x, y + 24, x + 12, y + 36), fill=color)
        draw.text((x + 18, y + 24), f"{name} body; IDs labeled", fill="#0b0b0b", font=font)
    draw.text(
        (8, y + 46),
        "Sequential scales: light is low. Dark is high. Body ID PNGs retain every identity.",
        fill="#0b0b0b",
        font=font,
    )
    canvas.save(out / "sanity.png")


def _window_indices(dataset, number):
    # Balance episodes before spending extra windows on an episode. Selection
    # depends on timeline only, not the measured geometry error.
    by_episode = {}
    for index, (episode, _, _) in enumerate(dataset.samples):
        by_episode.setdefault(episode, []).append(index)
    selected = []
    per_episode = max(1, int(np.ceil(number / len(by_episode))))
    for indices in by_episode.values():
        positions = np.linspace(0, len(indices) - 1, min(per_episode, len(indices))).round().astype(int)
        selected.extend(indices[position] for position in positions)
    if len(selected) > number:
        positions = np.linspace(0, len(selected) - 1, number).round().astype(int)
        selected = [selected[position] for position in positions]
    return selected


def check_dataset(path, out, windows=12, points_per_view=2048, seed=0):
    path, out = Path(path).resolve(), Path(out).resolve()
    if REPO not in out.parents:
        raise ValueError("sanity output must be a child directory of the new repository")
    if windows < 1 or points_per_view < 1:
        raise ValueError("windows and points_per_view must be positive")
    out.mkdir(parents=True, exist_ok=True)
    dataset = MetaworldWindowDataset(path, list_episodes(path), strides=TRAIN_STRIDES, with_eval=True)
    rng = np.random.default_rng(seed)
    records, errors, d2, d3 = [], [], [], []
    report = {
        "path": str(path),
        "seed": seed,
        "gpu_mapping": GPU_MAPPING,
        "moving_threshold_m": MOVING_M,
        "independent_occlusion_slack_m": OCCLUSION_SLACK_M,
        "metric_definitions": {
            "D2": "Absolute camera-z residual at bilinear same-body target support for body-warped moving pixels.",
            "D3": "Per-view nearest neighbor in fused other-camera mutually visible same-body point clouds.",
            "visibility": "No measured depth residual or nearest-neighbor distance is used to filter acceptance.",
            "pixel_centers": "u=column+0.5, v=row+0.5",
        },
        "denominator_units": {
            "D2": "Temporal-pair/source-camera/target-camera comparisons. Source pixels repeat across target views.",
            "D3": "Timestep/source-camera sampled points; each point counts once after fusion of other views.",
            "world_body_zero": "Valid world-body source pixels tested per supervised temporal pair.",
        },
        "limitations": [
            "Occlusion uses finite-resolution point clouds, not ground-truth triangle ray casting.",
            "Surfaces absent from every source view cannot occlude the reconstructed source cloud.",
            "Four-neighbor same-body support excludes boundaries and thin subpixel surfaces.",
        ],
    }
    try:
        with h5py.File(path, "r") as file:
            if not bool(file.attrs.get("complete", False)):
                raise ValueError("collected file is not marked complete")
            report["task"] = str(file.attrs["task"])
            report["body_names"] = json.loads(file.attrs["body_names"])
            for episode in dataset.episodes:
                group = file["episodes"][episode]
                for key in EPISODE_FIELDS:
                    if key not in group or group[key].shape[0] != int(group.attrs["length"]):
                        raise ValueError(f"episode {episode} field {key} is missing or has the wrong temporal length")
            for index in _window_indices(dataset, min(windows, len(dataset))):
                sample = dataset[index]
                try:
                    validate_batch(collate([sample]), training=False)
                except Exception as exc:
                    errors.append({"index": index, "error": repr(exc)})
                    continue
                result, e2, e3, bodies, poses = check_window(
                    sample,
                    file["episodes"][sample["meta"]["episode"]],
                    dataset.train_cams,
                    dataset.background_body,
                    points_per_view,
                    rng,
                )
                records.append(result)
                d2.extend(e2)
                d3.extend(e3)
                if len(records) == 1:
                    save_panel(sample, bodies, poses, dataset.background_body, out)
                    report["panel"] = str(out / "sanity.png")
        world_count = sum(record["world_body_pixels"] for record in records)
        nonzero = sum(record["nonzero_world_body_pixels"] for record in records)
        report.update(
            {
                "windows_checked": len(records),
                "contract_errors": errors,
                "windows": records,
                "d2": metric_summary(d2, D2_THRESHOLD_MM),
                "d3": metric_summary(d3, D3_THRESHOLD_MM),
                "world_body_zero": {
                    "pixels": world_count,
                    "nonzero_pixels": nonzero,
                    "passed": world_count > 0 and nonzero == 0,
                    "status": "insufficient_support" if world_count == 0 else ("pass" if nonzero == 0 else "fail"),
                },
            }
        )
        exclusion_keys = (
            "comparisons",
            "nonfinite_or_behind",
            "outside_interpolation_support",
            "invalid_target_depth_support",
            "target_body_or_boundary_mismatch",
            "independent_cloud_occlusion",
            "retained",
            "not_sampled",
        )
        report["d2_denominators_and_exclusions"] = {
            key: sum(comparison[key] for record in records for comparison in record["d2_comparisons"])
            for key in exclusion_keys
        }
        report["d3_denominators_and_exclusions"] = {
            key: sum(view[key] for record in records for view in record["d3_views"])
            for key in ("all_source_pixels", "valid_source_pixels", "sampled", "not_sampled", "no_other_view_mutual_support")
        }
        report["passed"] = (
            not errors and report["d2"]["passed"] and report["d3"]["passed"] and report["world_body_zero"]["passed"]
        )
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        if dataset._file is not None:
            dataset._file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--points-per-view", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = check_dataset(args.path, args.out, args.windows, args.points_per_view, args.seed)
    print(
        json.dumps(
            {key: report[key] for key in ("task", "windows_checked", "d2", "d3", "world_body_zero", "passed")}, indent=2
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
