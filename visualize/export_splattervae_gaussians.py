from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.splatter import render_predicted
from visualize.splattervae_common import build_visualization_models


def image_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    img = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img * 2.0 - 1.0).permute(2, 0, 1)


def invert_4x4(m: np.ndarray) -> np.ndarray:
    r = m[:3, :3]
    t = m[:3, 3:4]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = r.T
    out[:3, 3:4] = -(r.T @ t)
    return out


def sort_demo_keys(keys: Sequence[str]) -> List[str]:
    def key_fn(x: str) -> int:
        try:
            return int(x.replace("demo", ""))
        except Exception:
            return 10**9

    return sorted(keys, key=key_fn)


def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def parse_csv(value: str | None) -> List[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_timesteps(value: str | None, t_len: int, max_exports: int | None) -> List[int]:
    if value is None:
        steps = list(range(t_len))
    else:
        steps = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                parts = [int(p) if p else None for p in chunk.split(":")]
                if len(parts) > 3:
                    raise ValueError(f"Invalid timestep range: {chunk!r}")
                start = 0 if parts[0] is None else parts[0]
                stop = t_len if len(parts) < 2 or parts[1] is None else parts[1]
                stride = 1 if len(parts) < 3 or parts[2] is None else parts[2]
                steps.extend(range(start, stop, stride))
            else:
                steps.append(int(chunk))

    valid = []
    seen = set()
    for step in steps:
        if not (0 <= step < t_len):
            raise ValueError(f"Timestep {step} is outside valid range [0, {t_len - 1}].")
        if step not in seen:
            valid.append(step)
            seen.add(step)
        if max_exports is not None and len(valid) >= max_exports:
            break
    return valid


def load_camera_mats(demo: h5py.Group, cam_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    intr = np.asarray(demo["camera_params"]["intrinsics"], dtype=np.float32)
    world_t_cam = np.asarray(demo["camera_params"]["extrinsics_world_T_cam"], dtype=np.float32)
    camera_names_all = json.loads(demo.attrs["camera_names"])
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    mats = {}
    for cam in cam_names:
        cam_idx = camera_names_all.index(cam)
        k = intr[cam_idx]
        w2c_gl = invert_4x4(world_t_cam[cam_idx])
        w2c = gl_to_cv @ w2c_gl
        c2w = invert_4x4(w2c).astype(np.float32)
        mats[cam] = {"K": k, "w2c": w2c.astype(np.float32), "c2w": c2w}
    return mats


def rotation_matrix_to_quaternion_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (...,3,3), got {tuple(matrix.shape)}")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clamp(min=0.1))
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(dtype=torch.bool)
    quat = quat_candidates[best, :].reshape(*matrix.shape[:-2], 4)
    return F.normalize(quat, dim=-1, eps=1e-6)


def sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


SH_C0 = 0.28209479177387814


def gaussian_property_arrays(
    pc: Dict[str, torch.Tensor],
    store_mode: str,
    color_mode: str,
    opacity_eps: float = 1e-6,
) -> Tuple[List[Tuple[str, np.ndarray]], int]:
    xyz = pc["xyz"][0].detach().cpu().float().numpy()
    opacity = pc["opacity"][0, :, 0].detach().cpu().float().numpy()
    scaling = pc["scaling"][0].detach().cpu().float().numpy()
    rotation = pc["rotation"][0].detach().cpu().float().numpy()
    features_dc = pc["features_dc"][0].detach().cpu().float().numpy()  # (N,1,3)
    features_rest = pc.get("features_rest", torch.empty(1, xyz.shape[0], 0, 3))[0].detach().cpu().float().numpy()

    opacity = np.nan_to_num(opacity, nan=0.0, posinf=1.0, neginf=0.0)
    scaling = np.nan_to_num(scaling, nan=1e-6, posinf=1e2, neginf=1e-6)
    rotation = np.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
    rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation = rotation / np.maximum(rotation_norm, 1e-8)

    if store_mode == "graphdeco":
        opacity = np.log(np.clip(opacity, opacity_eps, 1.0 - opacity_eps) / np.clip(1.0 - opacity, opacity_eps, 1.0))
        scaling = np.log(np.clip(scaling, 1e-8, None))
    elif store_mode != "activated":
        raise ValueError(f"Unknown store_mode={store_mode!r}")

    if color_mode == "sh":
        f_dc_src = features_dc
    elif color_mode == "rgb":
        # Some viewers treat f_dc as RGB-like color.  For standard 3DGS tools,
        # use color_mode=sh so f_dc remains a spherical-harmonic coefficient.
        f_dc_src = np.clip(features_dc * SH_C0 + 0.5, 0.0, 1.0)
    elif color_mode == "rgb_to_sh":
        rgb = np.clip(features_dc, 0.0, 1.0)
        f_dc_src = (rgb - 0.5) / SH_C0
    else:
        raise ValueError(f"Unknown color_mode={color_mode!r}")

    f_dc = np.transpose(f_dc_src, (0, 2, 1)).reshape(xyz.shape[0], -1)
    if features_rest.size > 0:
        f_rest = np.transpose(features_rest, (0, 2, 1)).reshape(xyz.shape[0], -1)
    else:
        f_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    props: List[Tuple[str, np.ndarray]] = []
    props.extend((name, xyz[:, i]) for i, name in enumerate(("x", "y", "z")))
    props.extend((name, np.zeros(xyz.shape[0], dtype=np.float32)) for name in ("nx", "ny", "nz"))
    props.extend((f"f_dc_{i}", f_dc[:, i]) for i in range(f_dc.shape[1]))
    props.extend((f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1]))
    props.append(("opacity", opacity))
    props.extend((f"scale_{i}", scaling[:, i]) for i in range(scaling.shape[1]))
    props.extend((f"rot_{i}", rotation[:, i]) for i in range(rotation.shape[1]))
    return props, xyz.shape[0]


def filter_gaussians(
    pc: Dict[str, torch.Tensor],
    opacity_threshold: float,
    max_gaussians: int | None,
    visibility_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    opacities = pc["opacity"][0, :, 0]
    keep = opacities >= float(opacity_threshold)
    if visibility_mask is not None:
        keep = keep & visibility_mask.to(device=keep.device, dtype=torch.bool)
    if max_gaussians is not None and int(max_gaussians) > 0 and int(max_gaussians) < int(keep.sum().item()):
        kept_scores = opacities.masked_fill(~keep, -1.0)
        top_idx = torch.topk(kept_scores, k=int(max_gaussians), largest=True).indices
        new_keep = torch.zeros_like(keep)
        new_keep[top_idx] = True
        keep = new_keep

    if not bool(keep.any()):
        raise ValueError("No Gaussians remain after filtering. Lower --opacity_threshold or increase --max_gaussians.")

    out = {}
    for key, value in pc.items():
        if value.dim() >= 2 and value.shape[0] == 1 and value.shape[1] == keep.shape[0]:
            out[key] = value[:, keep].contiguous()
        else:
            out[key] = value
    return out


def write_gaussian_ply(path: Path, pc: Dict[str, torch.Tensor], store_mode: str, color_mode: str) -> int:
    props, count = gaussian_property_arrays(pc, store_mode=store_mode, color_mode=color_mode)
    dtype = np.dtype([(name, "<f4") for name, _arr in props])
    vertices = np.empty(count, dtype=dtype)
    for name, arr in props:
        vertices[name] = np.asarray(arr, dtype=np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated by visualize/export_splattervae_gaussians.py",
        "comment Compatible with common 3D Gaussian Splatting PLY field names",
        f"element vertex {count}",
    ]
    header_lines.extend(f"property float {name}" for name, _arr in props)
    header_lines.append("end_header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertices.tofile(f)
    return count


@torch.no_grad()
def generate_gaussians_for_source(
    vae,
    converter,
    image_u8: np.ndarray,
    source_k: np.ndarray,
    source_c2w: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    x = image_to_tensor(image_u8).unsqueeze(0).to(device)
    z_inv, _, z_dep, _, _ = vae.encode(
        x,
        deterministic_invariant=True,
        deterministic_dependent=True,
    )
    splatter = vae.decode(z_inv.contiguous(), z_dep.contiguous())

    k = torch.from_numpy(source_k).unsqueeze(0).to(device=device, dtype=torch.float32)
    c2w = torch.from_numpy(source_c2w).unsqueeze(0).to(device=device, dtype=torch.float32)
    source_quat = rotation_matrix_to_quaternion_wxyz(c2w[:, :3, :3])
    return converter(
        splatter=splatter,
        source_cameras_view_to_world=c2w,
        source_cv2wT_quat=source_quat,
        intrinsics=k,
        activate_output=True,
    )


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    demo_key: str,
    cam: str,
    timestep: int,
    num_gaussians: int,
    store_mode: str,
    source_k: np.ndarray,
    source_c2w: np.ndarray,
) -> None:
    metadata = {
        "dataset": args.dataset,
        "checkpoint": args.ckpt,
        "config": args.config,
        "demo": demo_key,
        "camera": cam,
        "timestep": int(timestep),
        "num_gaussians": int(num_gaussians),
        "ply_store_mode": store_mode,
        "ply_color_mode": args.color_mode,
        "opacity_threshold": float(args.opacity_threshold),
        "max_gaussians": args.max_gaussians,
        "visibility_filter": args.visibility_filter,
        "coordinate_frame": "world, OpenCV camera convention converted from dataset extrinsics_world_T_cam",
        "source_intrinsics": source_k.tolist(),
        "source_camera_to_world": source_c2w.tolist(),
        "fields": "GraphDeco/3DGS-style x,y,z,nx,ny,nz,f_dc_*,f_rest_*,opacity,scale_*,rot_*",
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SplatterVAE-generated 3D Gaussians as 3DGS-compatible PLY files.")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--dataset", required=True, help="Multi-view HDF5 dataset.")
    parser.add_argument("--ckpt", required=True, help="SplatterVAE checkpoint.")
    parser.add_argument("--out_dir", default="gaussian_exports", help="Directory for exported .ply files.")
    parser.add_argument("--demo", default=None, help="Demo key. Defaults to first demo.")
    parser.add_argument("--cameras", default=None, help="Comma-separated cameras. Defaults to first selected camera.")
    parser.add_argument(
        "--timesteps",
        default="0",
        help="Comma-separated timesteps/ranges, e.g. '0,10,20' or '0:100:10'. Defaults to 0.",
    )
    parser.add_argument("--max_exports", type=int, default=None, help="Maximum number of timestep exports per camera.")
    parser.add_argument("--opacity_threshold", type=float, default=0.02, help="Drop Gaussians below this activated opacity.")
    parser.add_argument("--max_gaussians", type=int, default=None, help="Keep only top-k Gaussians by opacity after thresholding.")
    parser.add_argument(
        "--store_mode",
        choices=["graphdeco", "activated"],
        default="graphdeco",
        help="graphdeco stores logit(opacity) and log(scale), matching common 3DGS PLY tools. activated stores raw activated values.",
    )
    parser.add_argument(
        "--color_mode",
        choices=["sh", "rgb", "rgb_to_sh"],
        default="sh",
        help="How to write f_dc values. 'sh' preserves renderer SH coefficients and is standard for GraphDeco/3DGS viewers.",
    )
    parser.add_argument(
        "--visibility_filter",
        choices=["source", "none"],
        default="source",
        help="When source, render from the source camera and export only splats with positive source-view radius.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device)
    if args.visibility_filter == "source" and device.type != "cuda":
        print(
            "Warning: --visibility_filter source requires CUDA gsplat rendering; "
            "falling back to --visibility_filter none for CPU export.",
            file=sys.stderr,
        )
        args.visibility_filter = "none"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
        demo_key = args.demo or demo_keys[0]
        if demo_key not in f["data"]:
            raise ValueError(f"Demo {demo_key!r} not found. Available examples: {demo_keys[:5]}")
        demo = f["data"][demo_key]
        all_cameras = json.loads(demo.attrs["camera_names"])
        cfg_camera_num = cfg.get("dataset", {}).get("camera_num", None)
        selected_cameras = all_cameras[: int(cfg_camera_num)] if cfg_camera_num is not None else all_cameras
        requested_cameras = parse_csv(args.cameras)
        cameras = requested_cameras or [selected_cameras[0]]
        for cam in cameras:
            if cam not in all_cameras:
                raise ValueError(f"Camera {cam!r} not in dataset cameras {all_cameras}.")
        t_len = int(demo.attrs.get("num_samples", demo["obs"][f"{cameras[0]}_rgb"].shape[0]))
        timesteps = parse_timesteps(args.timesteps, t_len=t_len, max_exports=args.max_exports)

    vae, converter, spl_cfg = build_visualization_models(cfg, args.dataset, demo_key, args.ckpt, device)
    bg = torch.ones(3, device=device) if spl_cfg.data.white_background else torch.zeros(3, device=device)

    exported = []
    with h5py.File(args.dataset, "r") as f:
        demo = f["data"][demo_key]
        mats = load_camera_mats(demo, cameras)
        obs = demo["obs"]
        for cam in cameras:
            for timestep in timesteps:
                image_u8 = np.asarray(obs[f"{cam}_rgb"][timestep], dtype=np.uint8)
                pc = generate_gaussians_for_source(
                    vae=vae,
                    converter=converter,
                    image_u8=image_u8,
                    source_k=mats[cam]["K"],
                    source_c2w=mats[cam]["c2w"],
                    device=device,
                )
                visibility_mask = None
                if args.visibility_filter == "source":
                    source_w2c = torch.from_numpy(mats[cam]["w2c"]).view(1, 1, 4, 4).to(device=device, dtype=torch.float32)
                    source_k = torch.from_numpy(mats[cam]["K"]).view(1, 1, 3, 3).to(device=device, dtype=torch.float32)
                    render_out = render_predicted(
                        pc=pc,
                        world_view_transform=source_w2c,
                        intrinsics=source_k,
                        bg_color=bg,
                        cfg=spl_cfg,
                    )
                    radii = render_out.get("radii", None)
                    if radii is not None:
                        radii = radii.detach()
                        n_gaussians = pc["opacity"].shape[1]
                        if radii.numel() == n_gaussians:
                            visibility_mask = radii.reshape(-1) > 0
                        elif radii.shape[-1] == n_gaussians:
                            visibility_mask = (radii > 0).reshape(-1, n_gaussians).any(dim=0)
                        elif radii.shape[-2] == n_gaussians:
                            visibility_mask = (radii > 0).transpose(-1, -2).reshape(-1, n_gaussians).any(dim=0)
                        else:
                            raise ValueError(f"Unexpected radii shape {tuple(radii.shape)} for {n_gaussians} Gaussians.")

                pc = filter_gaussians(
                    pc,
                    opacity_threshold=args.opacity_threshold,
                    max_gaussians=args.max_gaussians,
                    visibility_mask=visibility_mask,
                )

                stem = f"{sanitize_filename(demo_key)}_{sanitize_filename(cam)}_t{int(timestep):06d}"
                ply_path = out_dir / f"{stem}.ply"
                meta_path = out_dir / f"{stem}.json"
                num_gaussians = write_gaussian_ply(ply_path, pc, store_mode=args.store_mode, color_mode=args.color_mode)
                write_metadata(
                    meta_path,
                    args=args,
                    demo_key=demo_key,
                    cam=cam,
                    timestep=timestep,
                    num_gaussians=num_gaussians,
                    store_mode=args.store_mode,
                    source_k=mats[cam]["K"],
                    source_c2w=mats[cam]["c2w"],
                )
                exported.append(str(ply_path))
                print(f"Saved {ply_path} ({num_gaussians} Gaussians)")

    manifest = {
        "dataset": args.dataset,
        "checkpoint": args.ckpt,
        "config": args.config,
        "demo": demo_key,
        "cameras": cameras,
        "timesteps": timesteps,
        "store_mode": args.store_mode,
        "color_mode": args.color_mode,
        "opacity_threshold": float(args.opacity_threshold),
        "max_gaussians": args.max_gaussians,
        "visibility_filter": args.visibility_filter,
        "exports": exported,
    }
    manifest_path = out_dir / f"{sanitize_filename(demo_key)}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
