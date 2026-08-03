from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch


DEFAULT_GAPS = (3, 6, 9, 12, 18)
PREPROCESSING_VERSION = "sea-raft-hdf5-v1"
DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEA_RAFT_ROOT = DATASET_ROOT / "third_party" / "SEA-RAFT"
DEFAULT_CONFIG = DEFAULT_SEA_RAFT_ROOT / "config" / "eval" / "spring-M.json"
DEFAULT_CHECKPOINT = (
    DATASET_ROOT / "assets" / "sea_raft" / "Tartan-C-T-TSKH-spring540x960-M.pth"
)


def _camera_names(demo_group: h5py.Group) -> list[str]:
    value = demo_group.attrs.get("camera_names", None)
    if value is None:
        raise ValueError(f"{demo_group.name} has no camera_names attribute.")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return [str(name) for name in json.loads(value)]


def _load_sea_raft(
    sea_raft_root: str,
    config_path: str,
    checkpoint: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """Load the official SEA-RAFT implementation from the pinned submodule."""
    root = Path(sea_raft_root).resolve()
    core = root / "core"
    if not (core / "raft.py").is_file():
        raise FileNotFoundError(
            f"SEA-RAFT core was not found at {core}. Clone https://github.com/princeton-vl/SEA-RAFT."
        )
    config_file = Path(config_path).resolve()
    checkpoint_file = Path(checkpoint).resolve()
    if not config_file.is_file() or not checkpoint_file.is_file():
        raise FileNotFoundError("Both --config and --checkpoint must point to existing files.")
    for path in (str(root), str(core)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from raft import RAFT  # type: ignore

    config_values = json.loads(config_file.read_text(encoding="utf-8"))
    model_args = argparse.Namespace(**config_values)
    model = RAFT(model_args)
    checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
    if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]
    checkpoint_state = {
        str(key).removeprefix("module."): value for key, value in checkpoint_state.items()
    }
    model.load_state_dict(checkpoint_state, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, checkpoint_file.name


def _make_predictor(
    model: torch.nn.Module,
    device: torch.device,
    iterations: int | None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    @torch.inference_mode()
    def predict(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        image1 = torch.from_numpy(np.ascontiguousarray(first)).permute(0, 3, 1, 2)
        image2 = torch.from_numpy(np.ascontiguousarray(second)).permute(0, 3, 1, 2)
        image1 = image1.to(device=device, dtype=torch.float32)
        image2 = image2.to(device=device, dtype=torch.float32)
        inference_kwargs = {"test_mode": True}
        if iterations is not None:
            inference_kwargs["iters"] = int(iterations)
        output = model(image1, image2, **inference_kwargs)
        flow = output["final"] if "final" in output else output["flow"][-1]
        return flow.permute(0, 2, 3, 1).float().cpu().numpy()

    return predict


def _dataset_is_complete(dataset: h5py.Dataset, expected_shape: tuple[int, ...]) -> bool:
    return tuple(dataset.shape) == expected_shape and dataset.dtype == np.dtype(np.float16)


def _metadata_matches(
    flow_group: h5py.Group,
    checkpoint_identifier: str,
    resolution: tuple[int, int],
    gaps: tuple[int, ...],
) -> bool:
    def text_attr(name: str) -> str:
        value = flow_group.attrs.get(name, "")
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    stored_gaps = tuple(
        int(value)
        for value in np.asarray(
            flow_group.attrs.get("available_temporal_gaps", ()),
            dtype=np.int64,
        ).tolist()
    )
    try:
        stored_resolution = tuple(int(value) for value in json.loads(text_attr("flow_resolution")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        text_attr("model_name") == "SEA-RAFT"
        and text_attr("checkpoint_identifier") == str(checkpoint_identifier)
        and stored_resolution == resolution
        and text_attr("flow_direction") == "forward"
        and text_attr("flow_units") == "pixel_displacement"
        and stored_gaps == gaps
        and text_attr("storage_dtype") == "float16"
        and text_attr("preprocessing_version") == PREPROCESSING_VERSION
    )


def _flow_datasets_complete(
    flow_group: h5py.Group,
    obs_group: h5py.Group,
    cameras: Sequence[str],
    gaps: Sequence[int],
) -> bool:
    for camera in cameras:
        rgb_name = f"{camera}_rgb"
        if rgb_name not in obs_group or camera not in flow_group:
            return False
        rgb = obs_group[rgb_name]
        frames, height, width = map(int, rgb.shape[:3])
        for gap in gaps:
            name = f"gap_{int(gap)}"
            expected_shape = (max(0, frames - int(gap)), height, width, 2)
            if name not in flow_group[camera]:
                return False
            if not _dataset_is_complete(flow_group[camera][name], expected_shape):
                return False
    return True


def _write_gap(
    camera_group: h5py.Group,
    rgb: h5py.Dataset,
    gap: int,
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    batch_size: int,
    chunk_frames: int,
    compression: str | None,
    overwrite: bool,
) -> None:
    frames, height, width = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
    output_frames = max(0, frames - int(gap))
    expected_shape = (output_frames, height, width, 2)
    final_name = f"gap_{gap}"
    temporary_name = f".{final_name}_incomplete"
    if final_name in camera_group:
        if not overwrite and _dataset_is_complete(camera_group[final_name], expected_shape):
            return
        del camera_group[final_name]
    if temporary_name in camera_group:
        del camera_group[temporary_name]

    stored_chunk_frames = max(1, min(int(chunk_frames), max(1, output_frames)))
    chunks = (stored_chunk_frames, height, width, 2) if output_frames > 0 else None
    output = camera_group.create_dataset(
        temporary_name,
        shape=expected_shape,
        dtype=np.float16,
        chunks=chunks,
        compression=compression if output_frames > 0 else None,
    )
    for start in range(0, output_frames, int(batch_size)):
        stop = min(output_frames, start + int(batch_size))
        first = np.asarray(rgb[start:stop], dtype=np.uint8)
        second = np.asarray(rgb[start + gap : stop + gap], dtype=np.uint8)
        flow = predictor(first, second)
        if flow.shape != (stop - start, height, width, 2):
            raise RuntimeError(
                f"SEA-RAFT returned {flow.shape}; expected {(stop - start, height, width, 2)}."
            )
        if not np.isfinite(flow).all():
            raise RuntimeError("SEA-RAFT returned non-finite optical flow.")
        if np.abs(flow).max(initial=0.0) > np.finfo(np.float16).max:
            raise RuntimeError("SEA-RAFT flow exceeds the float16 storage range.")
        output[start:stop] = flow.astype(np.float16)
    camera_group.move(temporary_name, final_name)


def preprocess_hdf5(
    hdf5_path: str,
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    checkpoint_identifier: str,
    gaps: Sequence[int] = DEFAULT_GAPS,
    batch_size: int = 8,
    chunk_frames: int = 16,
    compression: str | None = "lzf",
    overwrite: bool = False,
    flow_group_name: str = "optical_flow",
) -> None:
    requested_gaps = tuple(sorted({int(gap) for gap in gaps}))
    if requested_gaps != DEFAULT_GAPS:
        raise ValueError(f"Training requires temporal gaps {DEFAULT_GAPS}, got {requested_gaps}.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if int(chunk_frames) <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}.")
    with h5py.File(hdf5_path, "r+") as h5_file:
        if "data" not in h5_file:
            raise ValueError(f"{hdf5_path} has no /data group.")
        for demo_key in sorted(h5_file["data"].keys()):
            demo_group = h5_file["data"][demo_key]
            if flow_group_name in demo_group and overwrite:
                del demo_group[flow_group_name]
            cameras = _camera_names(demo_group)
            obs_group = demo_group["obs"]
            reference_rgb = obs_group[f"{cameras[0]}_rgb"]
            resolution = (int(reference_rgb.shape[1]), int(reference_rgb.shape[2]))
            flow_group = demo_group.require_group(flow_group_name)
            has_existing_data = bool(flow_group.keys())
            has_metadata = "preprocessing_version" in flow_group.attrs
            if has_existing_data and not has_metadata:
                raise RuntimeError(
                    f"{flow_group.name} has flow data without provenance metadata. "
                    "Use --overwrite to rebuild it safely."
                )
            if has_metadata and not _metadata_matches(
                flow_group,
                checkpoint_identifier=checkpoint_identifier,
                resolution=resolution,
                gaps=requested_gaps,
            ):
                raise RuntimeError(
                    f"{flow_group.name} was produced with different preprocessing settings. "
                    "Use --overwrite to rebuild it."
                )
            if (
                has_metadata
                and bool(flow_group.attrs.get("preprocessing_complete", False))
                and _flow_datasets_complete(flow_group, obs_group, cameras, requested_gaps)
            ):
                print(f"[SEA-RAFT] skipped completed {Path(hdf5_path).name}:{demo_key}")
                continue

            flow_group.attrs.update(
                {
                    "model_name": "SEA-RAFT",
                    "checkpoint_identifier": str(checkpoint_identifier),
                    "flow_resolution": json.dumps(list(resolution)),
                    "flow_direction": "forward",
                    "flow_units": "pixel_displacement",
                    "available_temporal_gaps": np.asarray(requested_gaps, dtype=np.int32),
                    "storage_dtype": "float16",
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "preprocessing_complete": False,
                }
            )
            h5_file.flush()
            for camera in cameras:
                rgb_name = f"{camera}_rgb"
                if rgb_name not in obs_group:
                    raise ValueError(f"Missing {demo_group.name}/obs/{rgb_name}.")
                rgb = obs_group[rgb_name]
                camera_group = flow_group.require_group(camera)
                for gap in requested_gaps:
                    _write_gap(
                        camera_group=camera_group,
                        rgb=rgb,
                        gap=gap,
                        predictor=predictor,
                        batch_size=batch_size,
                        chunk_frames=chunk_frames,
                        compression=compression,
                        overwrite=overwrite,
                    )
            flow_group.attrs["preprocessing_complete"] = True
            h5_file.flush()
            print(f"[SEA-RAFT] completed {Path(hdf5_path).name}:{demo_key}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen SEA-RAFT flow inside Meta-World HDF5 datasets."
    )
    parser.add_argument("hdf5_paths", nargs="+", help="HDF5 datasets to update in place.")
    parser.add_argument(
        "--sea-raft-root",
        default=str(DEFAULT_SEA_RAFT_ROOT),
        help="SEA-RAFT submodule checkout.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="SEA-RAFT evaluation JSON config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Frozen SEA-RAFT checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=16,
        help="Frames per compressed HDF5 chunk, independent of inference batch size.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--flow-group", default="optical_flow")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    model, checkpoint_identifier = _load_sea_raft(
        sea_raft_root=args.sea_raft_root,
        config_path=args.config,
        checkpoint=args.checkpoint,
        device=device,
    )
    predictor = _make_predictor(model, device=device, iterations=args.iterations)
    compression = None if args.compression == "none" else args.compression
    for path in args.hdf5_paths:
        preprocess_hdf5(
            hdf5_path=path,
            predictor=predictor,
            checkpoint_identifier=checkpoint_identifier,
            batch_size=args.batch_size,
            chunk_frames=args.chunk_frames,
            compression=compression,
            overwrite=args.overwrite,
            flow_group_name=args.flow_group,
        )


if __name__ == "__main__":
    main()
