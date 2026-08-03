from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from visualize.visualize_splattervae_tsne import (
    build_vae,
    fixed_window_from_single_image,
    image_to_tensor,
    load_cfg,
    pooled_latent,
    preprocess_features,
    sort_demo_keys,
)


@dataclass(frozen=True)
class ImageRecord:
    demo: str
    camera: str
    timestep: int

    @property
    def key(self) -> Tuple[str, str, int]:
        return (self.demo, self.camera, int(self.timestep))

    def label(self) -> str:
        return f"{self.demo} | {self.camera} | t={int(self.timestep)}"


def parse_cameras(value: str | None, available: Sequence[str]) -> List[str]:
    if value is None or not value.strip():
        return list(available)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = [cam for cam in requested if cam not in available]
    if missing:
        raise ValueError(f"Requested cameras {missing} are not in dataset cameras {list(available)}.")
    return requested


def parse_demo_keys(value: str | None, available: Sequence[str]) -> List[str] | None:
    if value is None or not value.strip():
        return None
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = [demo for demo in requested if demo not in available]
    if missing:
        raise ValueError(f"Requested demos {missing} are not in dataset demos {list(available)}.")
    return requested


def load_camera_names(dataset_path: str, demo_key: str, cfg: dict) -> List[str]:
    with h5py.File(dataset_path, "r") as f:
        demo = f["data"][demo_key]
        cameras = json.loads(demo.attrs["camera_names"])
    configured_views = list(cfg.get("dataset", {}).get("views", cameras))
    selected = [camera for camera in configured_views if camera in cameras]
    if not selected:
        raise ValueError("No configured dataset.views are available in the selected demonstration.")
    return selected


def timestep_count(dataset_path: str, demo_key: str, camera: str) -> int:
    with h5py.File(dataset_path, "r") as f:
        return int(f["data"][demo_key]["obs"][f"{camera}_rgb"].shape[0])


def build_records(
    dataset_path: str,
    demos: Sequence[str],
    cameras: Sequence[str],
    *,
    max_steps: int,
    stride: int,
) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    stride = max(1, int(stride))
    with h5py.File(dataset_path, "r") as f:
        for demo_key in demos:
            demo = f["data"][demo_key]
            t_len = int(demo["obs"][f"{cameras[0]}_rgb"].shape[0])
            stop = min(t_len, int(max_steps)) if int(max_steps) > 0 else t_len
            for t in range(0, stop, stride):
                for cam in cameras:
                    if f"{cam}_rgb" in demo["obs"]:
                        records.append(ImageRecord(demo_key, cam, int(t)))
    return records


@torch.no_grad()
def encode_records(
    vae,
    dataset_path: str,
    records: Sequence[ImageRecord],
    *,
    batch_size: int,
    pool: str,
    device: torch.device,
) -> np.ndarray:
    if not records:
        raise ValueError("No image records were provided for encoding.")

    feats = []
    batch_size = max(1, int(batch_size))
    with h5py.File(dataset_path, "r") as f:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            images = []
            for rec in chunk:
                img = np.asarray(f["data"][rec.demo]["obs"][f"{rec.camera}_rgb"][rec.timestep], dtype=np.uint8)
                images.append(image_to_tensor(img))
            x = torch.stack(images, dim=0).to(device)
            features = vae.inference_features(fixed_window_from_single_image(x))
            feats.append(pooled_latent(features["s_inv"], pool).detach().cpu())
    return torch.cat(feats, dim=0).numpy()


def load_images(dataset_path: str, records: Sequence[ImageRecord]) -> Dict[Tuple[str, str, int], np.ndarray]:
    images: Dict[Tuple[str, str, int], np.ndarray] = {}
    with h5py.File(dataset_path, "r") as f:
        for rec in records:
            if rec.key in images:
                continue
            images[rec.key] = np.asarray(f["data"][rec.demo]["obs"][f"{rec.camera}_rgb"][rec.timestep], dtype=np.uint8)
    return images


def pairwise_distance(query: np.ndarray, candidates: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(candidates - query[None, :], axis=1)
    if metric == "cosine":
        q = query / max(float(np.linalg.norm(query)), 1e-8)
        c = candidates / np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-8)
        return 1.0 - (c @ q)
    raise ValueError(f"Unknown distance metric: {metric}")


def find_neighbors(
    query_records: Sequence[ImageRecord],
    all_records: Sequence[ImageRecord],
    features: np.ndarray,
    *,
    metric: str,
) -> List[dict]:
    index_by_key = {rec.key: idx for idx, rec in enumerate(all_records)}
    results = []
    for query in query_records:
        q_idx = index_by_key[query.key]
        candidate_indices = [idx for idx, rec in enumerate(all_records) if rec.camera != query.camera and rec.demo != query.demo]
        if not candidate_indices:
            raise ValueError(f"No different-viewpoint, different-demo candidates found for query {query}.")
        dists = pairwise_distance(features[q_idx], features[candidate_indices], metric)
        best_local = int(np.argmin(dists))
        best_idx = candidate_indices[best_local]
        neighbor = all_records[best_idx]
        results.append(
            {
                "query": query,
                "neighbor": neighbor,
                "distance": float(dists[best_local]),
            }
        )
    return results



def find_single_query_neighbors_by_camera(
    query: ImageRecord,
    target_cameras: Sequence[str],
    all_records: Sequence[ImageRecord],
    features: np.ndarray,
    *,
    metric: str,
    include_query_camera: bool = False,
) -> List[dict]:
    index_by_key = {rec.key: idx for idx, rec in enumerate(all_records)}
    if query.key not in index_by_key:
        raise ValueError(f"Query record {query} is not present in the retrieval index.")
    q_idx = index_by_key[query.key]
    results = []
    for camera in target_cameras:
        if camera == query.camera and not include_query_camera:
            continue
        candidate_indices = [idx for idx, rec in enumerate(all_records) if rec.camera == camera and rec.demo != query.demo]
        if not candidate_indices:
            raise ValueError(f"No different-demo candidate images found for target camera {camera!r}.")
        dists = pairwise_distance(features[q_idx], features[candidate_indices], metric)
        best_local = int(np.argmin(dists))
        best_idx = candidate_indices[best_local]
        neighbor = all_records[best_idx]
        results.append({"query": query, "neighbor": neighbor, "target_camera": camera, "distance": float(dists[best_local])})
    return results

def save_figure(results: Sequence[dict], images: Dict[Tuple[str, str, int], np.ndarray], out_path: Path, title: str) -> None:
    if not results:
        raise ValueError("No retrieval results to plot.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 260})
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(7.8, max(2.2, 2.1 * n)), squeeze=False)
    axes[0, 0].set_title("Query Image", fontsize=13.5, pad=8)
    axes[0, 1].set_title("Nearest Neighbor", fontsize=13.5, pad=8)

    for row, item in enumerate(results):
        query: ImageRecord = item["query"]
        neighbor: ImageRecord = item["neighbor"]
        for col, rec in enumerate((query, neighbor)):
            ax = axes[row, col]
            ax.imshow(images[rec.key])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#d1d5db")
        target_camera = item.get("target_camera", neighbor.camera)
        axes[row, 0].set_ylabel(target_camera, rotation=0, labelpad=24, fontsize=11, fontweight="bold", va="center")
        axes[row, 0].text(0.5, -0.08, f"Query: {query.label()}", transform=axes[row, 0].transAxes, ha="center", va="top", fontsize=9.5)
        axes[row, 1].text(
            0.5,
            -0.08,
            f"Nearest: {neighbor.label()} | d={float(item['distance']):.4f}",
            transform=axes[row, 1].transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
        )

    fig.suptitle(title, fontsize=15.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92, bottom=0.045, left=0.075, right=0.99, wspace=0.07, hspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize nearest-neighbor retrieval in SplatterVAE/SPLAT view-invariant embedding space.")
    parser.add_argument("--config", required=True, help="SplatterVAE training YAML config.")
    parser.add_argument("--dataset", required=True, help="Multi-view HDF5 dataset.")
    parser.add_argument("--ckpt", required=True, help="SplatterVAE checkpoint.")
    parser.add_argument("--out", default="outputs/splattervae_nearest_neighbors/retrieval.png")
    parser.add_argument("--query_demo", default=None, help="Query demo key. Defaults to the first demo.")
    parser.add_argument("--query_cam", default="cam0", help="Single query camera for per-viewpoint retrieval.")
    parser.add_argument("--query_timestep", type=int, default=None, help="Query timestep. Defaults to the middle of the encoded range.")
    parser.add_argument("--legacy_per_camera_queries", action="store_true", help="Use the old mode: one query per camera.")
    parser.add_argument("--exclude_query_camera", action="store_true", help="In single-query mode, skip retrieving from the query-camera viewpoint.")
    parser.add_argument("--include_query_camera", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate_demos", type=int, default=3, help="Number of demos to include in the retrieval index.")
    parser.add_argument("--retrieval_demos", default=None, help="Comma-separated demo keys used as retrieval candidates. Defaults to the first candidate demos.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum timesteps per candidate demo; <=0 uses the whole demo.")
    parser.add_argument("--stride", type=int, default=2, help="Candidate timestep stride.")
    parser.add_argument("--cameras", default=None, help="Comma-separated camera list. Defaults to all configured dataset cameras.")
    parser.add_argument("--pool", choices=["flatten", "mean"], default="flatten")
    parser.add_argument("--normalize", choices=["l2", "none"], default="l2")
    parser.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device)
    with h5py.File(args.dataset, "r") as f:
        demo_keys = sort_demo_keys(list(f["data"].keys()))
    if not demo_keys:
        raise ValueError(f"No demos found in {args.dataset}.")
    query_demo = args.query_demo or demo_keys[0]
    if query_demo not in demo_keys:
        raise ValueError(f"Query demo {query_demo!r} is not in the dataset.")

    available_cameras = load_camera_names(args.dataset, query_demo, cfg)
    cameras = parse_cameras(args.cameras, available_cameras)
    if args.query_cam not in cameras:
        raise ValueError(f"Query camera {args.query_cam!r} is not in the selected cameras {cameras}.")
    candidate_demos = parse_demo_keys(args.retrieval_demos, demo_keys)
    if candidate_demos is None:
        candidate_demos = demo_keys[: max(1, int(args.candidate_demos))]
        if not any(demo != query_demo for demo in candidate_demos):
            for demo in demo_keys:
                if demo != query_demo:
                    candidate_demos = list(candidate_demos) + [demo]
                    break
    if not any(demo != query_demo for demo in candidate_demos):
        raise ValueError(
            f"Retrieval candidates must include at least one demo different from query demo {query_demo!r}. "
            "Use --candidate_demos >= 2 or pass --retrieval_demos."
        )

    t_len = timestep_count(args.dataset, query_demo, cameras[0])
    max_steps = t_len if int(args.max_steps) <= 0 else min(t_len, int(args.max_steps))
    query_timestep = int(args.query_timestep) if args.query_timestep is not None else max_steps // 2
    query_timestep = max(0, min(query_timestep, t_len - 1))
    if args.legacy_per_camera_queries:
        query_records = [ImageRecord(query_demo, cam, query_timestep) for cam in cameras]
    else:
        query_records = [ImageRecord(query_demo, args.query_cam, query_timestep)]
    candidate_records = build_records(args.dataset, candidate_demos, cameras, max_steps=max_steps, stride=int(args.stride))

    unique: Dict[Tuple[str, str, int], ImageRecord] = {}
    for rec in list(query_records) + candidate_records:
        unique[rec.key] = rec
    all_records = list(unique.values())

    vae, _ = build_vae(cfg, args.dataset, query_demo, args.ckpt, device)
    features = encode_records(
        vae,
        args.dataset,
        all_records,
        batch_size=int(args.batch_size),
        pool=args.pool,
        device=device,
    )
    features = preprocess_features(features, args.normalize)
    if args.legacy_per_camera_queries:
        results = find_neighbors(query_records, all_records, features, metric=args.distance)
    else:
        results = find_single_query_neighbors_by_camera(
            query_records[0],
            cameras,
            all_records,
            features,
            metric=args.distance,
            include_query_camera=not bool(args.exclude_query_camera),
        )
    image_records = []
    for item in results:
        image_records.extend([item["query"], item["neighbor"]])
    images = load_images(args.dataset, image_records)

    out_path = Path(args.out)
    save_figure(results, images, out_path, "SPLAT / SplatterVAE View-Invariant Nearest Neighbors")
    serializable = {
        "config": args.config,
        "dataset": args.dataset,
        "checkpoint": args.ckpt,
        "query_demo": query_demo,
        "query_camera": args.query_cam,
        "query_timestep": query_timestep,
        "retrieval_mode": "legacy_per_camera_queries" if args.legacy_per_camera_queries else "single_query_per_target_camera",
        "different_demo_required": True,
        "candidate_demos": candidate_demos,
        "cameras": cameras,
        "pool": args.pool,
        "normalize": args.normalize,
        "distance": args.distance,
        "results": [
            {
                "query": {"demo": item["query"].demo, "camera": item["query"].camera, "timestep": item["query"].timestep},
                "neighbor": {"demo": item["neighbor"].demo, "camera": item["neighbor"].camera, "timestep": item["neighbor"].timestep},
                "target_camera": item.get("target_camera", item["neighbor"].camera),
                "distance": item["distance"],
            }
            for item in results
        ],
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved: {out_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
