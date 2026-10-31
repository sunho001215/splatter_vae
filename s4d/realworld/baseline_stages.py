"""Optional baseline-specific stages (NOT needed by our method): SNeRL DINO teacher features and SAM 3 masks.

Which baseline needs what (from s4d/baselines/*, as of this branch):

* SNeRL (s4d/baselines/snerl): RGB of 3 training cameras at 128x128 + poses; targets RGB, 4-class semantics
  (``SEMANTIC_CLASSES = background, robot, object, fixture``; Meta-World derives them from MuJoCo body ids via
  ``class_table``) and DINO ViT-S/8 features (``ViTExtractor('dino_vits8', stride=4)``, layer 11 "key" facet,
  no bin/CLS, scalar normalisation (x-0.45)/0.225, bilinear upsampling of the 31x31x384 grid; computed ONLINE
  by ``s4d.baselines.snerl.model.dino_target``). Real-world SNeRL therefore needs per-pixel semantic masks
  (``sam3_masks`` + ``instances_to_semantic``); DINO features are only cached optionally (``dino_snerl``).
* 3D-MVP: RGB-D of all training cameras + K + c2w and Meta-World ``SCENE_BOUNDS`` -> real-world bounds in the
  base frame must be chosen (not a preprocessing model stage).
* SinCro, ReViWo, DynaMo: RGB (+ cameras for SinCro) only.

Both stages consume model outputs only through isolated workers and record prompts, model identity, scores,
instances and explicit absent detections. Neither is validated with the real models here.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld import io as rio
from s4d.realworld.geometry import resize_affine, warp_affine
from s4d.realworld.workers import WorkerSpec, call_worker

SNERL_DINO = {
    "model": "dino_vits8 (facebookresearch/dino hubconf, official dino_deitsmall8_pretrain.pth)",
    "extractor": "ViTExtractor('dino_vits8', stride=4)",
    "layer": 11, "facet": "key", "bin": False, "include_cls": False,
    "normalisation": "scalar (x - 0.45) / 0.225 on RGB in [0,1] (NOT ImageNet per-channel)",
    "input_resolution": (128, 128), "token_grid": (31, 31), "dim": 384,
    "upsampling": "F.interpolate(..., size=image, mode='bilinear') (align_corners default)",
    "consumer": "s4d.baselines.snerl.model.dino_target",
}
SNERL_CLASSES = ("background", "robot", "object", "fixture")


def dino_snerl(cache_dir: Path, out: Path, spec: WorkerSpec, *, backbone: str = "scratch",
               resolution: tuple = SNERL_DINO["input_resolution"], allow_infrared: bool = False,
               allow_mock: bool = False, force: bool = False) -> dict:
    """Token grids for every exported window image, resized to the consumer resolution exactly as SNeRL images are
    (bilinear resize of the uint8 frame, then /255). Stored per window: tokens (T,V,31,31,384) float16 + record."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if any("infrared" in v for v in manifest["image_modality"].values()) and not allow_infrared:
        raise RuntimeError("DINO on infrared images is a domain shift for an RGB teacher; pass allow_infrared")

    def build(tmp: Path) -> dict:
        identity = None
        with h5py.File(cache_dir / f"{backbone}.h5", "r") as src, h5py.File(tmp / "dino_tokens.h5", "w") as dst:
            for name in sorted(src["windows"]):
                images = src["windows"][name]["images"][:]  # (T,V,3,H,W) uint8
                T, V, _, H, W = images.shape
                aff = resize_affine((H, W), resolution)
                flat = images.transpose(0, 1, 3, 4, 2).reshape(T * V, H, W, 3).astype(np.float32)
                small = np.stack([warp_affine(im, aff, resolution, order=1) for im in flat]) / 255.0
                resp, outp = call_worker(spec, "dino_snerl", {"output": "tokens"}, {"images": small.astype(np.float32)},
                                         tmp / "_work", allow_mock=allow_mock)
                tokens = outp["tokens"].reshape(T, V, *outp["tokens"].shape[1:])
                dst.create_dataset(f"windows/{name}/tokens", data=tokens.astype(np.float16), compression="lzf")
                identity = {**resp["model"], "meta": resp["meta"]}
        rio.atomic_write_json(tmp / "dino_spec.json", {**SNERL_DINO, "input_resolution": list(resolution),
                                                       "storage": "float16 token grid; upsample with the consumer's "
                                                                  "bilinear F.interpolate", "model_identity": identity})
        return {"model": identity, "windows": len(manifest["windows"])}

    return rio.run_stage(Path(out), stage="dino_snerl", version="1",
                         config={"backbone": backbone, "resolution": list(resolution), "spec": SNERL_DINO,
                                 "worker": spec.name, "allow_mock": allow_mock},
                         inputs={"export_manifest": rio.sha256_file(cache_dir / "manifest.json")}, build=build,
                         force=force)


def instances_to_semantic(masks: np.ndarray, scores: np.ndarray, frame: np.ndarray, prompt_index: np.ndarray,
                          prompt_classes: list[str], n_frames: int, hw: tuple, *, class_names=SNERL_CLASSES,
                          min_score: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel class ids (n_frames,H,W) uint8 (0 = background) and per-frame per-class presence (n_frames, C).

    Overlaps resolve to the highest-scoring instance; pixels without any instance stay background. Classes with
    no detection in a frame are reported absent in ``presence`` (never filled with a guessed mask)."""
    sem = np.zeros((n_frames, *hw), np.uint8)
    best = np.full((n_frames, *hw), -np.inf, np.float32)
    presence = np.zeros((n_frames, len(class_names)), bool)
    for m, s, f, p in zip(masks, scores, frame, prompt_index):
        if s < min_score:
            continue
        cls = class_names.index(prompt_classes[p])
        take = m & (s > best[f])
        sem[f][take] = cls
        best[f][take] = s
        presence[f, cls] = True
    presence[:, 0] = True
    return sem, presence


def sam3_masks(root: Path, episode_id: str, spec: WorkerSpec, *, cameras: list[str], prompts: dict,
               mode: str = "segment_images", chunk: int = 16, score_threshold: float = 0.5,
               allow_mock: bool = False, force: bool = False) -> dict:
    """``prompts`` maps text prompt -> SNeRL class (e.g. {"robot arm": "robot", "gripper": "robot",
    "red mug": "object", "table": "fixture"}). Stores raw instances + the derived semantic map per camera."""
    from s4d.realworld.pipeline import load_meta, read_frames

    meta = load_meta(root, episode_id)
    texts, classes = list(prompts), [prompts[t] for t in prompts]
    if any(c not in SNERL_CLASSES for c in classes):
        raise ValueError(f"prompt classes must be in {SNERL_CLASSES}")

    def build(tmp: Path) -> dict:
        summary = {}
        for cam in cameras:
            s = meta.stream(cam)
            n = len(meta.stream_timestamps_ns[cam])
            hw = (s.intrinsics.height, s.intrinsics.width)
            absent = []
            with h5py.File(tmp / f"{rio.safe_name(cam)}.h5", "w") as f:
                f.attrs.update(camera_id=cam, prompts=json.dumps(texts), prompt_classes=json.dumps(classes),
                               modality=s.modality, mode=mode, score_threshold=score_threshold)
                sem_ds = f.create_dataset("semantic", (n, *hw), np.uint8, chunks=(1, *hw), compression="lzf")
                pres_ds = f.create_dataset("class_present", (n, len(SNERL_CLASSES)), bool)
                for start in range(0, n, chunk):
                    idx = list(range(start, min(n, start + chunk)))
                    frames = read_frames(root, episode_id, cam, idx)
                    if frames.ndim == 3:
                        frames = np.repeat(frames[..., None], 3, -1)  # infrared: model input only, recorded
                    resp, o = call_worker(spec, mode, {"prompts": texts, "score_threshold": score_threshold},
                                          {"images": frames.astype(np.uint8)}, tmp / "_work", allow_mock=allow_mock)
                    g = f.create_group(f"chunks/{start:06d}")
                    for k in ("masks", "scores", "boxes_xyxy", "frame", "prompt_index", "instance_id", "count"):
                        g.create_dataset(k, data=o[k], compression="lzf" if o[k].ndim >= 2 else None)
                    g.attrs["model"] = json.dumps(resp["model"])
                    sem, pres = instances_to_semantic(o["masks"], o["scores"], o["frame"], o["prompt_index"], classes,
                                                      len(idx), hw, min_score=score_threshold)
                    sem_ds[start:start + len(idx)] = sem
                    pres_ds[start:start + len(idx)] = pres
                    for fi, row in enumerate(o["count"]):
                        for p, c in enumerate(row):
                            if c == 0:
                                absent.append([idx[fi], texts[p]])
                summary[cam] = {"frames": n, "absent_detections": len(absent), "model": resp["model"]}
            rio.atomic_write_json(tmp / f"{rio.safe_name(cam)}_absent.json", absent)
        return summary

    return rio.run_stage(rio.stage_dir(root, episode_id, "sam3_masks"), stage="sam3_masks", version="1",
                         config={"cameras": cameras, "prompts": prompts, "mode": mode, "chunk": chunk,
                                 "score_threshold": score_threshold, "worker": spec.name},
                         inputs={"ingest": rio.read_record(rio.episode_dir(root, episode_id) / "normalized")["fingerprint"]},
                         build=build, force=force)
