"""SNeRL sanity panels (docs/figures/baselines/snerl/).

``classes``: the semantic class maps of every task (body id -> background/robot/object/fixture), from the data only.
``render``: a trained run's renders against ground truth: RGB, depth, semantic classes and DINO-feature PCA, for an
input training camera, a non-input training camera and a held-out camera, from the latent of three input views.

    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/render_snerl.py classes --out docs/figures/baselines/snerl
    CUDA_VISIBLE_DEVICES=<uuid> python -I scripts/render_snerl.py render --run runs/pretrain/<name> \
        --out docs/figures/baselines/snerl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import REPO, guard_gpus, guard_mujoco  # noqa: E402

GPU_MAPPING = guard_gpus()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from s4d.baselines.data import split_episodes  # noqa: E402
from s4d.baselines.snerl import viz  # noqa: E402
from s4d.baselines.snerl.data import SEMANTIC_CLASSES, SNeRLScenes, class_lookup, class_table  # noqa: E402
from s4d.baselines.snerl.model import create, dino_target, encode_views, render_camera  # noqa: E402
from s4d.baselines.snerl.upstream import dino_extractor  # noqa: E402
from s4d.config import load_config  # noqa: E402
from s4d.data import METAWORLD_ROOT  # noqa: E402
from ops.resources.gpu_guard import validate_runtime_path  # noqa: E402

TASKS = ("hammer", "pick-place", "bin-picking", "coffee-push", "door-open", "peg-insert-side", "peg-unplug-side",
         "stick-push")


def class_panel(out: Path, cameras=(0, 3)) -> None:
    """One validation frame per task: RGB and class map for two training cameras; the body tables as JSON."""
    rows, tables = [], {}
    for task in TASKS:
        path = METAWORLD_ROOT / f"{task}.hdf5"
        episode = split_episodes(METAWORLD_ROOT / "splits" / f"{task}_seed0.json", "validation")[0]
        with h5py.File(path, "r") as f:
            names = json.loads(f.attrs["body_names"])
            table = class_table(task, names)
            lut = class_lookup(table, int(f.attrs["background_body"]))
            ep = f["episodes"][episode]
            t = int(ep.attrs["length"]) // 2
            row = []
            for v in cameras:
                classes = lut[ep["body_id"][t, v]]
                row += [viz.tile(ep["rgb"][t, v], f"{task} cam{v}"), viz.tile(viz.class_image(classes), "classes")]
            rows.append(row)
        tables[task] = {SEMANTIC_CLASSES[c]: [n for n, k in zip(names, table) if k == c] for c in range(4)}
    Image.fromarray(viz.grid(rows)).save(out / "semantic_classes.png", optimize=True)
    (out / "semantic_classes.json").write_text(json.dumps(tables, indent=1))


def replay_classes(scenes: SNeRLScenes, episode: str, t: int, cameras: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    """Real semantic GT at held-out cameras, missing from HDF5: replay one stored simulator state, never train on it.

    Native RGB/depth must reproduce the stored frames under the existing held-out replay checks; training-camera
    body IDs must match exactly. No simulator state, labels or camera information enters the RL encoder beyond c2w.
    """
    guard_mujoco()  # exactly one allowed UUID, before importing any MuJoCo/Meta-World code
    from s4d.data.metaworld.collect import MetaworldScene
    from s4d.data.metaworld.heldout_render import (
        MAX_DEPTH_MISMATCH_FRACTION,
        MAX_RGB_MISMATCH_FRACTION,
        _set_goal,
        _set_state,
    )

    scene = MetaworldScene(scenes.task, 0, 128, 128)
    try:
        with h5py.File(scenes.path, "r") as f:
            if json.loads(f.attrs["body_names"]) != scene.body_names:
                raise ValueError("native model body names differ from the stored semantic table")
            np.testing.assert_allclose(scene.rig["c2w"], scenes.c2w_cv, atol=1e-5, rtol=0)
            ep = f["episodes"][episode]
            _set_goal(scene, ep)
            _set_state(scene, ep, t)
            rgb, depth = scene.render(cameras)
            body = scene.render_body_ids(cameras)
            checks = {}
            for k, v in enumerate(cameras):
                drgb = np.abs(rgb[k].astype(np.int32) - ep["rgb"][t, v].astype(np.int32)).max(-1)
                ddepth = np.abs(depth[k].astype(np.int64) - ep["depth"][t, v].astype(np.int64))
                record = {"rgb_fraction_over_8": float((drgb > 8).mean()),
                          "depth_fraction_over_1mm": float((ddepth * float(f.attrs["depth_unit_m"]) > 1e-3).mean())}
                if v < scenes.num_cams:
                    np.testing.assert_array_equal(body[k], ep["body_id"][t, v])
                    record["body_ids_exact"] = True
                if (record["rgb_fraction_over_8"] > MAX_RGB_MISMATCH_FRACTION
                        or record["depth_fraction_over_1mm"] > MAX_DEPTH_MISMATCH_FRACTION):
                    raise RuntimeError(f"native replay does not match the stored frame at cam{v}: {record}")
                checks[f"cam{v}"] = record
        return scenes.lut[body], {"source": "native stored-state replay", "checks": checks}
    finally:
        scene.close()


@torch.no_grad()
def render_panel(run: Path, out: Path, inputs=(0, 2, 4), targets=(0, 1, 6), chunk: int = 4096) -> dict:
    cfg = load_config([run / "config.yaml"])
    device = torch.device("cuda")
    model = create(cfg["nerf"], device)
    ckpt = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.train(False)
    extractor = dino_extractor(device, (cfg.get("dino") or {}).get("weights"))
    interface = cfg["data_interface"]
    lookat, scale = tuple(interface["lookat"]), float(interface["pose_scale"])
    episode = split_episodes(interface["split_manifest"], "validation")[0]
    scenes = SNeRLScenes(cfg["dataset"]["hdf5_path"], [episode], 0, len(inputs), None, lookat, scale)
    with h5py.File(cfg["dataset"]["hdf5_path"], "r") as f:
        ep = f["episodes"][episode]
        t = int(ep.attrs["length"]) // 2
        rgb_all = ep["rgb"][t].astype(np.float32) / 255.0  # (10, H, W, 3)
        depth_all = ep["depth"][t].astype(np.float32) * float(f.attrs["depth_unit_m"])
        classes_train = scenes.lut[ep["body_id"][t]]  # (6, H, W)
    H, W = rgb_all.shape[1:3]
    classes_replay, replay = replay_classes(scenes, episode, t, tuple(targets))
    latent = encode_views(model, torch.from_numpy(rgb_all[list(inputs)]), torch.from_numpy(scenes.c2w_cv[list(inputs)]),
                          lookat, scale)
    to8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)  # noqa: E731
    rows = [[viz.tile(to8(rgb_all[v]), f"input cam{v}") for v in inputs]]
    metrics = {"run": str(run), "step": int(ckpt["global_step"]), "episode": episode, "t": t, "inputs": list(inputs),
               "semantic_ground_truth": replay, "ray_intrinsics": "rig K unchanged; no half-pixel offset"}
    for k, v in enumerate(targets):
        rendered = render_camera(model, latent, scenes.K_cv[v], scenes.c2w_cv[v], H, W, chunk, lookat, scale)
        kind = ("input" if v in inputs else "non-input training") if v < scenes.num_cams else "held-out"
        gt = torch.from_numpy(rgb_all[v]).to(device)
        teacher = dino_target(extractor, gt).cpu().numpy()
        gt_pca, pred_pca = viz.dino_pca(teacher, rendered["dino"].cpu().numpy())
        pred_rgb, pred_depth = rendered["rgb"].cpu().numpy(), rendered["depth"].cpu().numpy()
        valid = depth_all[v] > 0
        sem = rendered["semantic"].argmax(-1).cpu().numpy()
        record = {
            "camera": int(v), "kind": kind,
            "psnr": float(-10 * np.log10(np.mean((pred_rgb - rgb_all[v]) ** 2))),
            "depth_mae_m": float(np.abs(pred_depth - depth_all[v])[valid].mean()),
            "depth_within_near_far": float(((depth_all[v] >= 0.5) & (depth_all[v] <= 1.5))[valid].mean()),
            "dino_mse": float(np.mean((rendered["dino"].cpu().numpy() - teacher) ** 2)),
        }
        if v < scenes.num_cams:
            np.testing.assert_array_equal(classes_replay[k], classes_train[v])
        record["semantic_acc"] = float((sem == classes_replay[k]).mean())
        gt_classes = viz.tile(viz.class_image(classes_replay[k]), "GT classes" if v < scenes.num_cams else "GT (replay)")
        metrics[f"cam{v}"] = record
        rows.append([
            viz.tile(to8(rgb_all[v]), f"GT cam{v} ({kind})"),
            viz.tile(to8(pred_rgb), f"render {record['psnr']:.1f} dB"),
            viz.tile(viz.depth_image(depth_all[v], 0.3, 1.8, ~valid), "GT depth"),
            viz.tile(viz.depth_image(pred_depth, 0.3, 1.8), "depth"),
            gt_classes,
            viz.tile(viz.class_image(sem), "classes"),
            viz.tile(gt_pca, "DINO PCA (teacher)"),
            viz.tile(pred_pca, "rendered DINO"),
        ])
    name = Path(run).name
    Image.fromarray(viz.grid(rows)).save(out / f"render_{name}.png", optimize=True)
    (out / f"render_{name}.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("classes", "render"))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "docs/figures/baselines/snerl")
    args = ap.parse_args()
    args.out = validate_runtime_path(args.out, repository=REPO)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "classes":
        class_panel(args.out)
    else:
        if args.run is None:
            ap.error("--run is required for render")
        print(json.dumps(render_panel(args.run.resolve(), args.out), indent=1))


if __name__ == "__main__":
    main()
