"""Screen reserve Meta-World tasks under the campaign RL protocol (review item 3a; one GPU UUID for MuJoCo EGL).

Per task: random-policy and scripted-expert episodes with 125 agent steps, action repeat 2 and the v3 reward summed per
agent step (``configs/rl/base.yaml``), and object visibility in the six training cameras.

    CUDA_VISIBLE_DEVICES=$GPU5 python scripts/screen_tasks.py --tasks sweep-into coffee-push ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus, guard_mujoco

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from s4d.data.metaworld.collect import MetaworldScene  # noqa: E402
from s4d.diag.panels import grid  # noqa: E402

ROBOT = re.compile(r"^(base|pedestal.*|torso|head|screen|controller_box|right_.*|hand|.*claw|.*pad|world)$")  # as verify_*

VISIBILITY_EVERY = 5  # agent steps between visibility samples
MIN_CAMERAS = 4


def protocol() -> tuple[int, int]:
    env = yaml.safe_load((REPO / "configs/rl/base.yaml").read_text())["env"]
    return int(env["action_repeat"]), int(env["max_episode_steps"])


def object_bodies(scene: MetaworldScene, poses: np.ndarray) -> list[int]:
    """The non-robot rendered body that moves most during the episode, plus its descendants."""
    rendered = set(int(i) for i in scene.model.geom_bodyid)
    candidates = [
        i for i, name in enumerate(scene.body_names)
        if i in rendered and scene.model.body_mocapid[i] < 0 and not ROBOT.match(name)
    ]
    movement = np.linalg.norm(poses[..., :3] - poses[:1, ..., :3], axis=-1).max(0)
    movement += 0.05 * np.linalg.norm(poses[..., 3:] - poses[:1, ..., 3:], axis=-1).max(0)
    body = max(candidates, key=lambda b: movement[b])
    chosen = {body}
    for b in range(scene.model.nbody):
        parent = b
        while parent > 0:
            if parent in chosen:
                chosen.add(b)
                break
            parent = int(scene.model.body_parentid[parent])
    return sorted(chosen)


def run_episode(scene: MetaworldScene, seed: int, policy: str, rng, repeat: int, max_sim_steps: int, sheet=None) -> dict:
    """One episode with an explicit seed, reset as ``MetaWorldCameraEnv.reset`` does for evaluation."""
    env = scene.env.unwrapped
    if hasattr(env, "_freeze_rand_vec"):
        env._freeze_rand_vec = False
    env.seed(int(seed))
    mujoco.mj_resetData(scene.model, scene.data)
    obs, _ = scene.env.reset(seed=int(seed))
    total, success, sim_steps, agent_step = 0.0, False, 0, 0
    poses, masks = [], []
    while sim_steps < max_sim_steps:
        poses.append(np.concatenate((scene.data.xpos.copy(), scene.data.xquat.copy()), axis=1))
        if policy == "expert" and agent_step % VISIBILITY_EVERY == 0:
            masks.append(scene.render_body_ids(range(6)))
            if sheet is not None and agent_step % 25 == 0:
                for cam in scene.cams[:6]:
                    scene.rgb_r.update_scene(scene.data, camera=cam)
                    sheet.append(scene.rgb_r.render().copy())
        action = (
            np.asarray(scene.policy.get_action(obs), dtype=np.float32)
            if policy == "expert"
            else rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
        )
        for _ in range(repeat):
            obs, reward, terminated, truncated, info = scene.env.step(np.clip(action, -1.0, 1.0))
            total += float(reward)
            success = success or bool(info.get("success", 0.0) > 0.5)
            sim_steps += 1
            if terminated or truncated or sim_steps >= max_sim_steps:
                break
        agent_step += 1
        if terminated or truncated:
            break
    out = {"return": total, "success": success, "agent_steps": agent_step}
    if masks:
        poses = np.stack(poses)
        bodies = object_bodies(scene, poses)
        per_cam = np.isin(np.stack(masks), bodies).reshape(len(masks), 6, -1).any(-1)
        out["visible_cameras"] = per_cam.sum(1).tolist()
        out["object_bodies"] = [scene.body_names[b] for b in bodies]
        out["object_displacement_m"] = float(np.linalg.norm(poses[:, bodies[0], :3] - poses[0, bodies[0], :3], axis=-1).max())
    return out


def screen_task(task: str, episodes: int, out_dir: Path) -> dict:
    repeat, max_sim_steps = protocol()
    scene = MetaworldScene(task, 0, 128, 128)
    sheet: list = []
    try:
        rng = np.random.default_rng(0)
        rand = [run_episode(scene, 5000 + i, "random", rng, repeat, max_sim_steps) for i in range(episodes)]
        expert = [
            run_episode(scene, 6000 + i, "expert", rng, repeat, max_sim_steps, sheet if i == 0 else None)
            for i in range(episodes)
        ]
    finally:
        scene.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid(sheet, ncol=6)).save(out_dir / f"{task}.png")
    rr = np.asarray([e["return"] for e in rand])
    visible_steps = np.concatenate([np.asarray(e["visible_cameras"]) for e in expert])
    result = {
        "task": task,
        "protocol": {"action_repeat": repeat, "max_sim_steps": max_sim_steps, "episodes": episodes},
        "random_return": {"mean": float(rr.mean()), "p10": float(np.percentile(rr, 10)), "p50": float(np.median(rr)),
                          "p90": float(np.percentile(rr, 90)), "fraction_positive": float((rr > 0).mean())},
        "random_success": float(np.mean([e["success"] for e in rand])),
        "expert_return_mean": float(np.mean([e["return"] for e in expert])),
        "expert_success": float(np.mean([e["success"] for e in expert])),
        "object_bodies": expert[0]["object_bodies"],
        "object_visible_4_of_6_fraction": float((visible_steps >= MIN_CAMERAS).mean()),
        "visible_cameras_median": float(np.median(visible_steps)),
        "expert_object_displacement_m": float(np.mean([e["object_displacement_m"] for e in expert])),
        "contact_sheet": str(out_dir / f"{task}.png"),
    }
    result["passed"] = bool(
        result["expert_success"] >= 0.8 and result["object_visible_4_of_6_fraction"] > 0.5 and result["random_return"]["mean"] > 0
    )
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default=str(REPO / "docs/task_screening"))
    args = ap.parse_args()
    out = Path(args.out)
    summary = out / "summary.json"
    results = json.loads(summary.read_text()) if summary.exists() else {}
    for task in args.tasks:
        try:
            results[task] = screen_task(task, args.episodes, out)
        except Exception as exc:  # recorded, never hidden: the task fails the screen
            results[task] = {"task": task, "passed": False, "error": repr(exc)}
            print(json.dumps(results[task]), flush=True)
        results[task].update(gpu_mapping=GPU_MAPPING, egl_device=EGL_DEVICE)
        out.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(results, indent=2))
    passed = [r for r in results.values() if r.get("passed")]
    ranking = sorted(passed, key=lambda r: (r["random_success"], -r["expert_object_displacement_m"]))
    results_ranking = [r["task"] for r in ranking]
    (out / "ranking.json").write_text(json.dumps({"rule": "random success ascending, then 3D object displacement "
                                                  "descending; among tasks passing 3a", "ranking": results_ranking}, indent=2))
    print("ranking", results_ranking, flush=True)


if __name__ == "__main__":
    main()
