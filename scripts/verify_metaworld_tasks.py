"""Twenty noise-free policies per task, with sampled object visibility and contact sheets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _bootstrap import guard_gpus, guard_mujoco

GPU_MAPPING = guard_gpus()
EGL_DEVICE = guard_mujoco()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from s4d.data.metaworld.collect import MetaworldScene  # noqa: E402
from s4d.diag.panels import grid  # noqa: E402

ROBOT = re.compile(r"^(base|pedestal.*|torso|head|screen|controller_box|right_.*|hand|.*claw|.*pad|world)$")


def verify_task(task: str, out_dir: Path, episodes: int, max_steps: int) -> dict:
    scene = MetaworldScene(task, 123, 128, 128)
    successes, lengths, visibility, chosen = [], [], [], []
    contact = []
    try:
        base = scene.body_names.index("base")
        if not (np.allclose(scene.data.xpos[base], 0) and np.allclose(scene.data.xquat[base], [1, 0, 0, 0])):
            raise ValueError("simulation world is not the robot base frame")
        rendered_bodies = set(int(i) for i in scene.model.geom_bodyid)
        candidates = [
            i
            for i, name in enumerate(scene.body_names)
            if i in rendered_bodies and scene.model.body_mocapid[i] < 0 and not ROBOT.match(name)
        ]
        for episode in range(episodes):
            obs, _ = scene.env.reset(seed=1000 + episode)
            poses, masks, rgb_samples = [], [], []
            success = False
            for t in range(max_steps):
                poses.append(np.concatenate((scene.data.xpos.copy(), scene.data.xquat.copy()), axis=1))
                if t % 10 == 0:
                    masks.append(scene.render_body_ids(range(6)))
                    if episode == 0:
                        rgbs = []
                        for cam in scene.cams:
                            scene.rgb_r.update_scene(scene.data, camera=cam)
                            rgbs.append(scene.rgb_r.render().copy())
                        rgb_samples.append(np.stack(rgbs))
                action = scene.policy.get_action(obs)
                obs, _, terminated, truncated, info = scene.env.step(action)
                success = bool(info.get("success", 0) > 0.5)
                if success or terminated or truncated:
                    break
            poses = np.stack(poses)
            movement = np.linalg.norm(poses[..., :3] - poses[:1, ..., :3], axis=-1).max(0)
            movement += 0.05 * np.linalg.norm(poses[..., 3:] - poses[:1, ..., 3:], axis=-1).max(0)
            body = max(candidates, key=lambda bid: movement[bid])
            descendants = {body}
            for bid in range(scene.model.nbody):
                parent = bid
                while parent > 0:
                    if parent in descendants:
                        descendants.add(bid)
                        break
                    parent = int(scene.model.body_parentid[parent])
            m = np.stack(masks)
            per_cam = np.isin(m, list(descendants)).reshape(len(m), 6, -1).any(-1)
            visibility.append(float((per_cam.sum(1) >= 4).mean()))
            chosen.append(scene.body_names[body])
            successes.append(success)
            lengths.append(t + 1)
            if episode == 0:
                for i in np.linspace(0, len(rgb_samples) - 1, 4).round().astype(int):
                    contact.extend(rgb_samples[i])
    finally:
        scene.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / f"{task}.png"
    Image.fromarray(grid(contact, ncol=10)).save(sheet)
    result = {
        "task": task,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "successes": successes,
        "lengths": lengths,
        "object_bodies": chosen,
        "object_visible_in_4_of_6_cams_fraction": float(np.mean(visibility)),
        "visibility_by_episode": visibility,
        "visibility_sample_stride": 10,
        "robot_base_equals_simulation_world": True,
        "contact_sheet": str(sheet),
        "passed": bool(np.mean(successes) >= 0.8 and np.mean(visibility) > 0.5),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", default="docs/task_verification")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=450)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "summary.json"
    results = json.loads(summary.read_text()) if summary.exists() else {}
    for task in args.tasks:
        try:
            result = verify_task(task, out, args.episodes, args.max_steps)
        except Exception as exc:
            result = {"task": task, "passed": False, "error": repr(exc)}
            print(json.dumps(result), flush=True)
        result.update(gpu_mapping=GPU_MAPPING, egl_device=EGL_DEVICE)
        results[task] = result
        summary.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
