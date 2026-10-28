"""Method-independent camera visibility measurement for Meta-World (camera suite v2, user directive 2026-10-29).

For a camera (one pose) or a looping camera path, rendered on representative trajectories of a task (scripted-expert
episodes at the collection noise levels and uniform-random episodes, fixed seeds, the RL episode timing: one render
at reset and one per agent step of 2 simulator steps), every frame is measured from MuJoCo segmentation and depth:

* manipulated object(s) and end effector (the gripper: ``hand`` and its descendants): visible pixels in the normal
  scene, and in a *diagnostic* render in which the robot's proximal links are hidden (every robot body except the
  wrist ``right_l6`` and its descendants; the wrist covering the gripper from above is the gripper's own geometry,
  not proximal-link occlusion). The difference is the part the arm occludes. Diagnostic renders exist only here;
  policies always receive the normal, unmodified scene.
* goal region: 5 points at the task's goal location (the position the success criterion refers to; centre and
  +/-2 cm in x and y), each visible if it projects into the image and the surface seen there is not more than 2 cm
  in front of it (depth buffer, 3x3 maximum), or that surface is the manipulated object or the physical goal body
  (an object resting on the goal is the goal seen occupied). Counted with and without the arm.
* workspace: a grid on the table surface (x in [-0.35, 0.35] m, y in [0.40, 0.90] m, 5 cm spacing; z = the top of
  the table's box geoms at that point, 0 m in most tasks, 0.05 m on sweep-into's raised table) tested against the
  depth buffer: inside the image, and unoccluded (with and without the arm).
* camera safety: the minimum rendered depth (near-plane clipping) and the camera centre's clearance from every
  rendered geom's bounding box (a camera inside geometry).

Translucent geoms (effective alpha < 0.5: the glass panes of window-open) do not block the view in the RGB images,
so they are left out of both analysis renders. Geoms with alpha 0 are never rendered by MuJoCo.

Import order matters for GPU isolation: callers must run ``enforce_allowed_gpus`` and import ``mujoco_mig_setup``
before importing this module (it imports mujoco and metaworld).
"""

from __future__ import annotations

import copy
import zlib

import gymnasium as gym
import metaworld  # noqa: F401  (registers Meta-World/MT1)
import mujoco
import numpy as np
from metaworld.policies import ENV_POLICY_MAP

from s4d.data.metaworld.cameras import FOVY_DEG, LOOKAT, intrinsics_from_fovy
from s4d.data.metaworld.collect import EpisodePlan, encode_depth  # noqa: F401  (encode_depth: shared depth format)

CAMPAIGN_TASKS = (
    "hammer", "bin-picking", "door-open", "peg-insert-side", "peg-unplug-side", "pick-place",
    "stick-push", "coffee-push", "drawer-open", "window-open", "soccer", "sweep-into",
)

# Manipulated objects: the bodies verified as the moving object (docs/task_verification, runs/setup/task-screening-
# 20261028) plus their descendants; stick-push also counts the stick that pushes the thermos. "site:<name>" means the
# body that owns that site (bodies without a name in the task XML).
TASK_OBJECTS = {
    "hammer": ["hammer"],
    "bin-picking": ["objA"],
    "door-open": ["door_link"],
    "peg-insert-side": ["peg"],
    "peg-unplug-side": ["plug"],
    "pick-place": ["obj"],
    "stick-push": ["site:stick_end", "site:insertion"],
    "coffee-push": ["mug"],
    "drawer-open": ["drawer_link"],
    "window-open": ["windowb_a"],
    "soccer": ["soccer_ball1"],
    "sweep-into": ["obj"],
}

# Goal location per task: where the success criterion is measured. "target": the env's ``_target_pos``; "site": a
# site's current position; "target_x_site_yz": the target's x with a site's y and z. "bodies": the physical goal
# whose pixels at the goal location count as the goal seen (it surrounds or carries the location).
TASK_GOALS = {
    "hammer": {"site": "nailHead", "bodies": ["nail_link"],
               "why": "the nail head the hammer must drive in; the target point itself lies inside the nail box"},
    "bin-picking": {"target": True, "bodies": ["binB"], "why": "the floor of the target bin (no goal marker)"},
    "door-open": {"target": True, "why": "the handle's target position (goal marker)"},
    "peg-insert-side": {"site": "hole", "bodies": ["site:hole"],
                        "why": "the hole entrance; the target point lies 6.6 cm inside the box"},
    "peg-unplug-side": {"target": True, "why": "where the plug end must be pulled to (goal marker)"},
    "pick-place": {"target": True, "why": "where the object must be placed (goal marker)"},
    "stick-push": {"target": True, "why": "where the thermos must be pushed to (goal marker)"},
    "coffee-push": {"target": True, "why": "the spot under the machine's spout (marker not visibly rendered)"},
    "drawer-open": {"target": True, "why": "the handle's target position (goal marker)"},
    "window-open": {"target_x_site_yz": "handleOpenStart",
                    "why": "where the handle must slide to (success compares the handle's x with the target x); the "
                           "goal marker itself sits inside the window frame"},
    "soccer": {"target": True, "bodies": ["soccer_goal"], "why": "the target inside the goal net"},
    "sweep-into": {"target": True, "why": "the hole the object must be swept into (goal marker)"},
}
GOAL_POINT_OFFSETS = np.array([[0, 0, 0], [0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0, -0.02, 0]])
GOAL_DEPTH_TOLERANCE_M = 0.02
WRIST_ROOT = "right_l6"  # the wrist and gripper are not occluders; every other robot body is

WORKSPACE_X = np.round(np.arange(-0.35, 0.351, 0.05), 3)
WORKSPACE_Y = np.round(np.arange(0.40, 0.901, 0.05), 3)
DEPTH_TOLERANCE_M = 0.01  # against the 3x3 maximum depth (max_depth_3x3)
TRANSLUCENT_ALPHA = 0.5

# Representative trajectories: 10 scripted-expert episodes (2 per collection noise level, smooth noise as in
# collect.py) and 5 uniform-random episodes per task, fixed seeds, RL timing.
EXPERT_NOISE_LEVELS = (0.0, 0.05, 0.15, 0.30, 0.50)
EXPERT_PER_LEVEL = 2
RANDOM_EPISODES = 5
CONSTRUCTION_SEED = 1
EXPERT_SEED_BASE, RANDOM_SEED_BASE = 900_000, 900_100
AGENT_STEPS, ACTION_REPEAT = 125, 2

METRICS = (
    "obj_n", "obj_a", "ee_n", "ee_a", "goal_n", "goal_a",  # visible pixels (goal: of 5 points) with / without the arm
    "ws_inview", "ws_vis", "ws_vis_noarm", "goal_inview", "min_depth", "clearance",
    "min_depth_robust",  # minimum after a 3x3 median: MuJoCo's depth buffer has isolated single-pixel spikes
)
GROUP_HIDDEN = 5  # analysis-only geom/site group of the robot arm (disabled in the diagnostic render)
GROUP_TRANSLUCENT = 4  # disabled in both analysis renders (MuJoCo's default hides groups 3-5)


def _bit(name: str) -> int:
    return {"obj": 1, "ee": 2, "goal": 4}[name]


class SceneSets:
    """Body / geom / site sets of one task model."""

    def __init__(self, model: mujoco.MjModel, task: str):
        self.model = model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}" for i in range(model.nbody)]
        self.body_names = names

        def resolve(spec: str) -> int:
            if spec.startswith("site:"):
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec[5:])
                if site < 0:
                    raise ValueError(f"{task}: no site {spec[5:]!r}")
                return int(model.site_bodyid[site])
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec)
            if body < 0:
                raise ValueError(f"{task}: no body {spec!r}")
            return body

        def subtree(roots) -> set[int]:
            out = set(roots)
            for b in range(model.nbody):
                parent = b
                while parent > 0:
                    if parent in out:
                        out.add(b)
                        break
                    parent = int(model.body_parentid[parent])
            return out

        self.robot = subtree([resolve("base")])
        self.gripper = subtree([resolve("hand")])
        self.arm = self.robot - subtree([resolve(WRIST_ROOT)])  # proximal links: hidden in the diagnostic render
        self.objects = subtree([resolve(s) for s in TASK_OBJECTS[task]])
        goal = TASK_GOALS[task]
        self.goal = goal
        self.goal_bodies = subtree([resolve(s) for s in goal.get("bodies", [])])
        site = goal.get("site") or goal.get("target_x_site_yz")
        self.goal_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) if site else -1
        if site and self.goal_site < 0:
            raise ValueError(f"{task}: missing goal site {site!r}")
        if self.objects & self.robot or self.goal_bodies & self.robot:
            raise ValueError(f"{task}: object/goal bodies overlap the robot")
        geom_alpha = np.where(model.geom_matid >= 0, model.mat_rgba[np.maximum(model.geom_matid, 0), 3], 1.0)
        default_rgba = np.all(np.isclose(model.geom_rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
        alpha = np.where(default_rgba & (model.geom_matid >= 0), geom_alpha, model.geom_rgba[:, 3])
        alpha = np.minimum(alpha, np.where(model.geom_matid >= 0, geom_alpha, 1.0))
        self.translucent_geoms = [g for g in range(model.ngeom) if 0 < alpha[g] < TRANSLUCENT_ALPHA]
        # Pixel categories by geom / site id.
        self.geom_cat = np.zeros(model.ngeom, np.int32)
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            self.geom_cat[g] = (_bit("obj") * (b in self.objects) + _bit("ee") * (b in self.gripper)
                                + _bit("goal") * (b in self.goal_bodies or b in self.objects))
        self.site_cat = np.zeros(model.nsite, np.int32)
        for s in range(model.nsite):
            b = int(model.site_bodyid[s])
            self.site_cat[s] = (_bit("obj") * (b in self.objects) + _bit("ee") * (b in self.gripper)
                                + _bit("goal") * (b in self.goal_bodies or b in self.objects))
        self.arm_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in self.arm]
        self.arm_sites = [s for s in range(model.nsite) if int(model.site_bodyid[s]) in self.arm]
        visible = (model.geom_group < 3) & (model.geom_rgba[:, 3] > 0)
        self.solid_geoms = np.array([g for g in range(model.ngeom) if visible[g] and g not in self.translucent_geoms])

    def goal_location(self, env_unwrapped, data) -> np.ndarray:
        """The task's goal location (see ``TASK_GOALS``) at the current state."""
        target = np.asarray(env_unwrapped._target_pos, dtype=np.float64)
        if "site" in self.goal:
            return data.site_xpos[self.goal_site].astype(np.float64).copy()
        if "target_x_site_yz" in self.goal:
            return np.array([target[0], *data.site_xpos[self.goal_site][1:]], dtype=np.float64)
        return target.copy()

    def workspace_points(self, data) -> np.ndarray:
        """Grid points on the table surface: z is the top of the table body's box geoms above (x, y)."""
        table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tablelink")
        boxes = [g for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == table
                 and self.model.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_BOX) and self.model.geom_group[g] < 3]
        points = []
        for y in WORKSPACE_Y:
            for x in WORKSPACE_X:
                tops = []
                for g in boxes:
                    if not np.allclose(data.geom_xmat[g].reshape(3, 3), np.eye(3), atol=1e-6):
                        raise ValueError("table box geoms are expected to be axis-aligned")
                    c, h = data.geom_xpos[g], self.model.geom_size[g]
                    if abs(x - c[0]) <= h[0] and abs(y - c[1]) <= h[1]:
                        tops.append(c[2] + h[2])
                if tops:
                    points.append([x, y, max(tops)])
        return np.asarray(points, dtype=np.float64)

    def analysis_model(self) -> mujoco.MjModel:
        """A copy whose arm geoms/sites sit in ``GROUP_HIDDEN`` and translucent geoms in ``GROUP_TRANSLUCENT``."""
        if np.any(self.model.geom_group == GROUP_HIDDEN) or np.any(self.model.site_group == GROUP_HIDDEN):
            raise ValueError("the analysis group is already used by the task model")
        m = copy.deepcopy(self.model)
        m.geom_group[self.arm_geoms] = GROUP_HIDDEN
        m.site_group[self.arm_sites] = GROUP_HIDDEN
        if self.translucent_geoms:
            m.geom_group[self.translucent_geoms] = GROUP_TRANSLUCENT
        return m


def free_camera(pose) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(LOOKAT, dtype=np.float64)
    cam.distance, cam.azimuth, cam.elevation = float(pose.distance), float(pose.azimuth), float(pose.elevation)
    return cam


class SegDepthRenderer:
    """One render pass yields segmentation ids and metric depth (MuJoCo's Renderer needs two passes)."""

    def __init__(self, model: mujoco.MjModel, size: int):
        self.size = int(size)
        self.gl = mujoco.GLContext(self.size, self.size)
        self.gl.make_current()
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)
        self.ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.rect = mujoco.MjrRect(0, 0, self.size, self.size)
        self.rgb = np.empty((self.size, self.size, 3), np.uint8)
        self.depth = np.empty((self.size, self.size), np.float32)
        extent = model.stat.extent
        zfar, znear = np.float32(model.vis.map.zfar * extent), np.float32(model.vis.map.znear * extent)
        c = -(zfar + znear) / (zfar - znear)
        d = -(np.float32(2) * zfar * znear) / (zfar - znear)
        self.c_coef, self.d_coef = float(np.float32(-0.5) * c - np.float32(0.5)), float(np.float32(-0.5) * d)
        self.znear = float(znear)

    def update(self, model, data, option, camera, *, segment: bool) -> np.ndarray | None:
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = segment
        self.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = segment
        if not segment:
            return None
        n = self.scene.ngeom
        lut = np.zeros((n + 1, 2), np.int32) - 1
        for g in self.scene.geoms[:n]:
            if g.segid != -1:
                lut[g.segid + 1] = (g.objtype, g.objid)
        return lut

    def set_camera(self, model, data, camera) -> None:
        mujoco.mjv_updateCamera(model, data, camera, self.scene)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """(segment index image (H,W) int64 with 0 = background, metric depth (H,W) float32), top row first."""
        self.gl.make_current()
        mujoco.mjr_render(self.rect, self.scene, self.ctx)
        mujoco.mjr_readPixels(self.rgb, self.depth, self.rect, self.ctx)
        rgb = np.flipud(self.rgb).astype(np.int64)
        seg = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
        depth = (self.d_coef / (np.flipud(self.depth).astype(np.float64) + self.c_coef)).astype(np.float32)
        return seg, depth

    def render_rgb(self) -> np.ndarray:
        self.gl.make_current()
        mujoco.mjr_render(self.rect, self.scene, self.ctx)
        mujoco.mjr_readPixels(self.rgb, None, self.rect, self.ctx)
        return np.flipud(self.rgb).copy()

    def close(self) -> None:
        self.ctx.free()
        self.gl.free()


def opencv_c2w(pose) -> np.ndarray:
    from s4d.rl.env import camera_c2w  # noqa: PLC0415

    return camera_c2w(pose)


def project(points: np.ndarray, c2w: np.ndarray, K: np.ndarray, size: int):
    """World points (N,3) -> pixel rows/cols (int), camera depth z, and an inside-the-image mask."""
    w2c = np.linalg.inv(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    inside = (z > 1e-6) & (u >= 0) & (u < size) & (v >= 0) & (v < size)
    col = np.clip(np.floor(np.nan_to_num(u)), 0, size - 1).astype(int)
    row = np.clip(np.floor(np.nan_to_num(v)), 0, size - 1).astype(int)
    return row, col, z, inside


def max_depth_3x3(depth: np.ndarray) -> np.ndarray:
    """Per-pixel maximum depth over the 3x3 neighbourhood: a point is tested against the farthest surface within one
    pixel, so unoccluded table points pass at grazing angles (where depth changes by >1 cm per pixel)."""
    padded = np.pad(depth, 1, mode="edge")
    h, w = depth.shape
    return np.max([padded[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)


def median_3x3(depth: np.ndarray) -> np.ndarray:
    padded = np.pad(depth, 1, mode="edge")
    h, w = depth.shape
    return np.median([padded[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)


def geom_clearance(model, data, geoms: np.ndarray, position: np.ndarray) -> float:
    """Smallest signed distance (m) from ``position`` to the oriented bounding boxes of ``geoms`` (negative inside);
    planes use the distance above the plane."""
    local = np.einsum("gji,gj->gi", data.geom_xmat[geoms].reshape(-1, 3, 3), position - data.geom_xpos[geoms])
    aabb = model.geom_aabb[geoms]
    box = np.max(np.abs(local - aabb[:, :3]) - aabb[:, 3:], axis=1)
    plane = model.geom_type[geoms] == int(mujoco.mjtGeom.mjGEOM_PLANE)
    dist = np.where(plane, local[:, 2], box)
    return float(dist.min())


class VisibilityProbe:
    """Measures every view (a list of poses indexed by render count) at the current simulator state."""

    def __init__(self, task: str, model, size: int):
        self.task, self.size = task, int(size)
        self.sets = SceneSets(model, task)
        self.K = intrinsics_from_fovy(FOVY_DEG, self.size, self.size)
        self.analysis = self.sets.analysis_model()
        self.renderer = SegDepthRenderer(self.analysis, self.size)
        self.normal = mujoco.MjvOption()
        self.normal.geomgroup[GROUP_HIDDEN] = 1
        self.normal.sitegroup[GROUP_HIDDEN] = 1
        self.diagnostic = mujoco.MjvOption()  # MuJoCo default: groups 0-2 only, so the arm is hidden
        self._c2w_cache: dict = {}
        self.workspace = None

    def refresh(self, model) -> None:
        """Re-copy the task model after a reset (Meta-World moves fixed bodies and goal sites per episode)."""
        self.analysis = self.sets.analysis_model()
        self.workspace = None

    def _c2w(self, pose) -> np.ndarray:
        key = (pose.azimuth, pose.elevation, pose.distance)
        if key not in self._c2w_cache:
            self._c2w_cache[key] = opencv_c2w(pose)
        return self._c2w_cache[key]

    def _categories(self, lut: np.ndarray) -> np.ndarray:
        cat = np.zeros(len(lut), np.int32)
        geom = lut[:, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        site = lut[:, 0] == int(mujoco.mjtObj.mjOBJ_SITE)
        cat[geom] = self.sets.geom_cat[lut[geom, 1]]
        cat[site] = self.sets.site_cat[lut[site, 1]]
        return cat

    def measure(self, data, poses: list, goal_position: np.ndarray, env_model) -> np.ndarray:
        """(len(poses), len(METRICS)) float64 for the current state; ``poses[i]`` is view i's pose at this frame."""
        out = np.zeros((len(poses), len(METRICS)))
        goal_points = goal_position + GOAL_POINT_OFFSETS
        if self.workspace is None:
            self.workspace = self.sets.workspace_points(data)
        renders = {}
        for label, option in (("normal", self.normal), ("diag", self.diagnostic)):
            lut = self.renderer.update(self.analysis, data, option, free_camera(poses[0]), segment=True)
            cat = self._categories(lut)
            per_view = []
            for i, pose in enumerate(poses):
                if i:
                    self.renderer.set_camera(self.analysis, data, free_camera(pose))
                seg, depth = self.renderer.render()
                per_view.append((cat[seg], depth))
            renders[label] = per_view
        for i, pose in enumerate(poses):
            c2w = self._c2w(pose)
            (cat_n, depth_n), (cat_d, depth_d) = renders["normal"][i], renders["diag"][i]
            near_n = depth_n  # unfiltered, for the near-plane check
            depth_n, depth_d = max_depth_3x3(depth_n), max_depth_3x3(depth_d)
            row = out[i]
            for k, name in enumerate(("obj", "ee")):
                bit = _bit(name)
                row[2 * k] = np.count_nonzero(cat_n & bit)
                row[2 * k + 1] = np.count_nonzero(cat_d & bit)
            r, c, z, inside = project(self.workspace, c2w, self.K, self.size)
            row[6] = inside.mean()
            row[7] = (inside & (z <= depth_n[r, c] + DEPTH_TOLERANCE_M)).mean()
            row[8] = (inside & (z <= depth_d[r, c] + DEPTH_TOLERANCE_M)).mean()
            r, c, z, inside = project(goal_points, c2w, self.K, self.size)
            goal_bit = _bit("goal")
            for slot, depth, cat in ((4, depth_n, cat_n), (5, depth_d, cat_d)):
                seen = (z <= depth[r, c] + GOAL_DEPTH_TOLERANCE_M) | ((cat[r, c] & goal_bit) > 0)
                row[slot] = np.count_nonzero(inside & seen)
            _, _, _, inside = project(goal_position[None], c2w, self.K, self.size)
            row[9] = float(inside[0])
            valid = (near_n > 0) & (near_n < 0.5 * env_model.vis.map.zfar * env_model.stat.extent)
            row[10] = float(near_n[valid].min()) if valid.any() else np.inf
            row[11] = geom_clearance(env_model, data, self.sets.solid_geoms, c2w[:3, 3])
            robust = median_3x3(np.where(valid, near_n, np.inf))
            row[12] = float(robust[np.isfinite(robust)].min()) if np.isfinite(robust).any() else np.inf
        return out

    def close(self) -> None:
        self.renderer.close()


def representative_episodes(task: str) -> list[dict]:
    """10 scripted-expert episodes (2 per collection noise level) and 5 uniform-random episodes, fixed seeds."""
    def rng(index: int) -> np.random.Generator:  # one stream per episode: any subset replays identically
        return np.random.default_rng([zlib.crc32(task.encode()), 20261029, index])

    episodes = []
    for level in EXPERT_NOISE_LEVELS:
        for _ in range(EXPERT_PER_LEVEL):
            i = len(episodes)
            r = rng(i)
            params = {"strength": float(level), "smoothing": 0.85, "hold": int(r.integers(8, 17))}
            episodes.append({"kind": "expert", "noise": float(level), "seed": EXPERT_SEED_BASE + i,
                             "plan": EpisodePlan("expert_guided", params, r)})
    for j in range(RANDOM_EPISODES):
        episodes.append({"kind": "random", "noise": None, "seed": RANDOM_SEED_BASE + j, "rng": rng(len(episodes))})
    return episodes


class TaskEnv:
    """The Meta-World env reset exactly as ``MetaWorldCameraEnv.reset`` does for evaluation (explicit seed)."""

    def __init__(self, task: str):
        self.task = task
        self.env = gym.make("Meta-World/MT1", env_name=f"{task}-v3", seed=CONSTRUCTION_SEED)
        base = self.env.unwrapped
        if hasattr(base, "_freeze_rand_vec"):
            base._freeze_rand_vec = False
        self.model, self.data = base.model, base.data
        self.model.vis.global_.fovy = FOVY_DEG
        self.policy = ENV_POLICY_MAP[f"{task}-v3"]()

    def reset(self, seed: int):
        self.env.unwrapped.seed(int(seed))
        mujoco.mj_resetData(self.model, self.data)
        obs, _ = self.env.reset(seed=int(seed))
        return obs

    def goal_position(self) -> np.ndarray:
        return np.asarray(self.env.unwrapped._target_pos, dtype=np.float64).copy()

    def rollout(self, episode: dict, on_frame) -> dict:
        """Play one representative episode; ``on_frame(k)`` runs at render count k = 0 (reset) .. 125."""
        obs = self.reset(episode["seed"])
        on_frame(0)
        success = False
        for t in range(AGENT_STEPS):
            if episode["kind"] == "expert":
                expert = np.asarray(self.policy.get_action(obs), dtype=np.float32)
                action = episode["plan"].action(expert, t)
            else:
                action = episode["rng"].uniform(-1.0, 1.0, size=4).astype(np.float32)
            for _ in range(ACTION_REPEAT):
                obs, _, terminated, truncated, info = self.env.step(action)
                success = success or bool(info.get("success", 0.0) > 0.5)
                if terminated or truncated:
                    break
            on_frame(t + 1)
        return {"success": success}

    def close(self) -> None:
        self.env.close()


def measure_task(task: str, views: dict[str, list], *, size: int = 128, frame_stride: int = 1,
                 episodes: list[int] | None = None) -> dict:
    """Per-frame metrics of every view over the task's representative episodes.

    ``views``: name -> list of poses (one pose for a static camera; a path loops by render count)."""
    env = TaskEnv(task)
    probe = VisibilityProbe(task, env.model, size)
    names = list(views)
    rows, frames = [], []
    successes = []
    try:
        for e, episode in enumerate(representative_episodes(task)):
            if episodes is not None and e not in episodes:
                continue

            def on_frame(k: int, e: int = e) -> None:
                if k % frame_stride:
                    return
                if k == 0:
                    probe.refresh(env.model)
                poses = [views[n][k % len(views[n])] for n in names]
                rows.append(probe.measure(env.data, poses, probe.sets.goal_location(env.env.unwrapped, env.data),
                                          env.model))
                frames.append((e, k))

            successes.append(env.rollout(episode, on_frame)["success"])
    finally:
        probe.close()
        env.close()
    return {
        "task": task,
        "views": names,
        "metrics": np.stack(rows, axis=1).astype(np.float32),  # (views, frames, metrics)
        "frames": np.asarray(frames, dtype=np.int32),  # (frames, 2): episode index, render count
        "episode_kinds": [ep["kind"] for ep in representative_episodes(task)],
        "expert_success": successes,
        "image_pixels": size * size,
        "translucent_geoms_excluded": len(probe.sets.translucent_geoms),
    }


def summarize(metrics: np.ndarray, frames: np.ndarray, image_pixels: int) -> dict:
    """Aggregate one view's per-frame metrics (frames, len(METRICS)) over all frames of one task."""
    m = {name: metrics[:, i].astype(np.float64) for i, name in enumerate(METRICS[:metrics.shape[1]])}
    out: dict = {"frames": int(len(metrics))}
    for target in ("obj", "ee", "goal"):
        n, a = m[f"{target}_n"], m[f"{target}_a"]
        ratio = np.where(a > 0, np.minimum(n, a) / np.maximum(a, 1), 0.0)  # 1-pixel slivers can flip between renders
        seen = a > 0
        occluded = np.where(seen, 1.0 - ratio, np.nan)
        out[target] = {
            "visibility": float(ratio.mean()),  # mean visible share of the arm-free footprint (0 if out of view)
            "hidden_fraction": float((n == 0).mean()),  # frames with nothing visible
            "arm_occlusion_mean": float(np.nanmean(occluded)) if seen.any() else float("nan"),
            "arm_occlusion_p90": float(np.nanpercentile(occluded, 90)) if seen.any() else float("nan"),
            "arm_heavy_fraction": float((occluded[seen] > 0.5).mean()) if seen.any() else float("nan"),
            "hidden_by_arm_fraction": float(((n == 0) & seen).mean()),
            "median_count": float(np.median(n)),
            "median_pixel_fraction": float(np.median(n) / image_pixels) if target != "goal" else float("nan"),
            "longest_hidden_run": longest_run(n == 0, frames),
        }
    lost = (m["ws_inview"] == 0) | (m["ws_vis"] == 0)
    out["workspace"] = {
        "in_view_mean": float(m["ws_inview"].mean()),
        "in_view_min": float(m["ws_inview"].min()),
        "visible_mean": float(m["ws_vis"].mean()),
        "visible_min": float(m["ws_vis"].min()),
        "arm_occlusion_mean": float(np.mean(np.where(m["ws_vis_noarm"] > 0,
                                                     1 - m["ws_vis"] / np.maximum(m["ws_vis_noarm"], 1e-9), 0))),
        "lost_fraction": float(lost.mean()),
        "longest_lost_run": longest_run(lost, frames),
    }
    out["goal_point_in_view"] = float(m["goal_inview"].mean())
    out["min_depth_m"] = float(m["min_depth"].min())  # raw: includes MuJoCo's isolated depth-buffer spikes
    if "min_depth_robust" in m:
        out["min_depth_robust_m"] = float(m["min_depth_robust"].min())
    out["min_clearance_m"] = float(m["clearance"].min())
    out["score"] = float(np.mean([out[t]["visibility"] - out[t]["hidden_fraction"] for t in ("obj", "ee", "goal")]))
    return out


def longest_run(flags: np.ndarray, frames: np.ndarray) -> int:
    """Longest run of consecutive measured frames with ``flags`` inside one episode."""
    best = run = 0
    previous = None
    for flag, (episode, _) in zip(flags, frames):
        if previous is not None and episode != previous:
            run = 0
        run = run + 1 if flag else 0
        best = max(best, run)
        previous = episode
    return int(best)
