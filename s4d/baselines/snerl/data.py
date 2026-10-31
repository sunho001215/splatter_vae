"""SNeRL data adapter: our Meta-World HDF5 scenes -> the inputs of the official pretraining loop (run_nerf.py:685-745).

Official data (``load_blender.py``): ``images`` (V, T, H, W, 3) in [0, 1], white background; ``poses`` (V, T, 4, 4)
NeRF/OpenGL camera-to-world matrices with the cameras about 4 units from the scene centre (near/far 2/6); one focal
length with the principal point at ``0.5 * W`` in the integer-pixel convention of ``get_rays`` (the ray of pixel
column ``i`` passes through ``i - cx``); per-pixel semantic labels (PNG / 50, five values).

Our data: per episode ``rgb`` (T, 10, H, W, 3) uint8 (6 training cameras first), ``body_id`` (T, 6, H, W) MuJoCo body
ids of the training cameras (``background_body`` where no geometry), OpenCV ``K``/``c2w`` in continuous pixel
coordinates (pixel ``i`` covers [i, i+1)), cameras 1.0 m from the rig look-at point ``LOOKAT``. Conversions:

- poses: ``c2w_gl = c2w_cv @ diag(1, -1, -1, 1)``, re-centred on ``LOOKAT`` and scaled by ``POSE_SCALE`` = 4 (camera
  distance 4, the official geometry, so the official near/far 2/6 are 0.5 m / 1.5 m camera-z depth); the same
  normalised, flattened 4x4 matrix is the encoder's pose input in pretraining and RL;
- intrinsics: the rig K is used unchanged (principal point ``W/2``), as directed; official rays sample integer
  coordinates without +0.5, so they differ by half a pixel from our other continuous-pixel render/lift convention;
- semantics: body ids -> 4 classes (``SEMANTIC_CLASSES``) by a per-task table built from the file's ``body_names``;
  the official 5-way semantic head is kept, class 4 is never a target;
- no background compositing: our renders are full scenes without alpha (the official data was masked to white with
  simulator depth, which is privileged at RL time).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from s4d.data.metaworld.cameras import LOOKAT

POSE_SCALE = 4.0  # official scene scale: cameras 4 units from the centre (pose_spherical(..., 4)), near/far 2/6
GL_SIGN = (1.0, -1.0, -1.0, 1.0)  # columns of c2w_cv @ diag(1, -1, -1, 1): OpenCV -> OpenGL camera axes

SEMANTIC_CLASSES = ("background", "robot", "object", "fixture")  # the official 5-way head; class 4 is unused
NUM_SEMANTIC_OUTPUTS = 5
BACKGROUND, ROBOT, OBJECT, FIXTURE = range(4)
BACKGROUND_BODIES = ("world",)  # the world body's only geom is the floor plane; no-geometry pixels are background too
STATIC_FIXTURES = ("tablelink", "RetainingWall")
ROBOT_BODIES = (  # the Sawyer arm, pedestal, head and gripper of every Meta-World task model
    "base", "controller_box", "pedestal_feet", "torso", "pedestal", "right_arm_base_link", "right_l0", "head", "screen",
    "head_camera", "right_torso_itb", "right_l1", "right_l2", "right_l3", "right_l4", "right_arm_itb", "right_l5",
    "right_hand_camera", "right_wrist", "right_l6", "right_hand", "hand", "rightclaw", "rightpad", "leftclaw", "leftpad",
    "right_l4_2", "right_l2_2", "right_l1_2", "mocap",
)  # fmt: skip
# Manipulated objects (bodies the task moves, with their geometry children) and task fixtures, by HDF5 body name;
# unnamed MuJoCo bodies are stored as ``body<id>``.
TASK_BODIES = {
    "hammer": {"object": ("hammerbody", "hammer", "nail_link"), "fixture": ("box", "hammerblock")},
    "pick-place": {"object": ("obj",), "fixture": ()},
    "bin-picking": {"object": ("obj", "objA"), "fixture": ("bin_start", "binA", "bin_goal", "binB")},
    "coffee-push": {"object": ("obj", "mug"), "fixture": ("coffee_machine", "coffeemachine", "cm_link", "cmbutton")},
    "door-open": {"object": ("door_link",), "fixture": ("door", "doorlockB")},
    "peg-insert-side": {"object": ("peg",), "fixture": ("box", "body35")},
    "peg-unplug-side": {"object": ("plug1", "plug"), "fixture": ("box", "plug_wall")},
    "stick-push": {"object": ("stick", "body34", "object", "body36"), "fixture": ()},
    # New tasks (user approval 2026-10-31): bodies that move by > 1 cm within some training episode (training-split
    # xpos) are objects, the remaining task bodies fixtures. "absent": standard static fixtures that the task model
    # does not contain (the sweep-into model has no RetainingWall).
    "drawer-open": {"object": ("drawer_link",), "fixture": ("drawer", "body34", "drawercase_link")},
    "window-open": {"object": ("windowb_a",), "fixture": ("window", "body34", "windowb_b")},
    "soccer": {"object": ("soccer_ball", "soccer_ball1"), "fixture": ("goal_whole", "soccer_goal")},
    "sweep-into": {"object": ("obj",), "fixture": (), "absent": ("RetainingWall",)},
}


def class_table(task: str, body_names: list[str]) -> np.ndarray:
    """Semantic class of every body (index = MuJoCo body id). Every body must be assigned exactly once."""
    groups = {
        BACKGROUND: BACKGROUND_BODIES,
        ROBOT: ROBOT_BODIES,
        OBJECT: TASK_BODIES[task]["object"],
        FIXTURE: tuple(n for n in STATIC_FIXTURES if n not in TASK_BODIES[task].get("absent", ()))
        + TASK_BODIES[task]["fixture"],
    }
    table = np.full(len(body_names), -1, dtype=np.int64)
    for cls, names in groups.items():
        for name in names:
            if name not in body_names:
                raise ValueError(f"{task}: body {name!r} of class {SEMANTIC_CLASSES[cls]} is not in the model")
            idx = body_names.index(name)
            if table[idx] != -1:
                raise ValueError(f"{task}: body {name!r} assigned twice")
            table[idx] = cls
    if (missing := [body_names[i] for i in np.where(table < 0)[0]]):
        raise ValueError(f"{task}: bodies without a semantic class: {missing}")
    return table.astype(np.uint8)


def class_lookup(table: np.ndarray, background_body: int) -> np.ndarray:
    """uint16 body id -> class lookup table (65536 entries); the no-geometry id maps to background."""
    if background_body < len(table):
        raise ValueError("the background id collides with a body id")
    lut = np.zeros(1 << 16, dtype=np.uint8)
    lut[: len(table)] = table
    lut[background_body] = BACKGROUND
    return lut


def normalise_c2w(c2w_cv: torch.Tensor, lookat=LOOKAT, scale: float = POSE_SCALE) -> torch.Tensor:
    """OpenCV camera-to-world (..., 4, 4) -> the official NeRF/OpenGL convention in normalised scene units."""
    out = c2w_cv * c2w_cv.new_tensor(GL_SIGN)
    centre = torch.as_tensor(lookat, dtype=c2w_cv.dtype, device=c2w_cv.device)
    out[..., :3, 3] = (out[..., :3, 3] - centre) * scale
    return out


def denormalise_c2w(c2w: torch.Tensor, lookat=LOOKAT, scale: float = POSE_SCALE) -> torch.Tensor:
    """Inverse of ``normalise_c2w``."""
    out = c2w * c2w.new_tensor(GL_SIGN)
    out[..., :3, 3] = c2w[..., :3, 3] / scale + torch.as_tensor(lookat, dtype=c2w.dtype, device=c2w.device)
    return out


def official_K(K_cv: np.ndarray) -> np.ndarray:
    """Per-camera rig K unchanged: the official rays have no half-pixel offset (user-directed convention)."""
    return np.array(K_cv, dtype=np.float64, copy=True)


def file_class_lookup(path: str | Path) -> np.ndarray:
    """Body id -> class lookup of one task file (its ``task``, ``body_names`` and ``background_body`` attributes)."""
    with h5py.File(path, "r") as f:
        table = class_table(str(f.attrs["task"]), json.loads(f.attrs["body_names"]))
        return class_lookup(table, int(f.attrs["background_body"]))


class SNeRLScenes(Dataset):
    """Official training scenes from our episodes: item ``i`` is one time step of one episode, drawn uniformly over
    all frames (official ``time_index``, run_nerf.py:687), ``num_inputs`` distinct training cameras as encoder inputs
    (``obs_index``, :694) and a target camera uniform over all training cameras (``img_i = np.random.choice(i_train)``,
    :705, with ``i_train`` all views). Draws use ``np.random.default_rng([seed, i])``, so a resumed run continues the
    same sequence and the loader's worker count never changes it. Images stay uint8 (the trainer divides by 255)."""

    def __init__(self, path: str | Path, episodes: list[str], seed: int, num_inputs: int = 3, length: int | None = None,
                 lookat=LOOKAT, scale: float = POSE_SCALE):
        self.path, self.episodes, self.seed, self.num_inputs = str(path), list(episodes), int(seed), int(num_inputs)
        self.lookat, self.scale = tuple(float(x) for x in lookat), float(scale)
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.task = str(f.attrs["task"])
            cams = f["cameras"]
            train = np.where(cams["is_train"][:].astype(bool))[0]
            if not np.array_equal(train, np.arange(len(train))):
                raise ValueError("training cameras must be stored first in the camera table")
            self.num_cams = len(train)
            self.K_cv = cams["K"][:].astype(np.float64)
            self.c2w_cv = cams["c2w"][:].astype(np.float64)
            self.lengths = np.array([int(f["episodes"][ep].attrs["length"]) for ep in self.episodes])
            table = class_table(self.task, json.loads(f.attrs["body_names"]))
            self.lut = class_lookup(table, int(f.attrs["background_body"]))
        if self.num_inputs > self.num_cams:
            raise ValueError(f"{self.num_inputs} input views exceed the {self.num_cams} training cameras")
        self.c2w = normalise_c2w(torch.from_numpy(self.c2w_cv), self.lookat, self.scale).float()
        self.K = official_K(self.K_cv)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.length = int(length) if length is not None else int(self.offsets[-1])

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def close(self) -> None:
        """Close the HDF5 handle (before DataLoader workers fork, which must not inherit an open file)."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _episode(self, ep: str):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file["episodes"][ep]

    def __len__(self) -> int:
        return self.length

    def draw(self, index: int) -> tuple[str, int, list[int], int]:
        """(episode, frame, input cameras, target camera) of item ``index``, in the official draw order."""
        rng = np.random.default_rng([self.seed, int(index)])
        frame = int(rng.integers(self.offsets[-1]))
        e = int(np.searchsorted(self.offsets, frame, side="right") - 1)
        inputs = rng.choice(self.num_cams, self.num_inputs, replace=False).tolist()
        target = int(rng.integers(self.num_cams))
        return self.episodes[e], frame - int(self.offsets[e]), inputs, target

    def scene(self, ep: str, t: int, inputs: list[int], target: int | None, with_depth: bool = False) -> dict:
        """Encoder inputs of frame ``t`` and, for a training camera ``target``, its RGB and semantic target."""
        group = self._episode(ep)
        out = {
            "obs_rgb": np.stack([group["rgb"][t, v] for v in inputs]),  # (V_in, H, W, 3) uint8
            "obs_c2w": self.c2w[inputs].clone(),  # (V_in, 4, 4) normalised OpenGL
            "inputs": torch.as_tensor(inputs),
            "episode": ep,
            "t": int(t),
        }
        if target is not None:
            out.update(
                target_rgb=group["rgb"][t, target],
                target_semantic=self.lut[group["body_id"][t, target]].astype(np.int64),
                target_c2w=self.c2w[target].clone(),
                target_K=torch.from_numpy(self.K[target]),
                target=int(target),
            )
            if with_depth:
                out["target_depth"] = group["depth"][t, target].astype(np.float32) * self.depth_unit()
        return out

    def depth_unit(self) -> float:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return float(self._file.attrs["depth_unit_m"])

    def __getitem__(self, index: int) -> dict:
        ep, t, inputs, target = self.draw(index)
        return self.scene(ep, t, inputs, target)
