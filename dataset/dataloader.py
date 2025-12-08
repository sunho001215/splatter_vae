# dataloader.py

import os
import json
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.general_utils import load_image_rgb, image_to_tensor, invert_4x4

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class MultiViewTemporalDataset(Dataset):
    """
    Dataset that:
      - reads scene manifests (train.json / valid.json),
      - uses cameras.json for intrinsics & extrinsics,
      - returns:
            image_i_t, image_j_t, image_i_t1,
            T_ij, K_i, K_j

    Where:
      * image_i_t, image_j_t: same scene, same timestep, different cameras
      * image_i_t1: same scene + camera_i, but different random timestep
      * T_ij: transform from camera-i coordinates to camera-j view coordinates
              (used as world_view_transform in your training loop)
      * K_i, K_j: 3×3 intrinsics (fx, fy, cx, cy) for the **downscaled** images.

    Notes:
      - Instead of raw RGB, we load **masked RGB** using the dynamic union mask:
            scene_name/masks/_union_dynamic/<cam>/<frame>.png
        Masked regions keep the original RGB; background is black.
      - Output tensors are still (3,H,W) float32 in [-1,1] (via image_to_tensor).
    """

    def __init__(
        self,
        dataset_root: str,
        manifest_path: str,
        cameras_json_path: str,
    ):
        """
        Args:
            dataset_root: path to directory that contains scene folders.
                          For example: ".../output_root" from your generator.
            manifest_path: path to train.json or valid.json.
            cameras_json_path: path to cameras.json.
        """
        super().__init__()
        self.dataset_root = dataset_root

        # -------------------- Load camera parameters --------------------
        with open(cameras_json_path, "r") as f:
            cam_meta = json.load(f)
        cam_dict = cam_meta["cameras"]  # "cam00", "cam01", ...

        # Precompute intrinsics and extrinsics per camera.
        self.cameras: Dict[str, Dict[str, np.ndarray]] = {}
        for cam_id, info in cam_dict.items():
            fx = float(info["fx"])
            fy = float(info["fy"])
            cx = float(info["cx"])
            cy = float(info["cy"])

            # 3x3 intrinsics matrix
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            # cameras.json stores c2w in **OpenGL-style** camera coordinates:
            #   +x right, +y up, +z backward (camera looks along -Z)
            c2w_gl = np.array(info["c2w"], dtype=np.float32)

            # We want to use **OpenCV-style** camera coordinates everywhere:
            #   +x right, +y down, +z forward.
            S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            c2w = c2w_gl @ S

            # Compute world-to-camera (view) matrix.
            w2c = invert_4x4(c2w)

            self.cameras[cam_id] = {
                "K": K,
                "c2w": c2w,
                "w2c": w2c,
            }

        # -------------------- Load scene manifest ----------------------
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # manifest["scenes"] is a list of scenes
        self.scene_infos: List[Dict[str, Any]] = []
        for scene in manifest["scenes"]:
            # Each scene has:
            #   "name": "scene_000",
            #   "images_root": "scene_000/images",
            #   "frame_index": { "start": 1, "count": saved_count, "step": 1 },
            #   "cameras": { "cam00": { "size": [H,W], "path_format": "cam00/{:06d}.png" }, ... }
            scene_name = scene["name"]
            images_root_rel = scene["images_root"]  # e.g. "scene_000/images"
            images_root = os.path.join(self.dataset_root, images_root_rel)

            frame_info = scene["frame_index"]
            start = int(frame_info["start"])
            count = int(frame_info["count"])
            step = int(frame_info.get("step", 1))

            timesteps = list(range(start, start + count * step, step))

            # Cameras available in this scene
            scene_cam_ids = sorted(scene["cameras"].keys())
            # Only keep cameras that exist in cameras.json
            scene_cam_ids = [cid for cid in scene_cam_ids if cid in self.cameras]

            cams_info = {}
            for cid in scene_cam_ids:
                cams_info[cid] = {
                    "path_format": scene["cameras"][cid]["path_format"]  # e.g. "cam00/{:06d}.png"
                }

            self.scene_infos.append(
                dict(
                    id=scene["id"],
                    name=scene_name,
                    images_root=images_root,
                    timesteps=timesteps,
                    cam_ids=scene_cam_ids,
                    cams_info=cams_info,
                )
            )

        # ---------------- Build global index over (scene, t, cam_i, cam_j) ----
        # We precompute all possible tuples where:
        #   cam_i != cam_j
        #   t is any timestep in that scene
        # During __getitem__ we will additionally sample a random t1 != t.
        self.samples: List[Tuple[int, int, int, int]] = []
        for s_idx, sinfo in enumerate(self.scene_infos):
            cam_ids = sinfo["cam_ids"]
            timesteps = sinfo["timesteps"]
            for t_idx in range(len(timesteps)):
                for ci_idx, cam_i in enumerate(cam_ids):
                    for cj_idx, cam_j in enumerate(cam_ids):
                        if cam_i == cam_j:
                            continue
                        self.samples.append((s_idx, t_idx, ci_idx, cj_idx))

        print(f"[Dataset] Loaded {len(self.scene_infos)} scenes, {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    # -------------------- internal helper: load image + mask ------------------

    def _load_image_and_mask(
        self,
        rgb_path: str,
        scene_name: str,
        cam_id: str,
    ):
        """
        Load RGB image (unmasked) and the dynamic union mask.

        Returns:
            image_tensor: (3,H,W) float32 in [-1,1]
            mask_tensor:  (1,H,W) float32 in {0,1}
        """
        # Load RGB as HxWx3 uint8 (original image, no masking)
        img = load_image_rgb(rgb_path)  # np.uint8, (H,W,3)
        H, W, _ = img.shape

        # Build mask path: same filename as RGB, under masks/_union_dynamic/<cam_id>/
        scene_root = os.path.join(self.dataset_root, scene_name)
        filename = os.path.basename(rgb_path)
        mask_path = os.path.join(
            scene_root, "masks", "_union_dynamic", cam_id, filename
        )

        if os.path.exists(mask_path):
            # Load mask as grayscale, shape (H,W)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback: no valid mask, use full 1s (no masking)
                mask = np.ones((H, W), dtype=np.uint8) * 255
            else:
                # Ensure mask matches image size
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            # If mask file does not exist, fall back to full 1s (no masking)
            mask = np.ones((H, W), dtype=np.uint8) * 255

        # Convert mask to {0,1} float and add channel dim → (1,H,W)
        mask01 = (mask > 0).astype(np.float32)          # (H,W)
        mask_tensor = torch.from_numpy(mask01)[None, :] # (1,H,W)

        # Original image tensor in [-1,1], (3,H,W)
        image_tensor = image_to_tensor(img)

        return image_tensor, mask_tensor

    # -------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict with:
            image_i_t:  (3,H,W) float32 in [-1,1], masked by dynamic union mask
            image_j_t:  (3,H,W) float32 in [-1,1], masked by dynamic union mask
            image_i_t1: (3,H,W) float32 in [-1,1], masked by dynamic union mask
            T_ij:       (4,4) float32 (transform cam_i -> cam_j view)
            K_i:        (3,3) float32 intrinsics for cam_i 
            K_j:        (3,3) float32 intrinsics for cam_j 
            scene_id, t, t1, cam_i, cam_j: metadata (for debugging)
        """
        s_idx, t_idx, ci_idx, cj_idx = self.samples[idx]
        sinfo = self.scene_infos[s_idx]

        timesteps = sinfo["timesteps"]
        t = timesteps[t_idx]

        cam_ids = sinfo["cam_ids"]
        cam_i = cam_ids[ci_idx]
        cam_j = cam_ids[cj_idx]

        images_root = sinfo["images_root"]
        cams_info = sinfo["cams_info"]
        scene_name = sinfo["name"]

        # --------------------- Load original images + masks -------------------
        # image_i_t: cam_i, timestep t
        rel_i_t = cams_info[cam_i]["path_format"].format(t)
        path_i_t = os.path.join(images_root, rel_i_t)
        tensor_i_t, mask_i_t = self._load_image_and_mask(path_i_t, scene_name, cam_i)

        # image_j_t: cam_j, timestep t
        rel_j_t = cams_info[cam_j]["path_format"].format(t)
        path_j_t = os.path.join(images_root, rel_j_t)
        tensor_j_t, mask_j_t = self._load_image_and_mask(path_j_t, scene_name, cam_j)

        # image_i_t1: same camera i, random different timestep t1
        if len(timesteps) > 1:
            other_indices = [k for k in range(len(timesteps)) if k != t_idx]
            t1_idx = random.choice(other_indices)
        else:
            t1_idx = t_idx
        t1 = timesteps[t1_idx]

        rel_i_t1 = cams_info[cam_i]["path_format"].format(t1)
        path_i_t1 = os.path.join(images_root, rel_i_t1)
        tensor_i_t1, mask_i_t1 = self._load_image_and_mask(path_i_t1, scene_name, cam_i)

        # ---------------------- Camera matrices ------------------
        cam_i_meta = self.cameras[cam_i]
        cam_j_meta = self.cameras[cam_j]

        K_i = torch.from_numpy(cam_i_meta["K"])     # (3,3)
        K_j = torch.from_numpy(cam_j_meta["K"])     # (3,3)

        c2w_i = cam_i_meta["c2w"]                   # (4,4)
        w2c_j = cam_j_meta["w2c"]                   # (4,4)

        # Transform from cam_i coordinates to cam_j view coordinates:
        #
        # Given:
        #   X_world = c2w_i * X_cam_i
        #   X_cam_j = w2c_j * X_world
        #
        # => X_cam_j = (w2c_j * c2w_i) * X_cam_i
        #
        # We use this as T_ij (world_view_transform) while the Gaussian
        # positions are expressed in cam_i frame (treated as "world").
        T_ij_np = w2c_j @ c2w_i
        T_ij = torch.from_numpy(T_ij_np.astype(np.float32))  # (4,4)

        return {
            # Original images (no masking), in [-1,1]
            "image_i_t": tensor_i_t,
            "image_j_t": tensor_j_t,
            "image_i_t1": tensor_i_t1,

            # Dynamic masks in {0,1}, shape (1,H,W)
            "mask_i_t": mask_i_t,
            "mask_j_t": mask_j_t,
            "mask_i_t1": mask_i_t1,

            # Camera transforms & metadata
            "T_ij": T_ij,
            "K_i": K_i,
            "K_j": K_j,
            "scene_id": sinfo["id"],
            "scene_name": scene_name,
            "t": t,
            "t1": t1,
            "cam_i": cam_i,
            "cam_j": cam_j,
        }

# ---------------------------------------------------------------
# Convenience builders for train / valid dataloaders
# ---------------------------------------------------------------

def build_train_valid_loaders(
    dataset_root: str,
    train_manifest: str,
    valid_manifest: str,
    cameras_json_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Builds DataLoaders for training and validation.

    Args:
        dataset_root: directory that contains the scene_* folders.
        train_manifest: path to train.json
        valid_manifest: path to valid.json
        cameras_json_path: path to cameras.json
    """
    train_dataset = MultiViewTemporalDataset(
        dataset_root=dataset_root,
        manifest_path=train_manifest,
        cameras_json_path=cameras_json_path,
    )
    valid_dataset = MultiViewTemporalDataset(
        dataset_root=dataset_root,
        manifest_path=valid_manifest,
        cameras_json_path=cameras_json_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size*2,
        shuffle=True,
        num_workers=1,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, valid_loader
