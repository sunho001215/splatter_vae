from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError("Please `pip install mujoco`") from e

from .camera_math import CameraPose


@dataclass
class RenderOut:
    rgb_by_cam: Dict[str, np.ndarray]             # uint8 (H,W,3)
    seg_id_by_cam: Optional[Dict[str, np.ndarray]]  # int32 (H,W)
    seg_type_by_cam: Optional[Dict[str, np.ndarray]] # int32 (H,W), optional


class MujocoMultiCameraRenderer:
    """
    Uses mujoco.Renderer for RGB and segmentation.
    Segmentation mode: enable_segmentation_rendering() -> render() returns (H,W,2)
      channel 0: object IDs, channel 1: object types  (per example).  :contentReference[oaicite:10]{index=10}
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        *,
        cameras: List[CameraPose],
        height: int,
        width: int,
        enable_seg: bool,
        save_objtype: bool,
    ) -> None:
        self.model = model
        self.data = data
        self.cameras = cameras
        self.height = int(height)
        self.width = int(width)
        self.enable_seg = bool(enable_seg)
        self.save_objtype = bool(save_objtype)

        self._rgb_renderer = mujoco.Renderer(model, height=self.height, width=self.width)

        self._seg_renderer = None
        if self.enable_seg:
            self._seg_renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            self._seg_renderer.enable_segmentation_rendering()

    def _mjv_cam_from_pose(self, pose: CameraPose) -> "mujoco.MjvCamera":
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # MjvCamera uses lookat + distance + azimuth + elevation for "orbit" cameras,
        # but mujoco.Renderer also supports passing an MjvCamera directly.
        # We emulate orbit values using pose.pos relative to lookat stored in cam.lookat.
        # (We set lookat via update_scene; see render_all().)
        return cam

    def render_all(self, *, lookat: np.ndarray) -> RenderOut:
        rgb_by_cam: Dict[str, np.ndarray] = {}
        seg_id_by_cam: Optional[Dict[str, np.ndarray]] = {} if self.enable_seg else None
        seg_type_by_cam: Optional[Dict[str, np.ndarray]] = {} if (self.enable_seg and self.save_objtype) else None

        for pose in self.cameras:
            # Build an MjvCamera in orbit terms from spherical placement
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat.astype(np.float64)

            # Convert pose position back into orbit parameters:
            rel = pose.pos.astype(np.float64) - lookat.astype(np.float64)
            cam.distance = float(np.linalg.norm(rel))

            # azimuth: angle in XY plane from +X
            az = np.degrees(np.arctan2(rel[1], rel[0]))
            cam.azimuth = float(az)

            # elevation: angle above XY plane
            xy = np.sqrt(rel[0] ** 2 + rel[1] ** 2)
            el = np.degrees(np.arctan2(rel[2], xy))
            cam.elevation = float(el)

            # RGB
            self._rgb_renderer.update_scene(self.data, camera=cam)
            rgb = self._rgb_renderer.render()
            rgb_by_cam[pose.name] = rgb.astype(np.uint8)

            # Segmentation
            if self.enable_seg and self._seg_renderer is not None:
                self._seg_renderer.update_scene(self.data, camera=cam)
                seg = self._seg_renderer.render()  # (H,W,2)
                seg = np.asarray(seg)

                seg_id = seg[:, :, 0].astype(np.int32)    # IDs
                if seg_id_by_cam is not None:
                    seg_id_by_cam[pose.name] = seg_id

                if self.save_objtype and seg_type_by_cam is not None:
                    seg_type_by_cam[pose.name] = seg[:, :, 1].astype(np.int32)

        return RenderOut(
            rgb_by_cam=rgb_by_cam,
            seg_id_by_cam=seg_id_by_cam,
            seg_type_by_cam=seg_type_by_cam,
        )