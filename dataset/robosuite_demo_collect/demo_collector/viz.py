from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise ImportError("OpenCV is required for visualization: pip install opencv-python") from e


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    # rgb: HxWx3 uint8
    return rgb[..., ::-1].copy()


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Fast deterministic pseudo-color for int32 seg IDs.

    seg: HxW int32 (e.g., objid or instance id)
    returns: HxWx3 uint8 in BGR (for OpenCV)
    """
    s = np.asarray(seg).astype(np.int64)
    s = np.where(s < 0, 0, s)

    # Cheap hashing into 0..255
    b = (s * 1) & 255
    g = (s * 23) & 255
    r = (s * 63) & 255
    out = np.stack([b, g, r], axis=-1).astype(np.uint8)

    # Make background (id==0) black
    out[s == 0] = 0
    return out


def tile_images(imgs: List[np.ndarray], ncols: int, pad: int = 2) -> np.ndarray:
    """
    Tile a list of equally-sized HxWxC images into a grid (OpenCV-style).
    """
    assert len(imgs) > 0
    H, W = imgs[0].shape[:2]
    C = imgs[0].shape[2] if imgs[0].ndim == 3 else 1

    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    canvas_h = nrows * H + (nrows - 1) * pad
    canvas_w = ncols * W + (ncols - 1) * pad

    if C == 1:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_h, canvas_w, C), dtype=np.uint8)

    for i, im in enumerate(imgs):
        r = i // ncols
        c = i % ncols
        y0 = r * (H + pad)
        x0 = c * (W + pad)
        canvas[y0 : y0 + H, x0 : x0 + W] = im

    return canvas


@dataclass
class MultiCamVisualizer:
    camera_names: List[str]
    H: int
    W: int
    window_name: str = "robosuite demo (top=RGB, bottom=SEG)"
    ncols: int = 3
    pad: int = 4
    save_video_path: Optional[str] = None
    video_fps: float = 10.0

    def __post_init__(self):
        self._writer = None

    def update(
        self,
        rgb_by_cam: Dict[str, np.ndarray],   # RGB uint8
        seg_by_cam: Dict[str, np.ndarray],   # int32 mask
        step_i: Optional[int] = None,
    ) -> int:
        """
        Shows one combined window:
          - top: 6 RGB tiled
          - bottom: 6 SEG color tiled

        Returns: key from cv2.waitKey(1) (or -1 if none)
        """
        rgb_tiles = []
        seg_tiles = []

        for cam in self.camera_names:
            rgb = np.asarray(rgb_by_cam[cam])
            seg = np.asarray(seg_by_cam[cam])

            if rgb.shape[:2] != (self.H, self.W):
                raise ValueError(f"RGB size mismatch for {cam}: {rgb.shape}")
            if seg.shape[:2] != (self.H, self.W):
                raise ValueError(f"SEG size mismatch for {cam}: {seg.shape}")

            rgb_tiles.append(_rgb_to_bgr(rgb))
            seg_tiles.append(colorize_segmentation(seg))

        top = tile_images(rgb_tiles, ncols=self.ncols, pad=self.pad)
        bot = tile_images(seg_tiles, ncols=self.ncols, pad=self.pad)
        vis = np.vstack([top, bot])

        if step_i is not None:
            cv2.putText(
                vis,
                f"step={step_i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        
        cv2.imshow(self.window_name, vis)

        # Lazy-init video writer once we know final size
        if self.save_video_path and self._writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.save_video_path, fourcc, self.video_fps, (w, h))

        if self._writer is not None:
            self._writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        cv2.destroyWindow(self.window_name)