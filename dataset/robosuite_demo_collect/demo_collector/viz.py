# demo_collector/viz.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    Fast deterministic pseudo-color for int32 seg IDs.

    seg: HxW int32 (e.g., objid or instance id)
    returns: HxWx3 uint8 RGB (matplotlib expects RGB)
    """
    s = np.asarray(seg).astype(np.int64)
    s = np.where(s < 0, 0, s)

    # Cheap hashing into 0..255
    r = (s * 37) & 255
    g = (s * 17) & 255
    b = (s * 29) & 255
    out = np.stack([r, g, b], axis=-1).astype(np.uint8)

    # Background (id==0) black
    out[s == 0] = 0
    return out


@dataclass
class MultiCamVisualizer:
    camera_names: List[str]
    H: int
    W: int
    window_title: str = "robosuite demo (top=RGB, bottom=SEG)"
    show: bool = True
    pause_s: float = 0.001  # event-loop tick; plt.pause runs GUI loop :contentReference[oaicite:1]{index=1}

    def __post_init__(self):
        self._n = len(self.camera_names)
        if self._n == 0:
            raise ValueError("camera_names is empty")

        self._last_key: Optional[str] = None

        # Interactive mode: figures update without blocking. :contentReference[oaicite:2]{index=2}
        if self.show:
            plt.ion()

        self.fig, self.axes = plt.subplots(
            2,
            self._n,
            figsize=(2.6 * self._n, 5.2),
            squeeze=False,
        )
        try:
            # Some backends support this
            self.fig.canvas.manager.set_window_title(self.window_title)
        except Exception:
            pass

        # Pre-create image artists (so update() is cheap)
        blank_rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        blank_seg = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        self._rgb_artists = []
        self._seg_artists = []

        for i, cam in enumerate(self.camera_names):
            ax_rgb = self.axes[0, i]
            ax_seg = self.axes[1, i]

            ax_rgb.set_title(f"{cam} RGB")
            ax_seg.set_title(f"{cam} SEG")

            ax_rgb.axis("off")
            ax_seg.axis("off")

            rgb_im = ax_rgb.imshow(blank_rgb, interpolation="nearest")
            seg_im = ax_seg.imshow(blank_seg, interpolation="nearest")

            self._rgb_artists.append(rgb_im)
            self._seg_artists.append(seg_im)

        # Key handling (optional)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Show non-blocking
        if self.show:
            self.fig.tight_layout()
            self.fig.show()
            # One initial event-loop tick
            plt.pause(self.pause_s)

    def _on_key(self, event):
        # event.key is a string like 'q', 'escape', 'r', etc.
        self._last_key = event.key

    def update(
        self,
        rgb_by_cam: Dict[str, np.ndarray],  # RGB uint8
        seg_by_cam: Dict[str, np.ndarray],  # int32 mask (HxW)
        step_i: Optional[int] = None,
    ) -> Optional[str]:
        """
        Update the existing figure artists with new images.

        Returns:
            last pressed key (e.g. 'q', 'escape', 'r') or None
        """
        for i, cam in enumerate(self.camera_names):
            rgb = np.asarray(rgb_by_cam[cam])
            seg = np.asarray(seg_by_cam[cam])

            if rgb.shape[:2] != (self.H, self.W) or rgb.shape[-1] != 3:
                raise ValueError(f"RGB size mismatch for {cam}: got {rgb.shape}, expected ({self.H},{self.W},3)")
            if seg.shape[:2] != (self.H, self.W):
                raise ValueError(f"SEG size mismatch for {cam}: got {seg.shape}, expected ({self.H},{self.W})")

            self._rgb_artists[i].set_data(rgb)
            self._seg_artists[i].set_data(colorize_segmentation(seg))

            if step_i is not None:
                # Update row titles with step (optional, cheap)
                self.axes[0, i].set_title(f"{cam} RGB | t={step_i}")
                self.axes[1, i].set_title(f"{cam} SEG | t={step_i}")

        if self.show:
            # draw_idle + pause is a common interactive pattern; pause runs GUI loop. :contentReference[oaicite:3]{index=3}
            self.fig.canvas.draw_idle()
            plt.pause(self.pause_s)

        k = self._last_key
        self._last_key = None
        return k

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
