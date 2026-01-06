# demo_collector/spacemouse_pyspacemouse.py
"""
Linux-friendly SpaceMouse device using `pyspacemouse`.

- Works on Python 3.x
- Still requires HID access (udev rule), see PySpaceMouse docs. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _get_action_dim(env) -> int:
    # robosuite envs usually expose action_spec
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    if hasattr(env, "action_spec"):
        lo, hi = env.action_spec
        return int(np.asarray(lo).size)
    # last resort
    return 7


def _getattr_or_key(x: Any, name: str, default=0.0):
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default


class SpaceMousePySpaceMouse:
    """
    Minimal "robosuite device-like" interface:
      - start_control()
      - input2action(...) -> np.ndarray or None (reset)
    """

    def __init__(
        self,
        env,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        deadband: float = 0.02,
        max_raw: float = 350.0,
        reset_button_index: int = 1,
        gripper_button_index: int = 0,
    ):
        self.env = env
        self.action_dim = _get_action_dim(env)

        self.pos_sensitivity = float(pos_sensitivity)
        self.rot_sensitivity = float(rot_sensitivity)
        self.deadband = float(deadband)
        self.max_raw = float(max_raw)

        self.reset_button_index = int(reset_button_index)
        self.gripper_button_index = int(gripper_button_index)

        self._gripper_closed = False
        self._prev_buttons = None

        try:
            import pyspacemouse
        except ImportError as e:
            raise ImportError(
                "pyspacemouse not installed. Install it (and udev rules) per project docs."
            ) from e

        self.pyspacemouse = pyspacemouse

    def start_control(self):
        # opens the first detected SpaceMouse device
        ok = self.pyspacemouse.open()
        if not ok:
            raise OSError("pyspacemouse.open() failed (device not found or no permissions).")

        # prime one read so buttons state exists
        _ = self.pyspacemouse.read()
        self._prev_buttons = None

    def _apply_deadband(self, v: np.ndarray) -> np.ndarray:
        v2 = v.copy()
        v2[np.abs(v2) < self.deadband] = 0.0
        return v2

    def input2action(self, goal_update_mode: str = "target") -> Optional[np.ndarray]:
        """
        Returns:
          - np.ndarray action for env.step
          - None if reset button pressed
        """
        st = self.pyspacemouse.read()
        if st is None:
            # no new event; output "do nothing" action
            return np.zeros((self.action_dim,), dtype=np.float32)

        # Typical fields in pyspacemouse state are x,y,z, roll,pitch,yaw, and buttons[] 
        tx = float(_getattr_or_key(st, "x", 0.0))
        ty = float(_getattr_or_key(st, "y", 0.0))
        tz = float(_getattr_or_key(st, "z", 0.0))
        rr = float(_getattr_or_key(st, "roll", 0.0))
        rp = float(_getattr_or_key(st, "pitch", 0.0))
        ry = float(_getattr_or_key(st, "yaw", 0.0))

        buttons = _getattr_or_key(st, "buttons", None)
        if buttons is None:
            buttons = _getattr_or_key(st, "button", [])
        buttons = list(buttons) if buttons is not None else []

        # Edge detection for buttons
        prev = self._prev_buttons
        self._prev_buttons = buttons

        def pressed_edge(i: int) -> bool:
            if i < 0:
                return False
            cur = (i < len(buttons)) and bool(buttons[i])
            prv = (prev is not None) and (i < len(prev)) and bool(prev[i])
            return cur and (not prv)

        # Reset
        if pressed_edge(self.reset_button_index):
            return None

        # Toggle gripper
        if pressed_edge(self.gripper_button_index):
            self._gripper_closed = not self._gripper_closed

        # Normalize raw values to [-1,1] approximately
        dpos = np.array([tx, ty, tz], dtype=np.float32) / self.max_raw
        drot = np.array([rr, rp, ry], dtype=np.float32) / self.max_raw

        dpos = self._apply_deadband(dpos) * self.pos_sensitivity
        drot = self._apply_deadband(drot) * self.rot_sensitivity

        # Robosuite OSC pose actions are commonly [dx,dy,dz, dRx,dRy,dRz, gripper]
        grip = -1.0 if self._gripper_closed else 1.0

        act7 = np.concatenate([dpos, drot, np.array([grip], dtype=np.float32)], axis=0)

        # Fit to env action dim safely
        if act7.size < self.action_dim:
            act = np.pad(act7, (0, self.action_dim - act7.size), mode="constant")
        else:
            act = act7[: self.action_dim]

        return act.astype(np.float32)
