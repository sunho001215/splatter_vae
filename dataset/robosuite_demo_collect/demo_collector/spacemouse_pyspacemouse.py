from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import threading
import time
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
    """
    Helper that works with both attribute-style and dict-style access.
    """
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

    This version uses a background thread to continuously read from
    `pyspacemouse` and cache the latest state. The control loop then
    reads this cached state without blocking on HID I/O.
    """

    def __init__(
        self,
        env,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        deadband: float = 0.005,
        max_raw: float = 1.5,
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

        # Internal state for gripper and button edge detection
        self._gripper_closed = False
        self._prev_buttons = None

        # Shared SpaceMouse state (updated by background thread)
        self._state: Optional[Any] = None
        self._lock = threading.Lock()

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

        try:
            import pyspacemouse
        except ImportError as e:
            raise ImportError(
                "pyspacemouse not installed. Install it (and udev rules) per project docs."
            ) from e

        self.pyspacemouse = pyspacemouse

    def start_control(self):
        """
        Open the first detected SpaceMouse device and start the reader thread.

        The reader thread continuously calls pyspacemouse.read() and stores
        the latest state in self._state. The control loop should call
        input2action(), which only reads this cached state and never blocks
        on HID.
        """
        ok = self.pyspacemouse.open()
        if not ok:
            raise OSError("pyspacemouse.open() failed (device not found or no permissions).")

        # Start background reader thread
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        # Button edge detection state is maintained in the control thread
        self._prev_buttons = None

    def _reader_loop(self):
        """
        Background loop that continuously pulls events from the SpaceMouse.

        This keeps the latest state in self._state so the main control loop
        can poll it with minimal latency.
        """
        while self._running:
            try:
                st = self.pyspacemouse.read()
                if st is not None:
                    # Store latest state atomically
                    with self._lock:
                        self._state = st
            except Exception:
                # Swallow exceptions to avoid killing the thread; in a real
                # application you might want to log this.
                pass

            # Small sleep to avoid busy-waiting; adjust if needed
            time.sleep(0.001)

    def stop(self):
        """
        Optional: stop the background reader thread.

        Not strictly required if the process is exiting, since the thread
        is daemonized, but it is nice to have for explicit cleanup.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _apply_deadband(self, v: np.ndarray) -> np.ndarray:
        """
        Zero out small values to avoid jitter around zero.
        """
        v2 = v.copy()
        v2[np.abs(v2) < self.deadband] = 0.0
        return v2

    def input2action(self, goal_update_mode: str = "target") -> Optional[np.ndarray]:
        """
        Convert the latest SpaceMouse state into an action for the env.

        Returns:
          - np.ndarray action for env.step()
          - None if the reset button was pressed (edge-triggered)

        This function is intended to be called from the main control loop.
        It does *not* call pyspacemouse.read() directly and therefore
        does not block on HID I/O.
        """
        # Grab a snapshot of the latest state without blocking the reader
        with self._lock:
            st = self._state

        if st is None:
            # No data yet; output "do nothing" action
            return np.zeros((self.action_dim,), dtype=np.float32)

        # Typical fields in pyspacemouse state are x,y,z, roll,pitch,yaw, and buttons[]
        tx = float(_getattr_or_key(st, "y", 0.0))
        ty = -float(_getattr_or_key(st, "x", 0.0))
        tz = float(_getattr_or_key(st, "z", 0.0))

        # Only yaw control; roll and pitch are set to zero
        rr, rp = 0.0, 0.0
        ry = -float(_getattr_or_key(st, "yaw", 0.0))

        buttons = _getattr_or_key(st, "buttons", None)
        if buttons is None:
            buttons = _getattr_or_key(st, "button", [])
        buttons = list(buttons) if buttons is not None else []

        # Edge detection for buttons is done in the control thread
        prev = self._prev_buttons
        self._prev_buttons = buttons

        def pressed_edge(i: int) -> bool:
            """
            Returns True if button i transitioned from "not pressed"
            to "pressed" since the last call to input2action().
            """
            if i < 0:
                return False
            cur = (i < len(buttons)) and bool(buttons[i])
            prv = (prev is not None) and (i < len(prev)) and bool(prev[i])
            return cur and (not prv)

        # Reset signal
        if pressed_edge(self.reset_button_index):
            return None

        # Toggle gripper open/close
        if pressed_edge(self.gripper_button_index):
            self._gripper_closed = not self._gripper_closed

        # Normalize raw SpaceMouse values to approximately [-1, 1]
        dpos = np.array([tx, ty, tz], dtype=np.float32) / self.max_raw
        drot = np.array([rr, rp, ry], dtype=np.float32) / self.max_raw

        dpos = self._apply_deadband(dpos) * self.pos_sensitivity
        drot = self._apply_deadband(drot) * self.rot_sensitivity

        # Robosuite OSC pose actions are commonly [dx, dy, dz, dRx, dRy, dRz, gripper]
        grip = -1.0 if self._gripper_closed else 1.0

        act7 = np.concatenate([dpos, drot, np.array([grip], dtype=np.float32)], axis=0)

        # Fit to env action dimension safely
        if act7.size < self.action_dim:
            act = np.pad(act7, (0, self.action_dim - act7.size), mode="constant")
        else:
            act = act7[: self.action_dim]

        return act.astype(np.float32)
