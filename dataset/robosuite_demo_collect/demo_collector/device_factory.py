from __future__ import annotations

import inspect
from typing import Any


def _construct_device(cls: type, **kwargs) -> Any:
    """
    Robust constructor helper.

    Different robosuite versions sometimes change whether `env` is a positional
    arg or a keyword arg. We introspect the signature and pass only what it accepts.
    """
    sig = inspect.signature(cls)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    # If the constructor uses *args / **kwargs we can pass everything
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        filtered = kwargs

    return cls(**filtered)


def make_teleop_device(
    device_name: str,
    env,
    *,
    pos_sensitivity: float = 1.0,
    rot_sensitivity: float = 1.0,
    **kwargs,
):
    """
    Create and initialize a teleop device.

    Args:
        device_name: 'spacemouse', 'keyboard', 'dualsense', 'mjgui', or 'spacemouse_spnav' (optional Linux).
        env: robosuite environment instance (RobotEnv).
        pos_sensitivity / rot_sensitivity: scaling factors supported by robosuite teleop devices (commonly exposed).
        kwargs: device-specific extra options.

    Returns:
        device: a started device instance (device.start_control() already called).
    """
    name = device_name.lower().strip()

    if name == "spacemouse":
        # Try robosuite SpaceMouse first (may fail on Linux depending on HID setup)
        try:
            from robosuite.devices import SpaceMouse
            device = _construct_device(
                SpaceMouse,
                env=env,
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                **kwargs,
            )
        except Exception as e:
            # Linux fallback: pyspacemouse
            from demo_collector.spacemouse_pyspacemouse import SpaceMousePySpaceMouse
            device = SpaceMousePySpaceMouse(
                env=env,
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
            )

    elif name == "keyboard":
        from robosuite.devices import Keyboard

        device = _construct_device(
            Keyboard,
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
            **kwargs,
        )

    elif name == "dualsense":
        from robosuite.devices import DualSense

        device = _construct_device(
            DualSense,
            env=env,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
            **kwargs,
        )

    elif name == "mjgui":
        from robosuite.devices import MJGUI

        device = _construct_device(MJGUI, env=env, **kwargs)

    else:
        raise ValueError(
            f"Unknown device '{device_name}'. Try: spacemouse | keyboard | dualsense | mjgui | spacemouse_spnav"
        )

    # robosuite device contract: must call start_control() before reading commands. :contentReference[oaicite:7]{index=7}
    device.start_control()
    return device
