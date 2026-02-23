from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class PolicyInfo:
    policy_type: str   # "scripted" | "random"
    policy_name: str


class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


def _try_import_policy_module(env_name: str):
    # env_name like "reach-v3" -> "sawyer_reach_v3_policy"
    mod = f"metaworld.policies.sawyer_{env_name.replace('-', '_')}_policy"
    return importlib.import_module(mod)


def _find_concrete_policy_class_in_module(mod) -> type:
    """
    Pick a concrete (non-abstract) class defined in this module whose name ends with 'Policy',
    but is NOT the abstract base 'Policy'.
    """
    candidates = []
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if not name.endswith("Policy"):
            continue
        if name == "Policy":
            continue
        # only classes defined in THIS module (not imported base classes)
        if cls.__module__ != mod.__name__:
            continue
        if inspect.isabstract(cls):
            continue
        if not callable(getattr(cls, "get_action", None)):
            continue
        candidates.append(cls)

    if not candidates:
        # Debug info helps if metaworld changes structure
        all_policies = [n for n, c in inspect.getmembers(mod, inspect.isclass) if n.endswith("Policy")]
        raise RuntimeError(
            f"No concrete policy class found in {mod.__name__}. "
            f"Policy-like classes seen: {all_policies}"
        )

    # Heuristic: if multiple, prefer the longest name (often the task-specific one)
    candidates.sort(key=lambda c: len(c.__name__), reverse=True)
    return candidates[0]


def make_scripted_policy(env_name: str, override_class: Optional[str] = None) -> Tuple[Any, PolicyInfo]:
    """
    Returns (policy_instance, PolicyInfo).
    """
    import metaworld.policies  # noqa: F401

    if override_class is not None:
        # Search all policy modules for that class name
        import metaworld.policies as polpkg
        for m in pkgutil.iter_modules(polpkg.__path__):
            mod = importlib.import_module(f"{polpkg.__name__}.{m.name}")
            if hasattr(mod, override_class):
                cls = getattr(mod, override_class)
                if inspect.isclass(cls) and not inspect.isabstract(cls):
                    return cls(), PolicyInfo(policy_type="scripted", policy_name=override_class)
                raise RuntimeError(f"Found {override_class} but it is abstract / not a class.")
        raise RuntimeError(f"Could not find policy class '{override_class}' in metaworld.policies.*")

    mod = _try_import_policy_module(env_name)
    cls = _find_concrete_policy_class_in_module(mod)
    return cls(), PolicyInfo(policy_type="scripted", policy_name=cls.__name__)