"""RVT modules (github.com/NVlabs/RVT @ 367995a1, NVIDIA Source Code License, non-commercial) imported from the
pinned submodule ``third_party/rvt``.

Two RVT files are used, unmodified in function:

- ``rvt/mvt/attn.py``: the MVT layer primitives (``PreNorm``, ``Attention``, ``FeedForward``/``GEGLU``,
  ``Conv2DBlock``, ``DenseBlock``, ``cache_fn``) from which our encoder and MAE decoder are built;
- ``rvt/mvt/utils.py``: ``place_pc_in_cube``. Its unused top-level ``import matplotlib.pyplot`` is removed by
  ``third_party/patches/rvt/0001-mvt-utils-drop-unused-matplotlib-import.patch`` (matplotlib is not in our stack).

The files are loaded by path from ``s4d.third_party.source_tree("rvt")`` under unique module names, so no RVT package
(``rvt``, ``mvt``, ``utils``, ...) is ever put on ``sys.path``. ``attn.py`` has no RVT-internal imports; its
``xformers`` import is optional upstream and unused here (``use_fast=False``).
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

ATTN_FILE = "rvt/mvt/attn.py"
UTILS_FILE = "rvt/mvt/utils.py"
_NAMES = {ATTN_FILE: "s4d_rvt_mvt_attn", UTILS_FILE: "s4d_rvt_mvt_utils"}


def _load(relative: str) -> ModuleType:
    name = _NAMES[relative]
    module = sys.modules.get(name)
    if module is not None:
        return module
    from s4d.third_party import source_tree  # noqa: PLC0415

    path = source_tree("rvt") / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def attn() -> ModuleType:
    """``rvt/mvt/attn.py`` of the pinned RVT commit."""
    return _load(ATTN_FILE)


def mvt_utils() -> ModuleType:
    """``rvt/mvt/utils.py`` of the pinned RVT commit (with the matplotlib import removed)."""
    return _load(UTILS_FILE)
