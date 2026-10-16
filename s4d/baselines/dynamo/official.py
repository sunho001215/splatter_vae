"""The official DynaMo ``models`` package (jeffacce/dynamo_ssl @ 6287403, MIT), imported under a unique name.

The source is ``s4d.third_party.source_tree("dynamo")``: the pinned submodule with the patch files of
``third_party/patches/dynamo/`` applied to a cached copy (only ``0001``: DynaMoSSL without accelerate). The upstream
package is called ``models``, which would clash with other code, so it is loaded as ``s4d_third_party_dynamo.models``
straight from its directory; nothing is put on ``sys.path`` and its relative imports resolve inside the copy.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from functools import cache
from pathlib import Path

PACKAGE = "s4d_third_party_dynamo"


@cache
def tree() -> Path:
    from s4d.third_party import source_tree

    return source_tree("dynamo")


def _package() -> str:
    if PACKAGE not in sys.modules:
        root = types.ModuleType(PACKAGE)
        root.__path__ = []  # a container only: ``models`` below is loaded from the official tree by location
        sys.modules[PACKAGE] = root
    name = f"{PACKAGE}.models"
    if name not in sys.modules:
        models = tree() / "models"
        spec = importlib.util.spec_from_file_location(
            name, models / "__init__.py", submodule_search_locations=[str(models)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return name


def module(dotted: str) -> types.ModuleType:
    """An official module by its upstream path below ``models``, e.g. ``module("ssl.dynamo")``."""
    return importlib.import_module(f"{_package()}.{dotted}")


def resnet18():
    """``models/encoder/resnet.py::resnet18`` (configs/encoder/resnet18_random.yaml)."""
    return module("encoder.resnet").resnet18


def inverse_dynamics_projector():
    """``models/projector/inverse_dynamics.py::InverseDynamicsProjector``."""
    return module("projector.inverse_dynamics").InverseDynamicsProjector


def dynamo_ssl():
    """``models/ssl/dynamo.py::DynaMoSSL`` (patched: no accelerate)."""
    return module("ssl.dynamo").DynaMoSSL
