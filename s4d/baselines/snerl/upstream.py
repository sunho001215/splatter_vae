"""The official SNeRL and DINO code, loaded from the pinned submodules; nothing here re-implements upstream code.

- SNeRL (``third_party/snerl``, jayLEE0301/snerl_official @ d01533b, MIT): ``nerf_pretrain/{run_nerf_helpers,
  multiview_encoder, dino, run_nerf}.py`` from the patched copy built by ``s4d.third_party.source_tree("snerl")``
  (``third_party/patches/snerl/0001-*.patch`` only drops unused top-level imports of packages we do not install,
  keeping every line number). The modules are registered under unique names (``s4d_snerl_<name>``); their plain-name
  imports of each other (``from run_nerf_helpers import *``) resolve to these copies only while they load.
- DINO (``third_party/dino``, facebookresearch/dino @ 7c446df, Apache-2.0): the ViT-S/8 teacher through
  ``torch.hub.load(<checkout>, "dino_vits8", source="local")`` (no code download) with the official pretrained weights
  from a local file whose sha256 is checked, loaded exactly as ``hubconf.dino_vits8(pretrained=True)`` does.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from s4d.third_party import source_tree

REPO = Path(__file__).resolve().parents[3]
NERF_MODULES = ("run_nerf_helpers", "multiview_encoder", "dino", "run_nerf")  # dependency order
DINO_HUB_MODULES = ("hubconf", "vision_transformer", "utils")  # plain names the DINO hub code imports
DINO_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
DINO_WEIGHTS_SHA256 = "55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e"
DINO_WEIGHTS_FILE = "dino_deitsmall8_pretrain.pth"

_MODULES: dict[str, ModuleType] = {}


@contextlib.contextmanager
def _plain_module_names(names):
    """Hide existing ``sys.modules`` entries under ``names`` and drop whatever the block binds there afterwards. No
    bytecode is written meanwhile: a ``__pycache__`` inside a pinned checkout would make it dirty (``python -I``
    ignores ``PYTHONDONTWRITEBYTECODE``)."""
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    no_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        yield
    finally:
        sys.dont_write_bytecode = no_bytecode
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def official(*names: str) -> dict[str, ModuleType]:
    """The official ``nerf_pretrain`` modules ``names`` (default: all four) and the ones they import.

    Importing ``run_nerf`` runs its module-level ``np.random.seed(0)`` (run_nerf.py:21); the caller's NumPy global RNG
    state is restored afterwards so loading the code has no side effect on the RNG of the importing process."""
    wanted = names or NERF_MODULES
    needed = NERF_MODULES[: max(NERF_MODULES.index(n) for n in wanted) + 1]
    if any(name not in _MODULES for name in needed):
        root = source_tree("snerl") / "nerf_pretrain"
        rng_state = np.random.get_state()
        with _plain_module_names(NERF_MODULES):
            sys.modules.update(_MODULES)
            for name in needed:
                if name in _MODULES:
                    continue
                spec = importlib.util.spec_from_file_location(f"s4d_snerl_{name}", root / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(spec.name, None)
                    raise
                _MODULES[name] = module
        np.random.set_state(rng_state)
    return {name: _MODULES[name] for name in wanted}


@contextlib.contextmanager
def default_tensor_type(device: torch.device | str):
    """The official entry point sets ``torch.set_default_tensor_type('torch.cuda.FloatTensor')`` (run_nerf.py:890):
    its rendering code builds tensors without a device, partly with legacy constructors (``torch.Tensor([1e10])`` in
    ``raw2outputs``) that ``torch.set_default_device`` does not reach. The same setting is applied around the official
    calls only, and restored afterwards. A no-op on CPU."""
    if torch.device(device).type != "cuda":
        yield
        return
    previous = torch.Tensor().type()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"torch\.set_default_tensor_type\(\) is deprecated")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        yield
    finally:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"torch\.set_default_tensor_type\(\) is deprecated")
            torch.set_default_tensor_type(previous)


def dino_weights_path() -> Path:
    """``<cache>/torch/hub/checkpoints/dino_deitsmall8_pretrain.pth``: where ``torch.hub`` caches the official file under
    the repository's ``TORCH_HOME`` (``scripts/_bootstrap.py``; cache root ``S4D_CACHE_ROOT`` or ``<repo>/.cache``)."""
    root = Path(os.environ.get("S4D_CACHE_ROOT", str(REPO / ".cache")))
    return root / "torch" / "hub" / "checkpoints" / DINO_WEIGHTS_FILE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dino_vits8(weights: str | Path | None = None, pretrained: bool = True) -> torch.nn.Module:
    """The DINO ViT-S/8 of ``hubconf.dino_vits8`` from the pinned checkout. ``pretrained`` loads the official weights
    from ``weights`` (default ``dino_weights_path()``) after checking their sha256; nothing is downloaded."""
    tree = source_tree("dino")
    with _plain_module_names(DINO_HUB_MODULES):
        model = torch.hub.load(str(tree), "dino_vits8", source="local", pretrained=False)
    if pretrained:
        path = Path(weights) if weights is not None else dino_weights_path()
        if not path.is_file():
            raise FileNotFoundError(f"DINO ViT-S/8 weights missing: download {DINO_WEIGHTS_URL} to {path}")
        if (digest := sha256_file(path)) != DINO_WEIGHTS_SHA256:
            raise ValueError(f"{path} has sha256 {digest}, expected the official {DINO_WEIGHTS_SHA256}")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)  # as hubconf.dino_vits8(pretrained=True)
    return model


def dino_extractor(device: torch.device | str, weights: str | Path | None = None, pretrained: bool = True):
    """The official teacher ``ViTExtractor('dino_vits8', 4)`` (run_nerf.py:662) around the local hub model; its own
    ``create_model`` would fetch ``facebookresearch/dino:main`` from GitHub, so the model is passed in. Building the
    ViT (initialised before the weights are loaded) never draws from the caller's RNG, so a resumed run continues
    the checkpointed random stream; without ``pretrained`` the random teacher is seeded (0)."""
    vit_extractor = official("dino")["dino"].ViTExtractor
    with torch.random.fork_rng(devices=[]):
        # Only the CPU generator is seeded: torch.manual_seed also resets CUDA, including a restored checkpoint RNG.
        torch.random.default_generator.manual_seed(0)
        vit = load_dino_vits8(weights, pretrained)
    return vit_extractor("dino_vits8", 4, model=vit, device=str(torch.device(device)))
