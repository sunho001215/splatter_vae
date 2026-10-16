"""Build the official DynaMo modules from a ``configs/baselines/dynamo/<task>.yaml`` config, and the encoder export.

``build_modules`` follows ``train.py::Trainer.__init__`` (lines 56-61, 92-135): the encoder (``resnet18``,
optionally SyncBatchNorm, train.py:94-98) and its AdamW (train.py:99-104), then the inverse-dynamics projector and
its ``configure_optimizers`` AdamW (train.py:114-123), then ``DynaMoSSL`` (train.py:131-135), which builds the
forward-dynamics transformer and its own AdamW (dynamo.py:53-67). Construction happens on the CPU in exactly this
order, so a given seed gives the same initial weights as the official trainer; the modules are then moved to the
device, which is what ``accelerator.prepare`` did (train.py:105-108, 124-127; dynamo.py:68-71, removed by
``third_party/patches/dynamo/0001``). Hydra interpolations of the official configs are resolved the same way:
``window_size`` from the dataset section, ``feature_dim``/``input_dim`` = encoder ``output_dim``,
``projection_dim`` = projector ``output_dim``, and the SSL ``lr``/``weight_decay``/``betas`` from the train section.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from s4d.baselines.dynamo import official

EXPORT_FORMAT = "splatter4d-baseline-dynamo-v1"


def encoder_kwargs(cfg: dict) -> dict:
    """``configs/encoder/resnet18_random.yaml`` arguments."""
    enc = cfg["encoder"]
    return {"pretrained": bool(enc["pretrained"]), "output_dim": int(enc["output_dim"]), "unit_norm": bool(enc["unit_norm"])}


def projector_kwargs(cfg: dict) -> dict:
    """``configs/projector/inverse_dynamics_*.yaml`` with its interpolations resolved."""
    proj = cfg["projector"]
    return {
        "window_size": int(cfg["dataset"]["window_size"]),
        "input_dim": int(cfg["encoder"]["output_dim"]),
        "n_layer": int(proj["n_layer"]),
        "n_head": int(proj["n_head"]),
        "n_embd": int(proj["n_embd"]),
        "output_dim": int(proj["output_dim"]),
        "dropout": float(proj["dropout"]),
    }


def ssl_kwargs(cfg: dict) -> dict:
    """``configs/ssl/dynamo_*.yaml`` with its interpolations resolved (without encoder/projector)."""
    ssl, train = cfg["ssl"], cfg["train"]
    return {
        "window_size": int(cfg["dataset"]["window_size"]),
        "feature_dim": int(cfg["encoder"]["output_dim"]),
        "projection_dim": int(cfg["projector"]["output_dim"]),
        "n_layer": int(ssl["n_layer"]),
        "n_head": int(ssl["n_head"]),
        "n_embd": int(ssl["n_embd"]),
        "dropout": float(ssl["dropout"]),
        "covariance_reg_coef": float(ssl["covariance_reg_coef"]),
        "dynamics_loss_coef": float(ssl["dynamics_loss_coef"]),
        "ema_beta": None if ssl["ema_beta"] is None else float(ssl["ema_beta"]),
        "beta_scheduling": bool(ssl["beta_scheduling"]),
        "projector_use_ema": bool(ssl["projector_use_ema"]),
        "lr": float(train["ssl_lr"]),
        "weight_decay": float(train["ssl_weight_decay"]),
        "betas": tuple(float(b) for b in train["betas"]),
        "separate_single_views": bool(ssl["separate_single_views"]),
    }


def build_encoder(kwargs: dict) -> nn.Module:
    """The official ``resnet18`` (torchvision trunk without ``fc``, ImageNet normalisation inside)."""
    with warnings.catch_warnings():  # torchvision's deprecation notice for the upstream ``pretrained=`` argument
        warnings.simplefilter("ignore", UserWarning)
        return official.resnet18()(**kwargs)


@dataclass
class DynaMoModules:
    encoder: nn.Module
    projector: nn.Module
    ssl: nn.Module
    encoder_optim: torch.optim.Optimizer
    projector_optim: torch.optim.Optimizer

    @property
    def forward_dynamics_optim(self) -> torch.optim.Optimizer:
        return self.ssl.forward_dynamics_optimizer

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "ssl": self.ssl.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "projector_optim": self.projector_optim.state_dict(),
            "forward_dynamics_optim": self.forward_dynamics_optim.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.projector.load_state_dict(state["projector"], strict=True)
        self.ssl.load_state_dict(state["ssl"], strict=True)
        self.encoder_optim.load_state_dict(state["encoder_optim"])
        self.projector_optim.load_state_dict(state["projector_optim"])
        self.forward_dynamics_optim.load_state_dict(state["forward_dynamics_optim"])


def build_modules(cfg: dict, device: torch.device | str) -> DynaMoModules:
    """Official construction order on the CPU (see the module docstring), then everything on ``device``."""
    train = cfg["train"]
    lr, wd, betas = float(train["ssl_lr"]), float(train["ssl_weight_decay"]), tuple(float(b) for b in train["betas"])
    encoder = build_encoder(encoder_kwargs(cfg))
    if bool(train["sync_bn"]):  # train.py:95-98; one process: SyncBatchNorm computes plain batch norm
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder_optim = torch.optim.AdamW(params=encoder.parameters(), lr=lr, weight_decay=wd, betas=betas)
    projector = official.inverse_dynamics_projector()(**projector_kwargs(cfg))
    projector_optim = projector.configure_optimizers(lr=lr, weight_decay=wd, betas=betas)
    ssl = official.dynamo_ssl()(encoder=encoder, projector=projector, **ssl_kwargs(cfg))
    # Module.to keeps the Parameter objects (it swaps their data), so the optimizers above stay attached.
    encoder.to(device)
    projector.to(device)
    ssl.to(device)
    return DynaMoModules(encoder, projector, ssl, encoder_optim, projector_optim)


def save_export(
    path: Path, encoder: nn.Module, kwargs: dict, frame_spacing: int, image_size: int, step: int, random_init: bool
) -> None:
    """Encoder-only export for the RL side: weights (BatchNorm running statistics included), the official
    constructor arguments, the pretraining frame spacing and whether these are the untrained initial weights."""
    payload = {
        "format": EXPORT_FORMAT,
        "encoder_kwargs": dict(kwargs),
        "frame_spacing": int(frame_spacing),
        "image_size": int(image_size),
        "state_dict": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
        "step": int(step),
        "random_init": bool(random_init),
    }
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_export(path: str | Path) -> tuple[dict, nn.Module]:
    """Rebuild the official encoder from an export, frozen in eval mode (BatchNorm uses its running statistics).
    The global RNG is left untouched by the throw-away random initialisation."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a DynaMo encoder export")
    with torch.random.fork_rng(devices=[]):
        encoder = build_encoder(payload["encoder_kwargs"])
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return payload, encoder.eval().requires_grad_(False)
