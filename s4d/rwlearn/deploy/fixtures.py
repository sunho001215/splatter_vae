"""CPU FIXTURES for deployment tests (science and networking): tiny randomly initialised checkpoints of every FQL
condition, NEVER trained performance evidence. Frozen-encoder conditions use their real encoder adapters (dynamo,
snerl and mvp3d build the pinned upstream code: the checkout needs its third_party submodules; reviwo, sincro, e2e_cnn
and ours do not). ``ours`` has no trained policy and no fixed observation protocol: its fixture is an injected
single-view RGB sequence encoder, labelled FIXTURE, that makes no claim about the deferred method.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from s4d.rwlearn import actions as A
from s4d.rwlearn.deploy.contracts import NATIVE_DEPTH, SIDECAR_FORMAT, InputSpec
from s4d.rwlearn.deploy.history import Frame, HistoryWindow, ObservationWindow
from s4d.rwlearn.deploy.policy import DeployPolicy
from s4d.rwlearn.deploy.realsense import native_correction

FIXTURE_CONDITIONS = ("dynamo", "reviwo", "sincro", "snerl", "mvp3d_single", "mvp3d_multiview", "e2e_cnn", "ours")
NO_SUBMODULES = ("reviwo", "sincro", "e2e_cnn", "ours")
FIXTURE_CAMERAS = ("cam_a", "cam_b", "cam_c")
FIXTURE_NATIVE_HW = (128, 160)
FIXTURE_K = [[160.0, 0.0, 80.5], [0.0, 160.0, 64.5], [0.0, 0.0, 1.0]]


class FixtureSequence(nn.Module):
    """FIXTURE ONLY: ours has no trained policy or fixed observation protocol."""

    def forward(self, frames):
        return frames.mean((-1, -2)).flatten(1)


def _export(condition: str, export: Path, hw: int):
    if condition == "dynamo":
        from s4d.baselines.dynamo.model import build_encoder, save_export

        kw = {"pretrained": False, "output_dim": 8, "unit_norm": False}
        save_export(export, build_encoder(kw).eval(), kw, 1, hw, 0, True)
    elif condition == "reviwo":
        from s4d.baselines.reviwo.model import build_model, save_export

        small = {"n_layer": 1, "n_head": 2, "n_embed": 16, "dropout": 0.0}
        cb = {"n_embed": 8, "embed_dim": 4}
        cfg = {"patch_size": 8, "view_encoder": small, "latent_encoder": small, "decoder": small,
               "view_codebook": cb, "latent_codebook": cb}
        save_export(export, build_model(cfg, hw).eval(), cfg, hw, 0)
    elif condition == "sincro":
        from s4d.baselines.sincro.encoder import build_encoder, save_export

        cfg = {"img_size": hw, "patch_size": 8, "embed_dim": 16, "vit_depth": 1, "vit_num_heads": 2,
               "vit_mlp_dim": 32, "time_interval": 3, "decoder_depth": 1, "decoder_num_heads": 2,
               "decoder_mlp_dim": 32, "decoder_output_dim": 8, "num_views": 6, "num_ref_views": 2}
        save_export(export, build_encoder(cfg, "cpu").eval(), cfg, 1, 0)
    elif condition == "snerl":
        from s4d.baselines.snerl.model import build_encoder, save_export

        save_export(export, build_encoder().eval(), 0, 3, lookat=(0.3, 0, 0.3), scale=3.79, random_init=True)
    elif condition.startswith("mvp3d"):
        from s4d.baselines.mvp3d.model import MVP3DConfig, build_encoder, save_export

        cfg = MVP3DConfig(img_size=24, img_patch_size=6, im_channels=16, attn_dim=32, attn_heads=2,
                          attn_dim_head=16, depth=4, decoder_dim=32, decoder_heads=2, decoder_dim_head=16)
        save_export(export, build_encoder(cfg).eval(), asdict(cfg),
                    {"bounds": [-1, -1, -1, 1, 1, 2], "radius": 0.012}, 0, random_init=True)


def make_fixture_policy(root, condition: str, *, device="cpu", normalize_q_loss: bool = False) -> DeployPolicy:
    """A tiny random checkpoint of ``condition`` under ``root`` (created), loaded through the real ``DeployPolicy``
    path (``ours``: an injected labelled fixture through ``DeployPolicy.from_fixture``). Deterministic (seed 4)."""
    from s4d.rwlearn.fql import FQLAgent
    from s4d.rwlearn.fql_data import ObsEncoder, encoder_adapter, sha256_file
    from s4d.rwlearn.inference import OfflinePolicy

    if condition not in FIXTURE_CONDITIONS:
        raise ValueError(f"unknown fixture condition {condition!r}")
    path = Path(root)
    path.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(4)
    hw = 128 if condition == "snerl" else 32
    export = path / "encoder.pt"
    sequence, feat_dim = condition in ("sincro", "ours"), None
    if condition not in ("e2e_cnn", "ours"):
        _export(condition, export, hw)
        adapter = encoder_adapter(condition, export, hw, feature_dim=8)
        feat_dim = int(np.prod(adapter.replay_atom_shape))
    contract = A.ActionContract(0.07, 0.499, 1 / 15, 0.1, lead_m=0.04, lead_rot_deg=20.0).to_json()
    training = {"format": SIDECAR_FORMAT, "condition": condition, "camera_orders": [list(FIXTURE_CAMERAS)],
                "history_length": 3, "frame_spacing": 1, "training_depth": "foundation_stereo_depth",
                "depth_source_evidence": "FIXTURE ONLY", "export_sha256": sha256_file(export) if export.exists() else "",
                "contract": contract, "image_hw": [hw, hw],
                "fixture_protocol": {"views": 1, "depth": False, "cameras": False}}
    meta = {"condition": condition, "fixture": True, "contract": contract, "image_hw": [hw, hw],
            "observation_contract": training, "export_sha256": training["export_sha256"],
            "export_path": str(export), "feat_dim": feat_dim, "proj_dim": 8, "sequence_feature": sequence}
    if condition == "e2e_cnn":
        from s4d.rwlearn.pixel import ImpalaEncoder

        def enc():
            return ImpalaEncoder(9, (hw, hw), stack_sizes=(4, 4), mlp_hidden_dims=(8,), proprio_dim=10)

        meta["encoder_config"] = enc().config
    else:
        feat_dim = 9 if condition == "ours" else feat_dim

        def enc():
            return ObsEncoder(feat_dim, sequence, proj=8)

    agent = FQLAgent.create(1, enc().out_dim, 7, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,),
                                                   "normalize_q_loss": bool(normalize_q_loss)}, encoder=enc)
    if condition == "ours":
        offline = OfflinePolicy.__new__(OfflinePolicy)
        offline.meta, offline.condition, offline.device = meta, condition, torch.device(device)
        offline.contract, offline.agent, offline.sequence = A.ActionContract.from_json(contract), agent, True
        offline.adapter = SimpleNamespace(backbone=FixtureSequence(), observation=None)
        agent.train_mode(False)
        return DeployPolicy.from_fixture(offline, InputSpec.from_meta(meta, training))
    checkpoint = path / "policy.pt"
    torch.save({"meta": meta, "agent": agent.state_dict()}, checkpoint)
    return DeployPolicy(checkpoint, device=device, export_path=export if export.exists() else None)


def session_choices(policy: DeployPolicy, session_id: str = "fixture") -> dict:
    """``reset_session`` kwargs for a fixture policy: first trained camera (single view) or all, native 128 x 160
    grid, identity calibration K/c2w where needed, explicit identity native depth correction where needed."""
    spec = policy.input_spec
    cams = spec.trained_camera_orders[0] if spec.views > 1 else (spec.trained_camera_orders[0][0],)
    out = {"session_id": session_id, "native_hw": dict.fromkeys(cams, FIXTURE_NATIVE_HW)}
    if spec.views == 1:
        out["camera_id"] = cams[0]
    if spec.needs_calibration:
        out["calibration"] = {c: {"K_native": FIXTURE_K, "c2w": np.eye(4).tolist()} for c in cams}
    if "depth" in spec.modalities:
        out["native_depth_correction"] = {c: native_correction() for c in cams}
    return out


def synthetic_frames(spec: InputSpec, count: int = 3, *, seed: int = 12, start_seq: int = 0,
                     start_ns: int = 0) -> dict[str, list[Frame]]:
    """``count`` consecutive preprocessed FIXTURE frames per session camera (15 Hz stamps, cameras 1 ms apart)."""
    rng = np.random.default_rng(seed)
    out = {}
    for v, c in enumerate(spec.camera_ids):
        out[c] = []
        for k in range(count):
            seq = start_seq + k
            rgb = rng.integers(0, 256, (*spec.image_hw, 3), dtype=np.uint8)
            d = rng.uniform(0.3, 1, spec.image_hw).astype(np.float32) if "depth" in spec.modalities else None
            if d is not None:
                d[::3, ::3] = 0
            metadata = {} if d is None else {"depth_source": NATIVE_DEPTH,
                                             "native_correction": spec.native_depth_correction[c]}
            t = start_ns + int(round(seq * spec.frame_period_s * 1e9)) + v * 1_000_000
            out[c].append(Frame(seq, t, rgb, d, None if d is None else d > 0, metadata))
    return out


def synthetic_window(spec: InputSpec, *, seed: int = 12) -> ObservationWindow:
    """A complete 3-step FIXTURE history (seq 0..2 per camera; multi-view M5 evidence 1 mm per step)."""
    history = HistoryWindow(spec)
    for c, frames in synthetic_frames(spec, spec.history_length, seed=seed).items():
        for f in frames:
            history.push(c, f)
    return history.assemble(dict.fromkeys(spec.camera_ids, spec.history_length - 1),
                            surface_disp_m=[0.001] * spec.history_length if spec.views > 1 else None)
