"""FQL datasets on top of frozen representation baselines (design: docs/realworld_fql.md).

``build_fql_dataset`` runs a baseline's FROZEN encoder export once over the training store and writes, per episode
and timeline, one feature vector per observation time (per camera frame for single-view encoders, per synchronised
step for multi-view ones; for SinCro, whose encoder maps a 3-frame history to one state, the state of the history
ending at that frame) next to the transition arrays of ``s4d.rwlearn.transitions``. These are regenerable derived
vectors keyed to (store, encoder export sha256) - never images, and never copied per window: observation histories
and next observations are assembled by ``OfflineTransitions`` from compact index arrays at sampling time.

Viewpoint protocol: a single-view condition samples a (camera timeline, step) pair, so the viewpoint is drawn per
training sample and is the same for the whole observation history and for the paired next observation. Multi-view
conditions (SNeRL, 3D-MVP A) use all three cameras at every step (the rig has exactly three; SNeRL mean-pools its
views and 3D-MVP fuses them, so their order is irrelevant).

Policy input = the campaign's frozen-encoder adapter (``FrameMLPStackHead`` over the 3-frame stack, or
``SmallPostEncoderMLPHead`` for SinCro's fused state; 256-d, trainable) concatenated with the 10-d proprioception
(``actions`` module docstring); one adapter instance per FQL network, as the official agent gives every network its
own encoder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from s4d.rwlearn import actions as A
from s4d.rwlearn.store import FrameStore
from s4d.rwlearn.transitions import episode_transitions, history_indices, timelines

FORMAT = "rwlearn_fql_dataset_v1"
FRAME_STACK = 3


@dataclass(frozen=True)
class Condition:
    encoder_type: str
    views: int
    depth: bool
    cameras: bool
    label: str

    @property
    def multiview(self) -> bool:
        return self.views > 1


CONDITIONS = {
    "dynamo": Condition("dynamo", 1, False, False, "DynaMo, single RGB"),
    "reviwo": Condition("reviwo", 1, False, False, "ReViWo, single RGB"),
    "sincro": Condition("sincro", 1, False, False, "SinCro, single RGB"),
    "snerl": Condition("snerl", 3, False, True, "SNeRL, multi-view RGB + poses (privileged)"),
    "mvp3d_multiview": Condition("mvp3d", 3, True, True, "3D-MVP A, multi-view RGB-D + extrinsics (privileged)"),
    "mvp3d_single": Condition("mvp3d", 1, True, True, "3D-MVP B, single RGB-D + extrinsics (privileged)"),
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def encoder_adapter(condition: str, export_path, image_size: int = 128, feature_dim: int = 256):
    """The campaign's ``VisionEncoderAdapter`` for a frozen baseline export (backbone frozen, eval mode)."""
    from s4d.rl.encoders import VisionEncoderAdapter

    c = CONDITIONS[condition]
    cfg = {"vision": {"encoder_type": c.encoder_type, "export_path": str(export_path)},
           "env": {"frame_stack": FRAME_STACK, "image_size": image_size, "views": c.views, "depth": c.depth,
                   "cameras": c.cameras},
           "agent": {"feature_dim": feature_dim}}
    return VisionEncoderAdapter(cfg)


@torch.no_grad()
def encode_timeline(adapter, store: FrameStore, ep: str, tl, *, device, chunk: int = 64) -> np.ndarray:
    """(K, D) float16 frozen features of one timeline (see module docstring)."""
    backbone = adapter.backbone
    sequence = adapter.stack_feature
    K = len(tl.times)
    out = []
    for s0 in range(0, K, chunk):
        ks = np.arange(s0, min(K, s0 + chunk))
        if sequence:  # SinCro: state of the 3-frame history ending at k (single camera)
            hist = history_indices(tl.run, ks, FRAME_STACK)
            rgb = store.rgb(ep, tl.cameras[0], tl.frames[hist.reshape(-1), 0])
            x = torch.from_numpy(rgb).to(device).float().div(255).permute(0, 3, 1, 2)
            feat = backbone(x.reshape(len(ks), FRAME_STACK, *x.shape[1:]))
        elif adapter.observation is not None:
            views = range(len(tl.cameras))
            rgb = np.stack([store.rgb(ep, tl.cameras[v], tl.frames[ks, v]) for v in views], 1)
            obs = {"rgb": torch.from_numpy(rgb).to(device).float().div(255).permute(0, 1, 4, 2, 3)}
            if adapter.observation.get("depth"):
                obs["depth"] = torch.from_numpy(np.stack([store.depth(ep, tl.cameras[v], tl.frames[ks, v])
                                                          for v in views], 1)).to(device).float()
            if adapter.observation.get("cameras"):
                K_ = np.stack([store.intrinsics(ep, c) for c in tl.cameras])
                c2w = np.stack([np.linalg.inv(store.w2c(ep, c)) for c in tl.cameras])
                obs["K"] = torch.from_numpy(K_).float().to(device).expand(len(ks), -1, -1, -1).contiguous()
                obs["c2w"] = torch.from_numpy(c2w).float().to(device).expand(len(ks), -1, -1, -1).contiguous()
            feat = backbone(obs)
        else:
            rgb = store.rgb(ep, tl.cameras[0], tl.frames[ks, 0])
            feat = backbone(torch.from_numpy(rgb).to(device).float().div(255).permute(0, 3, 1, 2))
        out.append(feat.flatten(1).to(torch.float16).cpu().numpy())
    return np.concatenate(out)


def build_fql_dataset(store: FrameStore, condition: str, export_path, out: Path, *, episodes: list[str],
                      reward: str = "ogbench_final_success", device="cpu", max_sync_ns: int = 30_000_000,
                      raw_paths: dict | None = None, outcomes: dict | None = None,
                      assume_final_success: bool = False) -> dict:
    """Write ``out`` (atomic). ``raw_paths`` optionally maps episode -> raw recording path (else the store's).
    Outcomes are explicit: ``outcomes[ep]`` (True = successful demonstration) for every episode, or
    ``assume_final_success`` (every recording is a successful demonstration; the files carry no outcome marker)."""
    missing = [ep for ep in episodes if ep not in (outcomes or {})]
    if missing and not assume_final_success:
        raise ValueError(f"no outcome label for {len(missing)} episode(s) (e.g. {missing[0]}); give outcomes or "
                         "assume_final_success=True explicitly")
    c = CONDITIONS[condition]
    adapter = encoder_adapter(condition, export_path, store.image_hw[0]).to(device)
    adapter.backbone.eval()
    out = Path(out)
    tmp = out.with_name(f".{out.name}.partial")
    contract, stats = None, {"transitions": 0, "reasons": {}, "labels": {}}
    with h5py.File(tmp, "w") as f:
        for ep in episodes:
            raw = A.load_raw((raw_paths or {}).get(ep) or store.raw_recording(ep))
            A.check_contract_preconditions(raw)
            ep_contract = A.contract_for(raw)  # nominal period; each camera's measured period checked within 1 %
            if contract is None:
                contract = ep_contract
            elif ep_contract != contract:
                raise ValueError(f"{ep}: teleop limits differ from the dataset contract")
            for tl in timelines(store, ep, c.multiview, max_sync_ns):
                tr = episode_transitions(raw, tl, contract, reward, success=bool((outcomes or {}).get(ep, True)))
                g = f.create_group(f"episodes/{ep}/{tl.name}")
                g.create_dataset("features", data=encode_timeline(adapter, store, ep, tl, device=device),
                                 compression="lzf")
                for k in ("times", "run", "proprio", "state_ok", "transition_ok", "action", "action_phys", "reward",
                          "mask", "success", "terminal"):
                    g.create_dataset(k, data=tr[k])
                g.create_dataset("measured_pose", data=tr["measured_pose"].astype(np.float64))
                g.create_dataset("frames", data=tl.frames.astype(np.int32))
                g.attrs.update(cameras=json.dumps(tl.cameras), task=store.task(ep), label=tr["label"])
                stats["labels"][tr["label"]] = stats["labels"].get(tr["label"], 0) + 1
                stats["transitions"] += int(tr["transition_ok"].sum())
                for r, n in tr["reasons"].items():
                    stats["reasons"][r] = stats["reasons"].get(r, 0) + int(n)
        f.attrs.update(format=FORMAT, condition=condition, label=c.label, encoder_type=c.encoder_type,
                       export_path=str(export_path), export_sha256=sha256_file(export_path), reward=reward,
                       contract=json.dumps(contract.to_json()), frame_stack=FRAME_STACK,
                       sequence_feature=bool(adapter.stack_feature), episodes=json.dumps(list(episodes)),
                       image_hw=json.dumps(list(store.image_hw)), stats=json.dumps(stats),
                       outcomes=json.dumps({"assume_final_success": bool(assume_final_success),
                                            "labels": {ep: bool(v) for ep, v in (outcomes or {}).items()}}))
    tmp.replace(out)
    return {"path": str(out), **stats}


class ObsEncoder(nn.Module):
    """Per-network FQL observation encoder: campaign adapter head on the frozen features + raw proprioception."""

    def __init__(self, feat_dim: int, sequence: bool, proprio_dim: int = A.PROPRIO_DIM, proj: int = 256):
        super().__init__()
        from s4d.rl.encoders import FrameMLPStackHead, SmallPostEncoderMLPHead

        self.sequence = bool(sequence)
        self.head = (SmallPostEncoderMLPHead(feat_dim, proj, proj) if sequence
                     else FrameMLPStackHead(feat_dim, FRAME_STACK, proj, min(proj, 256), proj, proj))
        self.out_dim = proj + proprio_dim

    def forward(self, obs: dict) -> torch.Tensor:
        return torch.cat([self.head(obs["feat"].float()), obs["proprio"].float()], dim=-1)


class OfflineTransitions:
    """All transitions of ``episodes`` in memory: frozen features once per observation time + index arrays."""

    def __init__(self, path, episodes: list[str] | None = None, device="cpu"):
        self.path = str(path)
        feats, prop, rows = [], [], []
        base = 0
        with h5py.File(self.path, "r") as f:
            if f.attrs.get("format") != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file")
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            self.sequence = bool(self.attrs["sequence_feature"])
            self.contract = A.ActionContract.from_json(json.loads(self.attrs["contract"]))
            eps = episodes if episodes is not None else json.loads(self.attrs["episodes"])
            self.timelines = []
            for ep in eps:
                for name, g in f["episodes"][ep].items():
                    feat = g["features"][:]
                    run = g["run"][:]
                    ok = g["transition_ok"][:].astype(bool)
                    ks = np.flatnonzero(ok)
                    if self.sequence:
                        obs = (ks + base)[:, None]
                        nxt = (ks + 1 + base)[:, None]
                    else:
                        obs = history_indices(run, ks, FRAME_STACK) + base
                        nxt = history_indices(run, ks + 1, FRAME_STACK) + base
                    rows.append({"obs": obs, "next": nxt, "action": g["action"][:][ks], "reward": g["reward"][:][ks],
                                 "mask": g["mask"][:][ks], "obs_k": ks + base, "next_k": ks + 1 + base})
                    feats.append(feat)
                    prop.append(g["proprio"][:])
                    self.timelines.append((ep, name, base, len(feat)))
                    base += len(feat)
        if not rows:
            raise ValueError("no transitions")
        cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        dev = torch.device(device)
        self.features = torch.from_numpy(np.concatenate(feats)).to(dev)  # fp16
        self.proprio = torch.from_numpy(np.concatenate(prop)).to(dev)
        self.obs_idx = torch.from_numpy(cat["obs"]).to(dev)
        self.next_idx = torch.from_numpy(cat["next"]).to(dev)
        self.obs_k = torch.from_numpy(cat["obs_k"]).to(dev)
        self.next_k = torch.from_numpy(cat["next_k"]).to(dev)
        self.actions = torch.from_numpy(cat["action"]).to(dev).float()
        self.rewards = torch.from_numpy(cat["reward"]).to(dev).float()
        self.masks = torch.from_numpy(cat["mask"]).to(dev).float()
        self.size = len(self.actions)
        self.feat_dim = int(self.features.shape[-1])
        self.device = dev

    def _obs(self, hist, k):
        feat = self.features[hist]
        return {"feat": feat[:, 0] if self.sequence else feat, "proprio": self.proprio[k]}

    def batch(self, idx: torch.Tensor) -> dict:
        idx = idx.to(self.device)
        return {"observations": self._obs(self.obs_idx[idx], self.obs_k[idx]), "actions": self.actions[idx],
                "rewards": self.rewards[idx], "masks": self.masks[idx],
                "next_observations": self._obs(self.next_idx[idx], self.next_k[idx])}

    def sample(self, batch_size: int, generator: torch.Generator) -> dict:
        """Uniform over all (timeline, step) transitions: the viewpoint is drawn per sample (official
        ``Dataset.sample``: uniform indices)."""
        return self.batch(torch.randint(self.size, (batch_size,), generator=generator))

    def encoder_factory(self, proj: int = 256):
        return lambda: ObsEncoder(self.feat_dim, self.sequence, A.PROPRIO_DIM, proj)
