"""Our representation evaluation (cross-view retrieval, linear probes, state statistics) on frozen baseline encoders.

The states are exactly the RL agent's: the export is loaded through the RL-time adapter (``VisionEncoderAdapter`` with
the resolved RL encoder config), and every window's camera input is passed through
``VisionEncoderAdapter.extract_cacheable_feature``, the fp16 feature that DrM caches in replay and feeds to its
trainable head. A window of the validation data is the RL observation history: three frames ``stride`` simulator
steps apart (stride 2 is the RL spacing). The state of a window is chosen by ``feature`` (``FEATURES``); the default
is the head's whole input, the per-frame features of the three frames, oldest first, concatenated.

Our ``Evaluator`` computes every metric, with the same splits, windows, camera groups and probe targets; only
``Evaluator.states`` (the encoder call) is overridden. A *column* of a camera group is one RL camera input:

- single-camera encoders (DynaMo, ReViWo, SinCro, 3D-MVP condition B): one camera, as for our model;
- multi-view encoders (SNeRL, 3D-MVP condition A, 3 views): the RL evaluation design (``s4d.rl.evaluate``). Training
  column i holds training cameras (i, i+1, i+3) mod 6; a held-out camera column keeps the last two cameras of its
  nearest training camera's column and replaces the first by the held-out camera (near/trajectory-set cameras perturb
  that same training camera). Columns therefore share cameras (see ``docs/BASELINES.md``).

Rendering metrics (PSNR, depth, Chamfer, motion EPE, Gaussian usage) belong to our model and are not computed.
TODO(SNeRL rendering): SNeRL's NeRF decoder could render the held-out views for PSNR (step 4 of the acceptance
order); not implemented.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from s4d.rl.encoders import VisionEncoderAdapter
from s4d.rl.evaluate import VIEW_OFFSETS
from s4d.train.evaluate import HELDOUT_GROUPS, Evaluator, _cpu, frame_interval
from s4d.train.workers import fork_safe_iter

# Window state from the cached (T, D) per-frame features (oldest first).
FEATURES = {
    "stack": "per-frame features of the window's 3 frames concatenated (T*D), the input of the RL head",
    "newest": "the newest frame's feature (D), the frame the agent acts on",
    "middle": "the middle frame's feature (D), the time of the hand and object position probe targets",
}
# RL encoder configs (configs/rl/encoders/) of frozen baselines; the 3D-MVP conditions are separate configs.
BASELINE_ENCODERS = ("dynamo", "snerl", "mvp3d_multiview", "mvp3d_single", "sincro", "reviwo")


def load_adapter(encoder: str, task: str, export: str | Path) -> tuple[VisionEncoderAdapter, dict]:
    """The RL-time adapter of RL encoder config ``encoder`` for ``task`` with ``export`` (``vision.export_path``),
    resolved as ``scripts/train_rl.py`` resolves it."""
    from s4d.rl.protocol import resolve_config

    if encoder not in BASELINE_ENCODERS:
        raise ValueError(f"{encoder!r} is not a frozen baseline encoder config ({', '.join(BASELINE_ENCODERS)})")
    cfg = resolve_config(task, encoder, 0, [f"vision.export_path={export}"])
    adapter = VisionEncoderAdapter(cfg)
    if adapter.backbone_trainable:
        raise ValueError(f"{encoder} is trained end to end; there is no frozen representation to evaluate")
    return adapter, cfg


def input_views(adapter: VisionEncoderAdapter) -> int:
    return int((adapter.observation or {}).get("views", 1))


def nearest_training_cameras(train_c2w: torch.Tensor, query_c2w: torch.Tensor) -> torch.Tensor:
    """(Vt,4,4), (Vq,4,4) OpenCV c2w -> (Vq,) index of the training camera with the closest viewing direction, the
    rule of ``s4d.rl.evaluate.nearest_training_camera``."""
    return (query_c2w[:, :3, 2] @ train_c2w[:, :3, 2].T).argmax(1)


def input_columns(n_train: int, views: int, bases: torch.Tensor | None = None) -> torch.Tensor:
    """(C, views) camera indices of every column into the pool [training cameras, group cameras]: training columns
    without ``bases``, else one column per group camera with base training camera ``bases[j]``."""
    offsets = (0,) if views == 1 else VIEW_OFFSETS[views]

    def training(i: int) -> list[int]:
        return [(i + offset) % n_train for offset in offsets]

    if bases is None:
        return torch.tensor([training(i) for i in range(n_train)])
    return torch.tensor([[n_train + j, *training(int(b))[1:]] for j, b in enumerate(bases)])


def camera_pool(batch: dict, group: str = "") -> dict[str, torch.Tensor]:
    """Training cameras, then the cameras of held-out ``group`` (a ``HELDOUT_GROUPS`` prefix), along the view axis:
    ``images`` (B,T,V,3,H,W) uint8, ``depth`` (B,T,V,H,W) metres, ``K`` (B,V,3,3), ``c2w`` (B,V,4,4) OpenCV."""
    pool = {"images": batch["images"], "depth": batch["depth"][:, :, :, 0], "K": batch["K"], "c2w": batch["c2w"]}
    if not group:
        return pool
    c2w = batch.get(f"{group}_c2w")
    extra = {
        "images": batch[f"{group}_images"],
        "depth": batch[f"{group}_depth"][:, :, :, 0],
        "K": batch[f"{group}_K"],
        "c2w": torch.linalg.inv(batch[f"{group}_w2c"]) if c2w is None else c2w,  # the far cameras store only w2c
    }
    return {k: torch.cat((pool[k], extra[k].to(pool[k].device)), dim=2 if k in ("images", "depth") else 1) for k in pool}


def column_observations(adapter: VisionEncoderAdapter, pool: dict, columns: torch.Tensor) -> torch.Tensor | dict:
    """The RL observation of every (window, column), windows major: the channel-stacked uint8 history (N,3T,H,W) of
    one camera, or the dict observation the backbone declares (``s4d/rl/env.py`` ``stacked``) with ``rgb``
    (N,T,V,3,H,W) uint8, ``depth`` (N,T,V,H,W), ``K`` (N,T,V,3,3), ``c2w`` (N,T,V,4,4)."""
    B, T = pool["images"].shape[:2]
    rgb = pool["images"][:, :, columns].transpose(1, 2).flatten(0, 1)  # (B*C,T,Vin,3,H,W)
    if adapter.observation is None:
        return rgb[:, :, 0].flatten(1, 2)
    obs = {"rgb": rgb}
    if adapter.observation.get("depth"):
        obs["depth"] = pool["depth"][:, :, columns].transpose(1, 2).flatten(0, 1)
    if adapter.observation.get("cameras"):
        for key in ("K", "c2w"):
            per_window = pool[key][:, columns].flatten(0, 1)  # cameras are fixed over the history
            obs[key] = per_window[:, None].expand(-1, T, *per_window.shape[1:])
    return obs


class BaselineEvaluator(Evaluator):
    """``Evaluator`` retrieval, probes and state statistics with a frozen baseline's RL features as the states.

    ``cfg`` is our pretraining config (data split, strides, probe settings); the evaluated "model" is the adapter."""

    def __init__(self, cfg: dict, val_loaders: dict, probe_loaders: dict | None, device: torch.device, n_train_cams: int,
                 feature: str = "stack", chunk: int = 32):
        super().__init__(cfg, val_loaders, probe_loaders, None, device, n_train_cams)
        if feature not in FEATURES:
            raise ValueError(f"unknown feature {feature!r}; one of {sorted(FEATURES)}")
        self.feature, self.chunk = feature, int(chunk)

    @torch.no_grad()
    def __call__(self, adapter: VisionEncoderAdapter, full: bool = True) -> tuple[dict, dict]:
        """-> (summary with every metric suffixed ``@s<stride>``, validation batches per stride)."""
        if adapter.replay_atom_is_stack_feature and self.feature != "stack":
            raise ValueError(f"{adapter.vision_key} returns one state per history; only feature 'stack' applies")
        summary, num_batches = {}, {}
        for stride, loader in self.val_loaders.items():
            part, num_batches[stride] = self.evaluate_stride(adapter, 0, full, int(stride), loader)
            summary.update({f"{k}@s{stride}": v for k, v in part.items()})
        return summary, num_batches

    @torch.no_grad()
    def evaluate_stride(self, adapter, step: int, full: bool, stride: int, loader) -> tuple[dict, int]:
        """The state collection of ``Evaluator.evaluate_stride`` without the model's losses, renders and panels."""
        states, probe_states, strides = [], [], []
        group_states: dict[str, list] = defaultdict(list)
        count = 0
        for i, batch in enumerate(fork_safe_iter(loader)):
            if not full and i >= self.max_batches:
                break
            states.append(self.states(adapter, batch))
            for prefix, name in HELDOUT_GROUPS.items():
                if f"{prefix}_images" in batch:
                    group_states[name].append(self.states(adapter, batch, prefix))
            if "probe_state" in batch:
                probe_states.append(_cpu(batch["probe_state"]))
            strides.append(frame_interval(batch))
            count += 1
        if not states:
            raise ValueError("validation loader is empty")
        # retrieval and probes on every call, as in our full evaluation
        return self.representation(adapter, stride, loader, states, group_states, probe_states, strides, True), count

    def states(self, adapter, batch: dict, group: str = "") -> torch.Tensor:
        """(B, C, D) CPU float states of the columns of one camera group (see the module docstring)."""
        pool = camera_pool(batch, group)
        n_train = batch["images"].shape[2]
        bases = None
        if group:
            bases = nearest_training_cameras(pool["c2w"][0, :n_train].cpu(), pool["c2w"][0, n_train:].cpu())
        columns = input_columns(n_train, input_views(adapter), bases)
        B, C = batch["images"].shape[0], len(columns)
        obs = column_observations(adapter, pool, columns)
        feats = []
        for start in range(0, B * C, self.chunk):
            if isinstance(obs, dict):
                part = {k: v[start : start + self.chunk].to(self.device) for k, v in obs.items()}
            else:
                part = obs[start : start + self.chunk].to(self.device)
            feats.append(self.window_state(adapter, adapter.extract_cacheable_feature(part)))
        return torch.cat(feats).view(B, C, -1)

    def window_state(self, adapter, cached: torch.Tensor) -> torch.Tensor:
        """Cached fp16 features -> (N, D) float states: (N,T,D) per-frame features select ``self.feature``; a
        sequence-state backbone's feature is the state."""
        if adapter.replay_atom_is_stack_feature or self.feature == "stack":
            return _cpu(cached.flatten(1))
        return _cpu(cached[:, -1] if self.feature == "newest" else cached[:, cached.shape[1] // 2])
