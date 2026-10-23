"""Evaluator of the per-frame pilots (s4d/train/perframe.py).

The reference ``Evaluator`` with two per-frame parts: held-out camera groups render the set decoded from frame t at time
t (``render_group``), and the panels show per-frame reconstructions and, for the temporal pilot, the cross-time
reconstructions with the persistence reference. Retrieval, probes and state statistics use the chronological stack of
the per-frame states (``encode_states`` with ``frame_mode`` "stack"), i.e. exactly the policy state.
"""

from __future__ import annotations

import torch

from s4d.diag import panels as P
from s4d.diag import wandb_log as W
from s4d.train.evaluate import Evaluator
from s4d.train.perframe import render_per_time


class PerFrameEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.heldout_diag:
            raise ValueError("eval.heldout_sets diagnostics (oracle and per-window Chamfer) are not defined per frame")
        if self.future:
            raise ValueError("per-frame pilots use the reconstruction objective")

    def render_group(self, gs, batch: dict, prefix: str) -> dict:
        T = batch["images"].shape[1]
        return render_per_time(gs, gs.xyz, batch[f"{prefix}_w2c"], batch[f"{prefix}_K"], T, *batch["images"].shape[-2:],
                               self.near, self.far)

    @torch.no_grad()
    def panels(self, model, first, step: int, summary: dict, tag: str = "") -> None:
        batch, out, _gs, _heldout, _psnr = first
        out_dir = self.logger.eval_dir(step) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        b, src = 0, int(out["source"][0])
        T, V = batch["images"].shape[1:3]
        img01 = batch["images"].float() / 255.0
        media: dict = {}
        for v in range(V):
            items = []
            for row in ("gt", "render", "err", "gt_depth", "depth", "alpha"):
                for t in range(T):
                    gt, rd = img01[b, t, v], out["rgb"][b, t, v]
                    if row == "gt":
                        items.append(P.to_uint8(gt))
                    elif row == "render":
                        items.append(P.to_uint8(rd))
                    elif row == "err":
                        items.append(P.colorize_error((rd - gt).abs().mean(0), 0.25))
                    elif row == "gt_depth":
                        items.append(P.colorize_depth(batch["depth"][b, t, v], self.near, self.far))
                    elif row == "depth":
                        items.append(P.colorize_depth(out["depth"][b, t, v], self.near, self.far, out["alpha"][b, t, v] > 0.5))
                    else:
                        items.append(P.colorize_gray(out["alpha"][b, t, v]))
            panel = P.grid(items, ncol=T)
            self.logger.save_png(out_dir / f"recon_cam{v}.png", panel)
            media[f"panels/recon_cam{v}"] = W.image(
                panel, f"cam {v} (source {src}); per-frame decoding; rows: GT|render|err|GT depth|depth|alpha; cols t0..t2")
        if "cross_rgb" in out:  # temporal pilot: moved source sets at the target time, source camera
            items = []
            pairs = out["pairs"]
            for row in ("target", "cross", "persist", "err"):
                for p, (_a, tb, _) in enumerate(pairs):
                    gt = img01[b, tb, src]
                    if row == "target":
                        items.append(P.to_uint8(gt))
                    elif row == "cross":
                        items.append(P.to_uint8(out["cross_rgb"][b, p, src]))
                    elif row == "persist":
                        items.append(P.to_uint8(out["persist_rgb"][b, p, src]) if "persist_rgb" in out else P.to_uint8(gt * 0))
                    else:
                        items.append(P.colorize_error((out["cross_rgb"][b, p, src] - gt).abs().mean(0), 0.25))
            panel = P.grid(items, ncol=len(pairs))
            self.logger.save_png(out_dir / "cross_time.png", panel)
            media["panels/cross_time"] = W.image(
                panel, "source camera; cols pairs 0->1, 1->2, 0->2; rows: target frame|moved source set|unmoved (persistence)|err")
        self.logger.media(step, {f"{tag}/{k}" if tag else k: v for k, v in media.items()})
