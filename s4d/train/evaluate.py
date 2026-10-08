"""Periodic / final evaluation: metrics against the §10 criteria, W&B panels, and the local PNG mirror."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from s4d.config import get
from s4d.data.contract import collate
from s4d.data.metaworld.cameras import LOOKAT, RADIUS, opengl_to_opencv_c2w, orbit_c2w_opengl
from s4d.diag import panels as P
from s4d.diag import wandb_log as W
from s4d.diag.local_log import RunLogger
from s4d.diag.heldout import chamfer_metrics, oracle_views, render_chamfer
from s4d.diag.pointclouds import cloud_panel, fused_gt_points, gaussian_points
from s4d.diag.probes import cross_view_retrieval, fit_and_score, probe_targets
from s4d.diag.tracks import sample_track_pixels, track_panel
from s4d.geometry import lift_depth, project
from s4d.losses.depth import abs_rel
from s4d.losses.invariance import state_statistics
from s4d.losses.rgb import masked_psnr
from s4d.model.encoder import sample_tube_mask
from s4d.model.gaussians import GaussianSet
from s4d.model.render import render_rgbd
from s4d.train.loop import Model, forward_losses, move_batch
from s4d.train.workers import fork_safe_iter

RETRIEVAL_STATES = 64
# Held-out camera groups: "eval" = the four far cameras ("extrapolation", metric name "heldout", context only);
# "near" / "traj" = near-view sets within half / the full RL trajectory ranges (validation episodes only).
HELDOUT_GROUPS = {"eval": "heldout", "near": "near", "traj": "traj"}


def encode_states(model: Model, images_u8: torch.Tensor) -> torch.Tensor:
    """images (B,T,V,3,H,W) uint8 -> slots (B,V,K,Ds) without masking."""
    B, T, V = images_u8.shape[:3]
    x = images_u8.float().div(255.0).permute(0, 2, 1, 3, 4, 5).reshape(B * V, T, *images_u8.shape[3:])
    x = x[:, : model.encoder.cfg.num_frames]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        slots = model.encoder(x, None, mask_ratio=0.0)["slots"]
    return slots.float().view(B, V, *slots.shape[1:])


def _cpu(x):
    return x.detach().float().cpu() if torch.is_tensor(x) else x


def _select_retrieval_windows(episodes: list[str], t0s: list[int], n: int = RETRIEVAL_STATES) -> list[int]:
    """Exactly one middle window per distinct episode; never substitute within-episode states."""
    by_ep: dict[str, list[int]] = defaultdict(list)
    for i, episode in enumerate(episodes):
        by_ep[episode].append(i)
    chosen = []
    for indices in by_ep.values():
        indices.sort(key=lambda i: t0s[i])
        chosen.append(indices[len(indices) // 2])
    return chosen[:n]


def _spread_windows(episodes: list[str], t0s: list[int], n: int = RETRIEVAL_STATES) -> list[int]:
    """n windows spread evenly over time within every episode, taken round-robin across episodes."""
    by_ep: dict[str, list[int]] = defaultdict(list)
    for i, episode in enumerate(episodes):
        by_ep[episode].append(i)
    per = -(-n // max(1, len(by_ep)))
    columns = []
    for indices in by_ep.values():
        indices.sort(key=lambda i: t0s[i])
        picks = np.unique(np.linspace(0, len(indices) - 1, min(per, len(indices))).round().astype(int))
        columns.append([indices[k] for k in picks])
    chosen = [col[r] for r in range(per) for col in columns if r < len(col)]
    return chosen[:n]


def orbit_w2c(center, distance: float, elevation_deg: float, frames: int, device) -> torch.Tensor:
    mats = [
        np.linalg.inv(opengl_to_opencv_c2w(orbit_c2w_opengl(center, distance, az, elevation_deg)))
        for az in np.linspace(0, 360, frames, endpoint=False)
    ]
    return torch.tensor(np.stack(mats), dtype=torch.float32, device=device)


def image_plane_flow(
    depth0: torch.Tensor, disp: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, w2c: torch.Tensor
) -> np.ndarray:
    """depth0 (H,W), disp (3,H,W) world displacement -> (H,W,2) pixel flow of the lifted points."""
    xyz = lift_depth(depth0[None], K[None], c2w[None])[0]
    H, Wd = depth0.shape
    pts = torch.cat((xyz.reshape(1, -1, 3), (xyz + disp.permute(1, 2, 0)).reshape(1, -1, 3)), 1)
    uv, _ = project(pts, K[None], w2c[None])
    flow = (uv[0, H * Wd :] - uv[0, : H * Wd]).reshape(H, Wd, 2)
    flow[(depth0 <= 0)] = 0
    return flow.numpy()


class Evaluator:
    """Validation metrics, retrieval, probes and panels, computed separately for every validation stride.

    ``val_loaders`` and ``probe_loaders`` map a frame stride (simulator steps) to a loader. Every metric is
    reported once per stride with the suffix ``@s<stride>``; panels go to ``eval/step_*/s<stride>/``.
    """

    def __init__(
        self,
        cfg: dict,
        val_loaders: dict,
        probe_loaders: dict | None,
        logger: RunLogger,
        device: torch.device,
        n_train_cams: int,
    ):
        self.cfg, self.val_loaders, self.probe_loaders = cfg, dict(val_loaders), dict(probe_loaders or {})
        self.logger, self.device, self.n_train = logger, device, n_train_cams
        self.max_batches = int(get(cfg, "eval.max_batches", 8))
        self.heldout_diag = bool(get(cfg, "eval.heldout_sets", False))
        self.probe_every = int(get(cfg, "eval.probe_every", 10000))
        self.near, self.far = float(cfg["render"]["near"]), float(cfg["render"]["far"])

    # ------------------------------------------------------------------------------------ main
    @torch.no_grad()
    def __call__(self, model: Model, step: int, full: bool = False) -> dict:
        summary: dict = {}
        num_batches = {}
        for stride, loader in self.val_loaders.items():
            part, num_batches[stride] = self.evaluate_stride(model, step, full, int(stride), loader)
            summary.update({f"{k}@s{stride}": v for k, v in part.items()})
        out_dir = self.logger.eval_dir(step)
        (out_dir / "summary.json").write_text(json.dumps({"step": step, "num_batches": num_batches, **summary}, indent=1))
        self.logger.scalars(
            step, {f"val/{k.split('/', 1)[-1]}": v for k, v in summary.items() if isinstance(v, (int, float))}
        )
        for stride in self.val_loaders:
            self.logger.text(
                f"eval step {step} stride {stride}: psnr {summary.get(f'metric/psnr@s{stride}', float('nan')):.2f} "
                f"heldout {summary.get(f'metric/psnr_heldout@s{stride}', float('nan')):.2f} rel_epe02 "
                f"{summary.get(f'metric/rel_epe_02@s{stride}', float('nan')):.3f} retrieval "
                f"{summary.get(f'metric/retrieval_top1_train@s{stride}', float('nan')):.3f}"
            )
        return summary

    @torch.no_grad()
    def evaluate_stride(self, model: Model, step: int, full: bool, stride: int, loader) -> tuple[dict, int]:
        model.eval()
        probe_loader = self.probe_loaders.get(stride)
        sums: dict[str, float] = defaultdict(float)
        count = 0
        counts = defaultdict(int)
        states, episodes, t0s, probe_states, strides = [], [], [], [], []
        group_states: dict[str, list] = defaultdict(list)
        window_rows: list[dict] = []
        first = None
        for i, raw in enumerate(fork_safe_iter(loader)):
            if not full and i >= self.max_batches:
                break
            batch = move_batch(raw, self.device)
            B, T, V = batch["images"].shape[:3]
            out = forward_losses(
                model,
                batch,
                self.cfg,
                step,
                source=torch.zeros(B, dtype=torch.long, device=self.device),
                mask_ratio=0.0,
                return_renders=(i == 0),
            )
            gs = out["gs"]
            heldout = None
            metrics = {
                **{f"loss/{k}": v for k, v in out["losses"].items()},
                **{f"metric/{k}": v for k, v in out["metrics"].items()},
            }
            for prefix, name in HELDOUT_GROUPS.items():
                if f"{prefix}_images" not in batch:
                    continue
                rendered = render_rgbd(
                    gs,
                    gs.xyz_sequence(),
                    batch[f"{prefix}_w2c"],
                    batch[f"{prefix}_K"],
                    *batch["images"].shape[-2:],
                    self.near,
                    self.far,
                )
                if prefix == "eval":
                    heldout = rendered
                Hp = rendered["rgb"].flatten(0, 2)
                target = batch[f"{prefix}_images"].float().flatten(0, 2) / 255
                psnr_ho = masked_psnr(Hp, target, torch.ones_like(Hp[:, :1], dtype=torch.bool))
                ho_depth = batch[f"{prefix}_depth"].flatten(0, 2)
                absrel_ho = abs_rel(rendered["depth"].flatten(0, 2), ho_depth, ho_depth > 0)
                metrics.update({f"metric/psnr_{name}": psnr_ho.nanmean(), f"metric/depth_absrel_{name}": absrel_ho.mean()})
                if self.heldout_diag:
                    metrics.update(self.oracle_metrics(batch, prefix, name, Hp, target))
                    if prefix == "traj":
                        window_rows.append(render_chamfer(rendered, batch, 0, prefix, self.far))
                group_states[name].append(_cpu(encode_states(model, batch[f"{prefix}_images"]).flatten(2)))
            if self.heldout_diag:
                window_rows.append(chamfer_metrics(batch, gs, 0, self.far))
            for k, v in metrics.items():
                val = float(v)
                if math.isfinite(val):
                    denominator = B
                    metric = k.removeprefix("metric/")
                    pair = metric.rsplit("_", 1)[-1]
                    if metric == "dyn_alpha_share_moving":
                        denominator = float(out["metrics"]["dynamic_score_moving_count"])
                    elif metric.startswith(("rel_epe_", "epe_moving_")):
                        denominator = float(out["metrics"][f"motion_moving_count_{pair}"])
                    elif metric.startswith("epe_static_"):
                        denominator = float(out["metrics"][f"motion_static_count_{pair}"])
                    elif metric.startswith("epe_"):
                        denominator = float(out["metrics"][f"motion_valid_weight_{pair}"])
                    sums[k] += val * denominator
                    counts[k] += denominator
            count += 1
            states.append(_cpu(out["slots"].flatten(2)))
            episodes.extend(batch["meta"]["episode"])
            t0s.extend(int(t[0]) for t in batch["meta"]["t_indices"])
            if "probe_state" in batch:
                probe_states.append(_cpu(batch["probe_state"]))
            strides.append(
                torch.as_tensor(batch["meta"]["stride"]) * torch.as_tensor(batch["meta"].get("dt_seconds", [1.0] * B))
            )
            if i == 0:
                ones = torch.ones(B * T * V, 1, *batch["images"].shape[-2:], dtype=torch.bool, device=self.device)
                psnr_b = masked_psnr(out["rgb"].flatten(0, 2), (batch["images"].float() / 255).flatten(0, 2), ones)
                first = (
                    move_batch(raw, "cpu"),
                    {k: _cpu(v) if torch.is_tensor(v) else v for k, v in out.items() if k != "gs"},
                    gs,
                    heldout,
                    psnr_b.view(B, T, V).cpu(),
                )
        if not states:
            raise ValueError("validation loader is empty")
        summary = {k: v / counts[k] for k, v in sums.items()}
        if window_rows:
            keys = sorted({k for row in window_rows for k in row})
            summary.update(
                {f"metric/{k}": float(torch.tensor([row.get(k, float("nan")) for row in window_rows]).nanmean()) for k in keys}
            )
            summary["metric/cd_windows"] = float(sum("cd_centers_p2g_mean" in row for row in window_rows))
        all_states = torch.cat(states)  # (M,V,D)
        held = {name: torch.cat(rows) for name, rows in group_states.items() if rows}
        periodic = full or (self.probe_every > 0 and step % self.probe_every == 0)
        if periodic:
            summary.update(self.retrieval(model, stride))
            if self.heldout_diag:
                summary.update(self.retrieval_sets(model, stride, loader))
        summary.update({f"metric/{k}_val": float(v) for k, v in state_statistics(all_states).items()})
        if probe_loader is not None and probe_states and periodic:
            probe = self.probes(model, probe_loader, all_states, held, torch.cat(probe_states), torch.cat(strides))
            summary.update({f"metric/{k}": v for k, v in probe.items()})
        if first is not None:
            self.panels(model, first, step, summary, tag=f"s{stride}")
        model.train()
        return summary, count

    def oracle_metrics(self, batch: dict, prefix: str, name: str, rendered: torch.Tensor, target: torch.Tensor) -> dict:
        """Oracle sanity (training-camera GT fused and splatted into the held-out cameras) and the model's held-out
        PSNR on the pixels the oracle covers. ``rendered``/``target`` are (B*T*V,3,H,W) in [0,1]."""
        B, T = batch["images"].shape[:2]
        rgbs, covers = [], []
        for b in range(B):
            for t in range(T):
                rgb, covered = oracle_views(batch, b, t, prefix, self.near, self.far)
                rgbs.append(rgb)
                covers.append(covered)
        oracle, covered = torch.cat(rgbs), torch.cat(covers)  # same (b,t,v) order as flatten(0, 2)
        return {
            f"metric/oracle_coverage_{name}": covered.float().mean(),
            f"metric/oracle_psnr_covered_{name}": masked_psnr(oracle, target, covered).nanmean(),
            f"metric/psnr_{name}_covered": masked_psnr(rendered, target, covered).nanmean(),
        }

    @torch.no_grad()
    def retrieval_sets(self, model: Model, stride: int, loader) -> dict:
        """Retrieval on 64 validation windows (evenly spaced in time within each validation episode) for the training,
        far (heldout), near and trajectory cameras; query held-out camera vs every training camera, as ``retrieval``."""
        dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
        if getattr(dataset, "heldout", None) is None:
            return {}
        indices = [
            i
            for i in _spread_windows([x[0] for x in dataset.samples], [x[1] for x in dataset.samples])
            if dataset.samples[i][0] in dataset.heldout
        ]
        groups: dict[str, list] = defaultdict(list)
        for batch in DataLoader(Subset(dataset, indices), batch_size=8, collate_fn=collate):
            groups["train"].append(encode_states(model, batch["images"].to(self.device)).flatten(2).cpu())
            for prefix, name in HELDOUT_GROUPS.items():
                groups[name].append(encode_states(model, batch[f"{prefix}_images"].to(self.device)).flatten(2).cpu())
        train = torch.cat(groups.pop("train"))
        result = {
            "metric/retrieval_top1_train_val": cross_view_retrieval(train, self.n_train)["retrieval_top1_train"],
            "metric/retrieval_val_windows": float(len(train)),
        }
        for name, rows in groups.items():
            states = torch.cat((train, torch.cat(rows)), dim=1)
            result[f"metric/retrieval_top1_{name}_val"] = cross_view_retrieval(states, self.n_train)["retrieval_top1_heldout"]
        return result

    @torch.no_grad()
    def retrieval(self, model: Model, stride: int) -> dict:
        if get(self.cfg, "data.regime") != "metaworld":
            return {"metric/retrieval_protocol_available": False}
        from s4d.data.metaworld.dataset import MetaworldWindowDataset, list_episodes

        path = Path(get(self.cfg, "data.root")) / f"{get(self.cfg, 'data.task')}.hdf5"
        episodes = get(self.cfg, "data.episodes") or list_episodes(path)
        ds = MetaworldWindowDataset(path, episodes, strides=(stride,), with_eval=True)
        indices = _select_retrieval_windows([x[0] for x in ds.samples], [x[1] for x in ds.samples])
        result = {
            "metric/retrieval_distinct_episodes": len(indices),
            "metric/retrieval_protocol_available": len(indices) == RETRIEVAL_STATES,
        }
        if len(indices) != RETRIEVAL_STATES:
            return result
        views = []
        for batch in DataLoader(Subset(ds, indices), batch_size=8, collate_fn=collate):
            training = encode_states(model, batch["images"].to(self.device)).flatten(2).cpu()
            heldout = encode_states(model, batch["eval_images"].to(self.device)).flatten(2).cpu()
            views.append(torch.cat((training, heldout), dim=1))
        states = torch.cat(views)
        result.update({f"metric/{k}": v for k, v in cross_view_retrieval(states, self.n_train).items()})
        result.update(
            {f"metric/{k}_retrieval": float(v) for k, v in state_statistics(states[:, : self.n_train].mean(1)).items()}
        )
        result["retrieval_episode_pool"] = "all task episodes, distinct episodes, held-out camera queries"
        return result

    # ------------------------------------------------------------------------------------ probes
    @torch.no_grad()
    def probes(
        self,
        model: Model,
        probe_loader,
        val_states: torch.Tensor,
        val_held_states: dict[str, torch.Tensor],
        val_probe: torch.Tensor,
        val_stride: torch.Tensor,
    ) -> dict[str, float]:
        xs, ys = [], []
        for i, raw in enumerate(fork_safe_iter(probe_loader)):
            if i >= int(get(self.cfg, "eval.probe_batches", 24)):
                break
            batch = move_batch(raw, self.device)
            s = encode_states(model, batch["images"]).flatten(2).cpu()  # (B,V,D)
            B, V, D = s.shape
            xs.append(s.reshape(B * V, D))
            tg = probe_targets(
                batch["probe_state"].cpu(),
                torch.as_tensor(batch["meta"]["stride"]) * torch.as_tensor(batch["meta"].get("dt_seconds", [1.0] * B)),
            )
            ys.append({k: v.repeat_interleave(V, 0) for k, v in tg.items()})
        train_x = torch.cat(xs)
        train_y = {k: torch.cat([y[k] for y in ys]) for k in ys[0]}
        val_t = probe_targets(val_probe, val_stride)
        M, V, D = val_states.shape
        sets = {
            "val_traincams": (val_states.reshape(M * V, D), {k: v.repeat_interleave(V, 0) for k, v in val_t.items()}),
        }
        for name, held in val_held_states.items():
            if len(held) != M:  # held-out views exist only for some windows (never mixed within one evaluation)
                continue
            Ve = held.shape[1]
            sets[f"val_{name}"] = (held.reshape(M * Ve, D), {k: v.repeat_interleave(Ve, 0) for k, v in val_t.items()})
        return fit_and_score(train_x, train_y, sets)

    # ------------------------------------------------------------------------------------ panels
    @torch.no_grad()
    def panels(self, model: Model, first, step: int, summary: dict, tag: str = "") -> None:
        batch, out, gs, heldout, psnr_b = first
        out_dir = self.logger.eval_dir(step) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        b, src = 0, int(out["source"][0])
        T, V = batch["images"].shape[1:3]
        H, Wd = batch["images"].shape[-2:]
        media: dict = {}
        img01 = batch["images"].float() / 255.0
        depth_scale = self.far - self.near
        encoder = model.encoder
        score_bv = batch["motion_score"][b].permute(1, 0, 2, 3, 4).to(self.device)  # (V,T,1,H,W)
        vis_mask = sample_tube_mask(
            encoder.patch_scores(score_bv), encoder.cfg.mask_ratio, encoder.cfg.motion_patch_threshold
        ).cpu()
        # reconstruction panels per camera and timestep
        for v in range(V):
            items = []
            for row in ("gt", "render", "err", "gt_depth", "depth", "depth_err", "alpha", "score", "mask"):
                for t in range(T):
                    gt, rd = img01[b, t, v], out["rgb"][b, t, v]
                    gtd, rdd = batch["depth"][b, t, v], out["depth"][b, t, v]
                    valid = gtd > 0
                    if row == "gt":
                        items.append(P.to_uint8(gt))
                    elif row == "render":
                        items.append(P.to_uint8(rd))
                    elif row == "err":
                        items.append(P.colorize_error((rd - gt).abs().mean(0), 0.25))
                    elif row == "gt_depth":
                        items.append(P.colorize_depth(gtd, self.near, self.far))
                    elif row == "depth":
                        items.append(P.colorize_depth(rdd, self.near, self.far, out["alpha"][b, t, v] > 0.5))
                    elif row == "depth_err":
                        items.append(P.colorize_error((rdd - gtd).abs() * valid, 0.1 * depth_scale))
                    elif row == "alpha":
                        items.append(P.colorize_gray(out["alpha"][b, t, v]))
                    elif row == "score":
                        items.append(P.colorize_gray(batch["motion_score"][b, t, v]))
                    else:
                        items.append(P.overlay_patches(P.to_uint8(gt), vis_mask[v], encoder.cfg.grid))
            panel = P.grid(items, ncol=T)
            self.logger.save_png(out_dir / f"recon_cam{v}.png", panel)
            media[f"panels/recon_cam{v}"] = W.image(
                panel,
                f"cam {v} (source {src}); rows: GT|render|err|GT depth|depth|"
                "depth err|alpha|motion score|tube mask; cols t0..t2",
            )
        # motion panel (source camera): GT vs predicted image-plane flow + magnitude + error for the three pairs
        items = []
        for p, (_a, _, pix) in enumerate(out["pairs"]):
            gt_flow = image_plane_flow(
                batch["depth"][b, pix, src, 0],
                batch["motion3d"][b, p, src],
                batch["K"][b, src],
                batch["c2w"][b, src],
                batch["w2c"][b, src],
            )
            pr_flow = image_plane_flow(
                batch["depth"][b, pix, src, 0],
                out["pred_disp"][b, p, src],
                batch["K"][b, src],
                batch["c2w"][b, src],
                batch["w2c"][b, src],
            )
            gt_mag = batch["motion3d"][b, p, src].norm(dim=0)
            pr_mag = out["pred_disp"][b, p, src].norm(dim=0)
            err = (out["pred_disp"][b, p, src] - batch["motion3d"][b, p, src]).norm(dim=0) * (
                batch["motion_weight"][b, p, src, 0] > 0
            )
            items += [
                P.flow_color(gt_flow, 16.0),
                P.flow_color(pr_flow, 16.0),
                P.colorize_error(gt_mag, 0.1),
                P.colorize_error(pr_mag, 0.1),
                P.colorize_error(err, 0.05),
            ]
        panel = P.grid(items, ncol=5)
        self.logger.save_png(out_dir / "motion.png", panel)
        media["panels/motion"] = W.image(
            panel, "rows: pairs 01,12,02; cols: GT flow | pred flow | GT |d| (0-10cm) | pred |d| | error (0-5cm)"
        )
        # tracks per camera
        pixels_by_cam = {}
        for v in range(V):
            pixels = sample_track_pixels(batch["motion_score"][b, 0, v], batch["motion_weight"][b, 0, v])
            pixels_by_cam[v] = pixels
            if len(pixels) == 0:
                continue
            panel = track_panel(batch, out, b, v, pixels)
            self.logger.save_png(out_dir / f"tracks_cam{v}.png", panel)
            media[f"panels/tracks_cam{v}"] = W.image(panel, "GT tracks green, predicted red; cols t0..t2")
        source_pixels = pixels_by_cam[src]
        if len(source_pixels):
            for v in range(V):
                panel = track_panel(batch, out, b, src, source_pixels, target_v=v)
                self.logger.save_png(out_dir / f"same_tracks_source{src}_target{v}.png", panel)
                media[f"panels/same_tracks_cam{v}"] = W.image(panel, "identical world trajectories across cameras")
        if heldout is not None:
            # held-out cameras
            items = []
            for ve in range(batch["eval_images"].shape[2]):
                for t in range(T):
                    items.append(P.to_uint8(batch["eval_images"][b, t, ve].float() / 255.0))
                for t in range(T):
                    items.append(P.to_uint8(heldout["rgb"][b, t, ve]))
                for t in range(T):
                    items.append(
                        P.colorize_depth(
                            heldout["depth"][b, t, ve].cpu(), self.near, self.far, heldout["alpha"][b, t, ve].cpu() > 0.5
                        )
                    )
            panel = P.grid(items, ncol=T)
            self.logger.save_png(out_dir / "heldout.png", panel)
            media["panels/heldout"] = W.image(panel, "held-out cameras: GT | render | rendered depth (never trained on)")
        # cross-source consistency: decode from every camera, render to train cam 0 at t0
        slots_all = encode_states(model, batch["images"][b : b + 1].to(self.device))[0]
        if "eval_images" in batch:
            extra = encode_states(model, batch["eval_images"][b : b + 1].to(self.device))[0]
            slots_all = torch.cat((slots_all, extra), dim=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gs_all = model.decoder(slots_all)
        target_w2c = batch["w2c"][b : b + 1, 0:1].to(self.device).expand(len(slots_all), -1, -1, -1)
        target_K = batch["K"][b : b + 1, 0:1].to(self.device).expand(len(slots_all), -1, -1, -1)
        renders = render_rgbd(gs_all, gs_all.xyz[:, None], target_w2c, target_K, H, Wd, self.near, self.far)["rgb"][:, 0, 0]
        std_map = renders.std(0).mean(0)
        items = [P.to_uint8(r) for r in renders.cpu()] + [P.colorize_error(std_map.cpu(), 0.2)]
        panel = P.grid(items, ncol=min(len(items), 6))
        self.logger.save_png(out_dir / "cross_source.png", panel)
        media["panels/cross_source"] = W.image(
            panel, "render of cam0 decoded from each train/held-out camera; last: pixel std"
        )
        summary["metric/cross_source_std"] = float(std_map.mean())
        # point clouds
        pts = gaussian_points(gs, b)
        gt_pts = fused_gt_points(batch, b)
        media["pointcloud/gaussians_rgb"] = W.object3d(pts["by_rgb"])
        media["pointcloud/gaussians_group"] = W.object3d(pts["by_group"], pts["vectors"])
        media["pointcloud/gt_fused"] = W.object3d(gt_pts)
        self.logger.save_ply(out_dir / "gaussians_rgb.ply", pts["by_rgb"])
        self.logger.save_ply(out_dir / "gt_fused.ply", gt_pts)
        self.logger.save_ply(out_dir / "gaussians_group.ply", pts["by_group"])
        np.save(out_dir / "gaussian_motion_vectors.npy", pts["vectors"])
        for name, points, vectors in (
            ("gaussians_rgb", pts["by_rgb"], None),
            ("gaussians_group", pts["by_group"], pts["vectors"]),
            ("gt_fused", gt_pts, None),
        ):
            panel = cloud_panel(points, vectors)
            self.logger.save_png(out_dir / f"pointcloud_{name}.png", panel)
            media[f"panels/pointcloud_{name}"] = W.image(panel, "local orthographic point-cloud mirror")
        # videos: rendered t0->t2 from the source camera and a 24-frame orbit
        seq = np.stack([P.to_uint8(out["rgb"][b, t, src]) for t in range(T)])
        self.logger.save_gif(out_dir / "sequence.gif", seq, fps=2)
        media["video/sequence"] = W.video(seq, fps=2, caption="rendered t0..t2, source camera")
        center = get(self.cfg, "eval.orbit_center", list(LOOKAT))
        w2c_orbit = orbit_w2c(
            center,
            float(get(self.cfg, "eval.orbit_distance", RADIUS)),
            float(get(self.cfg, "eval.orbit_elevation", -45.0)),
            24,
            self.device,
        )
        gs_b = GaussianSet(
            gs.xyz[b : b + 1],
            gs.scales[b : b + 1],
            gs.quats[b : b + 1],
            gs.opacity[b : b + 1],
            gs.rgb[b : b + 1],
            gs.delta01[b : b + 1],
            gs.delta12[b : b + 1],
            gs.group,
        )
        orbit = render_rgbd(
            gs_b,
            gs_b.xyz[:, None],
            w2c_orbit[None, :],
            batch["K"][b : b + 1, 0:1].to(self.device).expand(-1, 24, -1, -1),
            H,
            Wd,
            self.near,
            self.far,
        )["rgb"][0, 0]
        frames = np.stack([P.to_uint8(f) for f in orbit.cpu()])
        self.logger.save_gif(out_dir / "orbit.gif", frames, fps=8)
        media["video/orbit"] = W.video(frames, fps=8, caption="24-frame orbit around the workspace at t0")
        # per-sample table
        rows, local_rows = [], []
        for i in range(batch["images"].shape[0]):
            thumbnail = P.to_uint8(out["rgb"][i, 0, int(out["source"][i])])
            thumbnail_name = f"sample_{i:03d}_render_t0.png"
            self.logger.save_png(out_dir / thumbnail_name, thumbnail)
            thumb = W.image(thumbnail)
            rows.append(
                [
                    batch["meta"]["episode"][i],
                    int(batch["meta"]["t_indices"][i][0]),
                    int(batch["meta"]["stride"][i]),
                    float(psnr_b[i].nanmean()),
                    thumb,
                ]
            )
            local_rows.append(
                {**dict(zip(("episode", "t0", "stride", "psnr"), rows[-1][:4], strict=True)), "render_t0": thumbnail_name}
            )
        (out_dir / "samples.json").write_text(json.dumps(local_rows, indent=2))
        media["table/samples"] = W.table(["episode", "t0", "stride", "psnr", "render_t0"], rows)
        self.logger.media(step, {f"{tag}/{k}" if tag else k: v for k, v in media.items()})
