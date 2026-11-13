"""Identical held-out protocol at 200k/300k/400k, with outputs only in this evaluation job's directory."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
from _bootstrap import REPO, guard_gpus, require_passed_tests

guard_gpus()


def main():
    require_passed_tests()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from train import build_data

    from ops.jobs.run_identity import record_run_identity
    from ops.resources.gpu_guard import validate_runtime_path
    from s4d.config import dump_config, get, load_config
    from s4d.data.contract import collate
    from s4d.diag.local_log import RunLogger
    from s4d.diag.ours_length_eval import Means, grasp_anchor, handle_roi, probe_labels, score_probes, track_errors
    from s4d.losses.rgb import masked_psnr, masked_ssim
    from s4d.model.render import render_rgbd, require_prebuilt_renderer
    from s4d.train.checkpoint import load_checkpoint
    from s4d.train.continuation import file_sha
    from s4d.train.evaluate import Evaluator, encode_states
    from s4d.train.loop import build_model, forward_losses, move_batch, seed_everything
    from s4d.train.workers import fork_safe_iter

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    out_dir = validate_runtime_path(a.output_root / a.name)
    # Read checkpoint/config; never instantiate a logger or write anything inside the source run.
    if a.checkpoint.resolve().parent.parent == out_dir.resolve():
        raise ValueError("evaluation outputs must be separate from the source checkpoint run")
    cfg = load_config(a.config, a.set)
    cfg.setdefault("run", {})["name"] = a.name
    cfg.setdefault("eval", {})["heldout_sets"] = False  # no Chamfer diagnostics
    seed_everything(0)
    require_prebuilt_renderer()
    out_dir.mkdir(parents=True, exist_ok=True)
    record_run_identity(cfg, REPO, out_dir)
    dump_config(cfg, out_dir / "config.yaml")
    device = torch.device("cuda", 0)
    model = build_model(cfg).to(device).eval()
    digest = file_sha(a.checkpoint)
    step = load_checkpoint(a.checkpoint, model.encoder, model.decoder, restore_rng=False)
    _, val_loaders, probe_loaders, cameras = build_data(cfg)
    if cameras != 3:
        raise ValueError("length evaluation requires the established three-camera rig")
    logger = RunLogger(out_dir)
    try:
        standard = Evaluator(cfg, val_loaders, probe_loaders, logger, device, cameras)(model, step)
        anchors = {}
        for loader in [*val_loaders.values(), *probe_loaders.values()]:
            ds = loader.dataset
            for ep in ds.episodes:
                if ep not in anchors:
                    ef = ds.store.episode(ep)
                    anchors[ep] = grasp_anchor(ef.robot(ef.reference_camera))
        (out_dir / "grasp_sites.json").write_text(json.dumps(anchors, indent=2))
        # LPIPS uses pretrained AlexNet only if its torchvision weights are cached. Never download during evaluation.
        lpips_model, lpips_status = None, "unavailable: lpips package or cached AlexNet weights absent"
        if (Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth").is_file():
            try:
                import lpips

                lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
                lpips_status = "pretrained AlexNet, cached weights"
            except ImportError:
                pass
        extended, probes = {}, {}
        for stride, loader in val_loaders.items():
            means = Means()
            with torch.no_grad():
                for raw in itertools.islice(fork_safe_iter(loader), int(get(cfg, "eval.max_batches", 8))):
                    batch = move_batch(raw, device)
                    B, T, V, _, H, W = batch["images"].shape
                    result = forward_losses(
                        model,
                        batch,
                        cfg,
                        step,
                        source=torch.zeros(B, dtype=torch.long, device=device),
                        mask_ratio=0.0,
                        return_renders=True,
                    )
                    roi = handle_roi(batch, anchors)
                    target = batch["images"].float() / 255
                    # Camera 0 is the only source; cameras 1 and 2 are withheld from the encoder.
                    for region, mask in [("heldout", torch.ones_like(roi)), ("handle_initial_grasp_roi", roi)]:
                        pred = result["rgb"][:, :, 1:].flatten(0, 2)
                        gt = target[:, :, 1:].flatten(0, 2)
                        mask = mask[:, :, 1:].flatten(0, 2)
                        means.add(f"{region}/psnr", masked_psnr(pred, gt, mask))
                        means.add(f"{region}/ssim", masked_ssim(pred, gt, mask))
                        pd = result["depth"][:, :, 1:].flatten(0, 2)
                        gd = batch["depth"][:, :, 1:].flatten(0, 2)
                        valid = mask & (gd > 0) & (gd < cfg["render"]["far"])
                        means.add(f"{region}/depth_l1_m", (pd - gd).abs()[valid])
                        means.add(f"{region}/depth_absrel", ((pd - gd).abs() / gd.clamp_min(1e-6))[valid])
                        for axis in (-1, -2):
                            pg, gg = pd.diff(dim=axis), gd.diff(dim=axis)
                            neighbor_valid = valid.narrow(axis, 0, valid.shape[axis] - 1) & valid.narrow(
                                axis, 1, valid.shape[axis] - 1
                            )
                            means.add(f"{region}/depth_edge_l1_m", (pg - gg).abs()[neighbor_valid])
                        if lpips_model is not None and region == "heldout":
                            means.add("heldout/lpips", lpips_model(pred * 2 - 1, gt * 2 - 1))
                    for key, value in track_errors(batch, result).items():
                        means.add(key, value)
                    states = encode_states(model, batch["images"]).flatten(2)
                    for c1, c2 in ((0, 1), (0, 2), (1, 2)):
                        means.add(f"view_cosine/c{c1}_c{c2}", F.cosine_similarity(states[:, c1], states[:, c2], dim=-1))
                    synth_w2c = batch["w2c"].clone()
                    synth_w2c[..., 0, 3] += 0.01  # deterministic 1 cm camera-x shift, every camera
                    rgb = render_rgbd(
                        result["gs"],
                        result["gs"].xyz_sequence(),
                        synth_w2c,
                        batch["K"],
                        H,
                        W,
                        cfg["render"]["near"],
                        cfg["render"]["far"],
                    )["rgb"].clamp(0, 1)
                    synth_states = encode_states(model, (rgb * 255).round().to(torch.uint8)).flatten(2)
                    means.add("synthetic_view_feature_cosine", F.cosine_similarity(states[:, :1], synth_states, dim=-1))
            extended[stride] = means.result()
            collected = []
            for split, probe_loader, limit in [
                ("train", probe_loaders[stride], int(get(cfg, "eval.probe_batches", 24))),
                ("validation", loader, int(get(cfg, "eval.max_batches", 8))),
            ]:
                # Fresh deterministic train/val order: independent of the standard Evaluator's iterator history.
                fixed = DataLoader(
                    probe_loader.dataset,
                    batch_size=cfg["train"]["batch_size"],
                    shuffle=split == "train",
                    generator=torch.Generator().manual_seed(0),
                    num_workers=0,
                    collate_fn=collate,
                )
                xs, targets = [], []
                with torch.no_grad():
                    for raw in itertools.islice(fork_safe_iter(fixed), limit):
                        batch = move_batch(raw, device)
                        # Keep the 3-frame architecture; replace the unseen future frame with the present frame.
                        history = batch["images"].clone()
                        history[:, 2] = history[:, 1]
                        xs.append(encode_states(model, history)[:, 0].flatten(1).cpu())
                        targets.append(probe_labels(batch, anchors))
                if not xs:
                    raise ValueError("empty probe split")
                collected.append((torch.cat(xs), {k: torch.cat([y[k] for y in targets]) for k in targets[0]}))
            probes[stride] = score_probes(*collected)
        if file_sha(a.checkpoint) != digest:
            raise ValueError("source checkpoint changed during evaluation")
        report = dict(
            protocol="ours-length-v1",
            checkpoint=str(a.checkpoint),
            checkpoint_sha256=digest,
            step=step,
            standard=standard,
            extended=extended,
            probes=probes,
            lpips_status=lpips_status,
            labels="initial grasp site / heuristic phase proxies; not tracked handle annotations",
        )

        # Standard diagnostics can emit NaN for absent dynamic regions; use explicit null in the final artifact.
        def finite(node):
            import math

            if isinstance(node, dict):
                return {str(k): finite(v) for k, v in node.items()}
            return None if isinstance(node, float) and not math.isfinite(node) else node

        (out_dir / "evaluation.json").write_text(json.dumps(finite(report), indent=2, allow_nan=False))
        (out_dir / "completion.json").write_text(
            json.dumps({"status": "completed", "steps": step, "protocol": "ours-length-v1"})
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
