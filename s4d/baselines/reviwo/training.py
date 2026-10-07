"""ReViWo training loss on a multi-view batch.

Copied verbatim from the reference repository (sunho001215/splatter_vae @ c0abf56, ``baselines/ReViWo/train.py``),
which mirrors the original ``MultiViewViTTrainer`` losses; only imports differ. The splatter4d entry point is
``scripts/train_reviwo.py``.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from s4d.baselines.reviwo.multiview_vae import MultiViewBetaVAE
from s4d.baselines.reviwo.utils import compute_similarity, create_adaptive_weight_map, normalize_tensor, WeightedMSELoss

@dataclass
class ReViWoTrainConfig:
    # Stop conditions
    num_epochs: int = 50
    max_global_steps: Optional[int] = None

    # Optimizer
    lr: float = 1e-4
    device: str = "cuda"

    # Basic reconstruction loss choice: "MAE", "MSE", or "Weighted_MSE"
    loss_form: str = "MSE"

    # Coefficients (mirror MultiViewViTTrainer)
    vq_coef: float = 1.0
    latent_consistency_coef: float = 1.0
    view_consistency_coef: float = 1.0
    latent_contrastive_coef: float = 1.0
    view_contrastive_coef: float = 1.0

    # Shuffle reconstruction coefficients
    shuffled_v_coef: float = 1.0
    shuffled_l_coef: float = 1.0
    shuffled_vl_coef: float = 1.0

    # Contrastive temperature + similarity lower bound
    temperature: float = 0.25
    lower_bound: float = 0.9

    # Logging / eval / saving
    eval_every: int = 1000   # steps
    save_every: int = 5000   # steps
    log_every: int = 50      # stdout print frequency
    ckpt_dir: str = "./checkpoints_reviwo"
    resume_from_last: bool = False

    # Multi-view setup: will be overwritten from dataset (# of cameras)
    camera_num: int = 2


def compute_reviwo_loss(
    model: MultiViewBetaVAE,
    batch: Dict[str, torch.Tensor],
    cfg_train: ReViWoTrainConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model_device = next(model.parameters()).device
    assert model_device == device, "Model and device must match."

    # non_blocking=True because DataLoader already uses pin_memory=True
    images = batch["images"].to(device, non_blocking=True).float()  # (B, A, 3, H, W)

    B, A, C, H, W = images.shape
    assert H == W, "ReViWo code assumes square images."

    camera_num = cfg_train.camera_num
    assert camera_num == A, (
        f"cfg_train.camera_num={camera_num} but dataset has {A} cameras. "
        "Set camera_num from the dataset in main()."
    )

    x = images.view(B * camera_num, C, H, W).contiguous()

    z_v, view_embed_loss, z_l, latent_embed_loss, encoding_indices = model.encode(x)
    y = model.decode(z_v, z_l)

    shuffle_cam_idx = torch.randperm(camera_num, device=device)
    shuffled_z_l = (
        z_l.reshape(B, camera_num, -1)[:, shuffle_cam_idx, :]
        .reshape(B * camera_num, -1, z_l.shape[-1])
    )
    y_shuffle_l = model.decode(z_v, shuffled_z_l)

    shuffle_batch_idx = torch.randperm(B, device=device)
    shuffled_z_v = (
        z_v.reshape(B, camera_num, -1)[shuffle_batch_idx, :, :]
        .reshape(B * camera_num, -1, z_v.shape[-1])
    )
    y_shuffle_v = model.decode(shuffled_z_v, z_l)
    y_shuffle_vl = model.decode(shuffled_z_v, shuffled_z_l)

    normalized_z_l = normalize_tensor(z_l).reshape(B, camera_num, -1)
    normalized_z_v = normalize_tensor(z_v).reshape(B, camera_num, -1)

    temperature = cfg_train.temperature
    lower_bound = cfg_train.lower_bound
    eps = 1e-12

    # ----------------------------------------------------------------------
    # Vectorized contrastive losses
    # ----------------------------------------------------------------------
    same_batch = torch.eye(B, device=device, dtype=torch.bool)
    same_cam = torch.eye(camera_num, device=device, dtype=torch.bool)

    # ---- View contrastive: positives = same camera, different sample
    #                       negatives = all other cameras, all samples
    view_sim = compute_similarity(
        normalized_z_v[:, :, None, None, :],   # (B, A, 1, 1, D)
        normalized_z_v[None, None, :, :, :],   # (1, 1, B, A, D)
        dim=-1,
        way="cosine-similarity",
        lower_bound=lower_bound,
    )  # -> (B, A, B, A)

    view_exp = (view_sim / temperature).exp()

    view_pos_mask = (~same_batch)[:, None, :, None] & same_cam[None, :, None, :]
    view_neg_mask = (~same_cam)[None, :, None, :]

    view_pos = view_exp.masked_fill(~view_pos_mask, 0.0).sum(dim=(2, 3))
    view_neg = view_exp.masked_fill(~view_neg_mask, 0.0).sum(dim=(2, 3))

    view_contrastive_loss = -torch.log(
        view_pos.clamp_min(eps) / (view_pos + view_neg).clamp_min(eps)
    ).mean()

    # ---- Latent contrastive: positives = same sample, different camera
    #                         negatives = all cameras from other samples
    latent_sim = compute_similarity(
        normalized_z_l[:, :, None, None, :],   # (B, A, 1, 1, D)
        normalized_z_l[None, None, :, :, :],   # (1, 1, B, A, D)
        dim=-1,
        way="cosine-similarity",
        lower_bound=lower_bound,
    )  # -> (B, A, B, A)

    latent_exp = (latent_sim / temperature).exp()

    latent_pos_mask = same_batch[:, None, :, None] & (~same_cam)[None, :, None, :]
    latent_neg_mask = (~same_batch)[:, None, :, None]

    latent_pos = latent_exp.masked_fill(~latent_pos_mask, 0.0).sum(dim=(2, 3))
    latent_neg = latent_exp.masked_fill(~latent_neg_mask, 0.0).sum(dim=(2, 3))

    latent_contrastive_loss = -torch.log(
        latent_pos.clamp_min(eps) / (latent_pos + latent_neg).clamp_min(eps)
    ).mean()

    # Consistency losses
    latent_consistency_loss = torch.mean(
        (torch.mean(normalized_z_l, dim=1, keepdim=True) - normalized_z_l).abs()
    )
    view_consistency_loss = torch.mean(
        (torch.mean(normalized_z_v, dim=0, keepdim=True) - normalized_z_v).abs()
    )

    # Reconstruction losses
    if cfg_train.loss_form == "MAE":
        rec_loss = torch.abs(x - y).mean()
        shuffled_l_rec_loss = torch.abs(x - y_shuffle_l).mean()
        shuffled_v_rec_loss = torch.abs(x - y_shuffle_v).mean()
        shuffled_vl_rec_loss = torch.abs(x - y_shuffle_vl).mean()
    elif cfg_train.loss_form == "MSE":
        rec_loss = torch.mean((x - y) ** 2)
        shuffled_l_rec_loss = torch.mean((x - y_shuffle_l) ** 2)
        shuffled_v_rec_loss = torch.mean((x - y_shuffle_v) ** 2)
        shuffled_vl_rec_loss = torch.mean((x - y_shuffle_vl) ** 2)
    elif cfg_train.loss_form == "Weighted_MSE":
        criterion = WeightedMSELoss()
        rec_loss = torch.mean((x - y) ** 2)
        shuffled_l_rec_loss = torch.mean((x - y_shuffle_l) ** 2)
        shuffled_v_rec_loss = torch.mean((x - y_shuffle_v) ** 2)
        weight_map_shuffle_vl = create_adaptive_weight_map(x, y_shuffle_vl)
        shuffled_vl_rec_loss = criterion(x, y_shuffle_vl, weight_map_shuffle_vl)
    else:
        raise NotImplementedError(f"Unknown loss_form: {cfg_train.loss_form}")

    vq_loss = latent_embed_loss.mean() + view_embed_loss.mean()

    loss = (
        rec_loss
        + cfg_train.shuffled_l_coef * shuffled_l_rec_loss
        + cfg_train.shuffled_v_coef * shuffled_v_rec_loss
        + cfg_train.shuffled_vl_coef * shuffled_vl_rec_loss
        + cfg_train.vq_coef * vq_loss
        + cfg_train.latent_consistency_coef * latent_consistency_loss
        + cfg_train.view_consistency_coef * view_consistency_loss
        + cfg_train.latent_contrastive_coef * latent_contrastive_loss
        + cfg_train.view_contrastive_coef * view_contrastive_loss
    )

    return {
        "loss": loss,
        "rec_loss": rec_loss,
        "shuffled_l_rec_loss": shuffled_l_rec_loss,
        "shuffled_v_rec_loss": shuffled_v_rec_loss,
        "shuffled_vl_rec_loss": shuffled_vl_rec_loss,
        "vq_loss": vq_loss,
        "view_consistency_loss": view_consistency_loss,
        "latent_consistency_loss": latent_consistency_loss,
        "view_contrastive_loss": view_contrastive_loss,
        "latent_contrastive_loss": latent_contrastive_loss,
    }

