# train_reviwo_robosuite.py

import os
import sys
import yaml
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# Ensure repo root is on sys.path so absolute imports work when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# RoboSuite multi-view HDF5 dataloader
from baselines.ReViWo.dataloader import build_train_valid_loaders_robosuite

# ReViWo model
from baselines.ReViWo.ReViWo.common.models.multiview_vae import MultiViewBetaVAE

# ST transformer + codebook configs 
from models.transformer import STTransConfig       
from models.vae import CodebookConfig

# Utilities: set_random_seed + losses/similarity helpers (same as ReViWo)
from baselines.ReViWo.ReViWo.common.utils import (
    normalize_tensor,
    create_adaptive_weight_map,
    compute_similarity,
    WeightedMSELoss,
)
from utils.general_utils import set_random_seed


# ===========================================================================
# Training config (ReViWo-specific)
# ===========================================================================

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


# ===========================================================================
# Loss computation for ReViWo on RoboSuite multi-view batch
# ===========================================================================

def compute_reviwo_loss(
    model: MultiViewBetaVAE,
    batch: Dict[str, torch.Tensor],
    cfg_train: ReViWoTrainConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    ReViWo loss adapted from MultiViewViTTrainer.get_loss, but using a
    RoboSuite batch that contains ALL camera views:

        batch["images"]: (B, A, 3, H, W), in [-1, 1]
            B = batch size
            A = number of cameras / views per timestep (e.g. 6)

    We apply:
      - reconstruction loss
      - shuffled latent/view reconstruction
      - view & latent consistency losses
      - view & latent contrastive losses
      - VQ loss
    """
    model_device = next(model.parameters()).device
    assert model_device == device, "Model and device must match."

    # ----------------------------------------------------------------------
    # 1) Prepare multi-view batch: (B, A, C, H, W) -> (B*A, C, H, W)
    # ----------------------------------------------------------------------
    images = batch["images"].to(device).float()  # (B, A, 3, H, W) in [-1, 1]

    B, A, C, H, W = images.shape
    assert H == W, "ReViWo code assumes square images."

    camera_num = cfg_train.camera_num
    assert camera_num == A, (
        f"cfg_train.camera_num={camera_num} but dataset has {A} cameras. "
        "Set camera_num from the dataset in main()."
    )

    # Flatten across camera dimension → (B * A, C, H, W)
    x = images.view(B * camera_num, C, H, W).contiguous()

    # ----------------------------------------------------------------------
    # 2) Encode & decode
    # ----------------------------------------------------------------------
    z_v, view_embed_loss, z_l, latent_embed_loss, encoding_indices = model.encode(x)
    y = model.decode(z_v, z_l)  # reconstruction for each (sample, view)

    # ----------------------------------------------------------------------
    # 3) Shuffle latents across cameras / batch (same as original trainer)
    # ----------------------------------------------------------------------
    # Shuffle latent z_l along camera dimension (swap view-specific content)
    shuffle_cam_idx = torch.randperm(camera_num, device=device)
    shuffled_z_l = (
        z_l.reshape(B, camera_num, -1)[:, shuffle_cam_idx, :]
        .reshape(B * camera_num, -1, z_l.shape[-1])
    )
    y_shuffle_l = model.decode(z_v, shuffled_z_l)

    # Shuffle view-latent z_v across batch dimension (swap environment/content)
    shuffle_batch_idx = torch.randperm(B, device=device)
    shuffled_z_v = (
        z_v.reshape(B, camera_num, -1)[shuffle_batch_idx, :, :]
        .reshape(B * camera_num, -1, z_v.shape[-1])
    )
    y_shuffle_v = model.decode(shuffled_z_v, z_l)

    # Shuffle both view & latent
    y_shuffle_vl = model.decode(shuffled_z_v, shuffled_z_l)

    # ----------------------------------------------------------------------
    # 4) Normalize latents and compute contrastive + consistency losses
    # ----------------------------------------------------------------------
    normalized_z_l = normalize_tensor(z_l).reshape(B, camera_num, -1)
    normalized_z_v = normalize_tensor(z_v).reshape(B, camera_num, -1)

    latent_contrastive_loss = 0.0
    view_contrastive_loss = 0.0
    temperature = cfg_train.temperature
    lower_bound = cfg_train.lower_bound

    # ---- View Consistency (across batch, same camera index) --------------
    for j in range(camera_num):
        for i in range(B):
            # positive: same camera j, different samples i_ != i
            positive = sum(
                [
                    (
                        compute_similarity(
                            normalized_z_v[i, j],
                            normalized_z_v[i_, j],
                            dim=-1,
                            way="cosine-similarity",
                            lower_bound=lower_bound,
                        )
                        / temperature
                    ).exp()
                    for i_ in range(B)
                    if i_ != i
                ]
            )

            # negative: all other cameras for all samples
            negative_state_idxs = [idx for idx in range(camera_num) if idx != j]
            negative = (
                compute_similarity(
                    normalized_z_v[i : i + 1, j : j + 1].repeat(B, camera_num - 1, 1),
                    normalized_z_v[:, negative_state_idxs],
                    dim=-1,
                    way="cosine-similarity",
                    lower_bound=lower_bound,
                )
                / temperature
            ).exp().sum()

            view_contrastive_loss -= (positive / (positive + negative)).log()

    # ---- Latent Consistency (across cameras, same sample) -----------------
    for i in range(B):
        for j in range(camera_num):
            # positive: same sample i, other cameras j_ != j
            positive = sum(
                [
                    (
                        compute_similarity(
                            normalized_z_l[i, j],
                            normalized_z_l[i, j_],
                            dim=-1,
                            way="cosine-similarity",
                            lower_bound=lower_bound,
                        )
                        / temperature
                    ).exp()
                    for j_ in range(camera_num)
                    if j_ != j
                ]
            )

            # negative: all cameras of all OTHER samples
            negative_state_idxs = [idx for idx in range(B) if idx != i]
            negative = (
                compute_similarity(
                    normalized_z_l[i : i + 1, j : j + 1].repeat(B - 1, camera_num, 1),
                    normalized_z_l[negative_state_idxs],
                    dim=-1,
                    way="cosine-similarity",
                    lower_bound=lower_bound,
                )
                / temperature
            ).exp().sum()

            latent_contrastive_loss -= (positive / (positive + negative)).log()

    view_contrastive_loss = view_contrastive_loss / (B * camera_num)
    latent_contrastive_loss = latent_contrastive_loss / (B * camera_num)

    # Consistency L1 losses (same as original)
    latent_consistency_loss = torch.mean(
        (torch.mean(normalized_z_l, dim=1, keepdim=True) - normalized_z_l).abs()
    )
    view_consistency_loss = torch.mean(
        (torch.mean(normalized_z_v, dim=0, keepdim=True) - normalized_z_v).abs()
    )

    # ----------------------------------------------------------------------
    # 5) Reconstruction + shuffled reconstruction losses
    # ----------------------------------------------------------------------
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

    # VQ loss (view + latent)
    vq_loss = latent_embed_loss.mean() + view_embed_loss.mean()

    # ----------------------------------------------------------------------
    # 6) Total loss with coefficients
    # ----------------------------------------------------------------------
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


# ===========================================================================
# Validation logging (images + scalars) to wandb
# ===========================================================================

@torch.no_grad()
def log_validation_images_reviwo(
    model: MultiViewBetaVAE,
    valid_dataloader: DataLoader,
    cfg_train: ReViWoTrainConfig,
    device: torch.device,
    global_step: int,
    max_vis: int = 4,
):
    model.eval()

    try:
        batch = next(iter(valid_dataloader))
    except StopIteration:
        return

    # Compute same loss dict as training (just no grad)
    loss_dict = compute_reviwo_loss(model, batch, cfg_train, device)

    # Build multi-view batch for visualization:
    #   images: (B, A, 3, H, W)
    images = batch["images"].to(device)
    B, A, C, H, W = images.shape
    n_vis = min(max_vis, B)

    # Take first n_vis samples with all cameras
    x_vis = images[:n_vis]  # (n_vis, A, 3, H, W)

    vis_img_np, latent_encoding_indices, view_encoding_indices = model.visualize(x_vis)
    # vis_img_np: (H_vis, W_vis, 3) uint8
    vis_img = torch.from_numpy(vis_img_np).permute(2, 0, 1).float() / 255.0

    wandb.log(
        {
            "val/vis": wandb.Image(vis_img),
            "val/loss": loss_dict["loss"].item(),
            "val/rec_loss": loss_dict["rec_loss"].item(),
            "val/shuffled_l_rec_loss": loss_dict["shuffled_l_rec_loss"].item(),
            "val/shuffled_v_rec_loss": loss_dict["shuffled_v_rec_loss"].item(),
            "val/shuffled_vl_rec_loss": loss_dict["shuffled_vl_rec_loss"].item(),
            "val/vq_loss": loss_dict["vq_loss"].item(),
            "val/view_consistency_loss": loss_dict["view_consistency_loss"].item(),
            "val/latent_consistency_loss": loss_dict["latent_consistency_loss"].item(),
            "val/view_contrastive_loss": loss_dict["view_contrastive_loss"].item(),
            "val/latent_contrastive_loss": loss_dict["latent_contrastive_loss"].item(),
        },
        step=global_step,
    )


# ===========================================================================
# Training loop
# ===========================================================================

def train_reviwo(
    model: MultiViewBetaVAE,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    cfg_train: ReViWoTrainConfig,
    resume_ckpt: Optional[str] = None,
):
    device = torch.device(cfg_train.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)

    # Resume checkpoint
    start_epoch = 0
    global_step = 0
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        print(f"[Resume] Loading checkpoint from: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] Resuming from epoch={start_epoch}, global_step={global_step}")

    os.makedirs(cfg_train.ckpt_dir, exist_ok=True)

    epoch = start_epoch
    while True:
        for step, batch in enumerate(train_dataloader):

            if cfg_train.max_global_steps is not None and global_step >= cfg_train.max_global_steps:
                print(f"[Stop] Reached max_global_steps={cfg_train.max_global_steps}.")
                print("Training finished.")
                return

            model.train()

            # Compute losses
            loss_dict = compute_reviwo_loss(model, batch, cfg_train, device)
            total_loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # wandb logging (same style as your Splatter code)
            log_values = {
                "train/total_loss": total_loss.item(),
                "train/rec_loss": loss_dict["rec_loss"].item(),
                "train/shuffled_l_rec_loss": loss_dict["shuffled_l_rec_loss"].item(),
                "train/shuffled_v_rec_loss": loss_dict["shuffled_v_rec_loss"].item(),
                "train/shuffled_vl_rec_loss": loss_dict["shuffled_vl_rec_loss"].item(),
                "train/vq_loss": loss_dict["vq_loss"].item(),
                "train/view_consistency_loss": loss_dict["view_consistency_loss"].item(),
                "train/latent_consistency_loss": loss_dict["latent_consistency_loss"].item(),
                "train/view_contrastive_loss": loss_dict["view_contrastive_loss"].item(),
                "train/latent_contrastive_loss": loss_dict["latent_contrastive_loss"].item(),
                "train/epoch": epoch,
                "train/step_in_epoch": step,
                "train/global_step": global_step,
            }
            wandb.log(log_values, step=global_step)

            if step % cfg_train.log_every == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(rec={loss_dict['rec_loss'].item():.4f}, "
                    f"vq={loss_dict['vq_loss'].item():.4f}, "
                    f"view_con={loss_dict['view_contrastive_loss'].item():.4f}, "
                    f"latent_con={loss_dict['latent_contrastive_loss'].item():.4f})"
                )

            # Periodic validation
            if (
                cfg_train.eval_every > 0
                and global_step > 0
                and global_step % cfg_train.eval_every == 0
            ):
                log_validation_images_reviwo(
                    model=model,
                    valid_dataloader=valid_dataloader,
                    cfg_train=cfg_train,
                    device=device,
                    global_step=global_step,
                )

            # Periodic checkpoint saving
            if (
                cfg_train.save_every > 0
                and global_step > 0
                and global_step % cfg_train.save_every == 0
            ):
                ckpt_path = os.path.join(
                    cfg_train.ckpt_dir,
                    f"step_{global_step:08d}.pth",
                )
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")

            global_step += 1

        epoch += 1
        if cfg_train.max_global_steps is None and epoch >= cfg_train.num_epochs:
            print(f"[Stop] Reached num_epochs={cfg_train.num_epochs}.")
            break

    print("Training finished.")


# ===========================================================================
# main(): wire YAML config + dataloaders + model + wandb
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file for ReViWo + RoboSuite.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------------- Dataset config (same style as your Splatter code) ----
    ds_cfg = cfg.get("dataset", {})
    dataset_path = ds_cfg.get("hdf5_path", None)
    if dataset_path is None:
        raise ValueError('Config "dataset.hdf5_path" must be provided.')

    batch_size = ds_cfg.get("batch_size", 128)
    num_workers = ds_cfg.get("num_workers", 8)
    pin_memory = ds_cfg.get("pin_memory", True)
    train_ratio = float(ds_cfg.get("train_ratio", 0.90))
    seed = int(ds_cfg.get("seed", 42))
    num_episodes = ds_cfg.get("num_episodes", None)
    max_frames_per_demo = ds_cfg.get("max_frames_per_demo", None)
    min_time_gap = ds_cfg.get("min_time_gap", 25)  # unused by our dataset, but kept for config

    set_random_seed(seed)

    train_loader, valid_loader = build_train_valid_loaders_robosuite(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_ratio=train_ratio,
        seed=seed,
        num_episodes=num_episodes,
        max_frames_per_demo=max_frames_per_demo,
        min_time_gap=min_time_gap,
    )

    # Peek a batch to get resolution and number of cameras
    sample_batch = next(iter(train_loader))
    # sample_batch["images"]: (B, A, 3, H, W)
    _, A, _, H, W = sample_batch["images"].shape
    print(f"[Info] Training image resolution: H={H}, W={W}, num_cameras={A}")

    assert H == W, "ReViWo implementation assumes square images."
    img_size = H  # use this for ReViWo

    # ---------------- ReViWo model config ----------------------------------
    reviwo_cfg = cfg.get("reviwo", {})

    # Encoder / decoder configs (mirror original ReViWo YAML)
    view_enc_cfg_dict = reviwo_cfg.get("view_encoder", {})
    latent_enc_cfg_dict = reviwo_cfg.get("latent_encoder", {})
    decoder_cfg_dict = reviwo_cfg.get("decoder", {})

    view_enc_cfg = STTransConfig(**view_enc_cfg_dict)
    latent_enc_cfg = STTransConfig(**latent_enc_cfg_dict)
    decoder_cfg = STTransConfig(**decoder_cfg_dict)

    # Codebook configs (view = "view VQ", latent = "latent VQ")
    view_cb_cfg_dict = reviwo_cfg.get("view_codebook", {})
    latent_cb_cfg_dict = reviwo_cfg.get("latent_codebook", {})
    view_cb_cfg = CodebookConfig(**view_cb_cfg_dict)
    latent_cb_cfg = CodebookConfig(**latent_cb_cfg_dict)

    patch_size = reviwo_cfg.get("patch_size", 16)
    fusion_style = reviwo_cfg.get("fusion_style", "plus")
    use_latent_vq = reviwo_cfg.get("use_latent_vq", True)
    is_latent_ae = reviwo_cfg.get("is_latent_ae", False)
    use_view_vq = reviwo_cfg.get("use_view_vq", True)
    is_view_ae = reviwo_cfg.get("is_view_ae", False)

    model = MultiViewBetaVAE(
        view_encoder_config=view_enc_cfg,
        latent_encoder_config=latent_enc_cfg,
        decoder_config=decoder_cfg,
        view_cb_config=view_cb_cfg,
        latent_cb_config=latent_cb_cfg,
        img_size=img_size,
        patch_size=patch_size,
        fusion_style=fusion_style,
        use_latent_vq=use_latent_vq,
        is_latent_ae=is_latent_ae,
        use_view_vq=use_view_vq,
        is_view_ae=is_view_ae,
    )

    # ---------------- Training config --------------------------------------
    train_cfg_dict = cfg.get("train", {})
    cfg_train = ReViWoTrainConfig(**train_cfg_dict)

    # Overwrite camera_num from dataset (# of cameras in HDF5)
    cfg_train.camera_num = A

    # ---------------- wandb init -------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "ReViWo-RoboSuite")
    wandb_entity = wandb_cfg.get("entity", None)

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "num_epochs": cfg_train.num_epochs,
            "lr": cfg_train.lr,
            "device": cfg_train.device,
            "eval_every": cfg_train.eval_every,
            "save_every": cfg_train.save_every,
            "ckpt_dir": cfg_train.ckpt_dir,
            "img_size": img_size,
            "num_cameras": cfg_train.camera_num,
            "loss_form": cfg_train.loss_form,
            "vq_coef": cfg_train.vq_coef,
            "latent_consistency_coef": cfg_train.latent_consistency_coef,
            "view_consistency_coef": cfg_train.view_consistency_coef,
            "latent_contrastive_coef": cfg_train.latent_contrastive_coef,
            "view_contrastive_coef": cfg_train.view_contrastive_coef,
            "shuffled_v_coef": cfg_train.shuffled_v_coef,
            "shuffled_l_coef": cfg_train.shuffled_l_coef,
            "shuffled_vl_coef": cfg_train.shuffled_vl_coef,
            "temperature": cfg_train.temperature,
            "lower_bound": cfg_train.lower_bound,
            "reviwo": reviwo_cfg,
        },
    )
    print(f"[wandb] Logging to project: {wandb_project}")

    # ---------------- Resume from last checkpoint if requested -------------
    resume_ckpt = None
    if os.path.isdir(cfg_train.ckpt_dir) and cfg_train.resume_from_last:
        ckpt_files = [
            f for f in os.listdir(cfg_train.ckpt_dir)
            if f.endswith(".pth") and f.startswith("step_")
        ]
        if ckpt_files:
            ckpt_files.sort()
            resume_ckpt = os.path.join(cfg_train.ckpt_dir, ckpt_files[-1])
            print(f"[Resume] Found latest checkpoint: {resume_ckpt}")

    # ---------------- Run training -----------------------------------------
    train_reviwo(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        cfg_train=cfg_train,
        resume_ckpt=resume_ckpt,
    )


if __name__ == "__main__":
    main()
