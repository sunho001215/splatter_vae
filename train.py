# train.py

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # <-- NEW

from models.vae import InvariantDependentSplatterVAE
from models.splatter import (
    VAESplatterToGaussians,
    SplatterConfig,
    SplatterDataConfig,
    SplatterModelConfig,
    render_predicted,
)
from models.losses import infonce_loss
from utils.ray_utils import (
    build_full_proj_matrix,
    camera_center_from_viewmatrix,
)

# If you want to build loaders inside main()
from dataset.dataloader import build_train_valid_loaders
from models.transformer import STTransConfig
from models.vae import CodebookConfig

# -------------------------------------------------------------------------
# Small config for training hyperparameters
# -------------------------------------------------------------------------

@dataclass
class TrainConfig:
    num_epochs: int = 50
    lr: float = 1e-4
    device: str = "cuda"
    # Loss weights
    rec_weight: float = 1.0
    vq_weight: float = 1.0
    inv_contrastive_weight: float = 1.0
    dep_contrastive_weight: float = 1.0
    # Contrastive settings
    temperature: float = 0.25
    # How often to run validation rendering (in training steps)
    eval_every: int = 1000


# -------------------------------------------------------------------------
# Helper: render validation samples and log to TensorBoard
# -------------------------------------------------------------------------

@torch.no_grad()
def log_validation_images(
    vae: InvariantDependentSplatterVAE,
    splatter_cfg: SplatterConfig,
    splatter_to_gaussians: VAESplatterToGaussians,
    valid_dataloader: DataLoader,
    device: torch.device,
    bg: torch.Tensor,
    writer: SummaryWriter,
    global_step: int,
    max_vis: int = 4,
):
    """
    Take one batch from valid_dataloader and render:

      1) rendered_i_from_i       : image_i_t → i-view
      2) rendered_j_from_i       : image_i_t → j-view
      3) rendered_i_from_shuf    : shuffled latents, rendered to i-view
      4) rendered_j_from_shuf    : shuffled latents, rendered to j-view
      5) gt_i_t                  : ground truth image_i_t
      6) gt_j_t                  : ground truth image_j_t

    All images are logged to TensorBoard as [0,1] floats.
    """
    vae.eval()
    splatter_to_gaussians.eval()

    # Get a single validation batch
    try:
        batch = next(iter(valid_dataloader))
    except StopIteration:
        return

    image_i_t  = batch["image_i_t"].to(device)   # (B,3,H,W) in [-1,1]
    image_j_t  = batch["image_j_t"].to(device)   # (B,3,H,W) in [-1,1]
    image_i_t1 = batch["image_i_t1"].to(device)  # (B,3,H,W)

    T_ij = batch["T_ij"].to(device)              # (B,4,4)
    K_i  = batch["K_i"].to(device)               # (B,3,3)
    K_j  = batch["K_j"].to(device)               # (B,3,3)

    B, _, H, W = image_i_t.shape
    n_vis = min(max_vis, B)

    # Convert GT to [0,1] for visualization
    gt_i_t_01 = (image_i_t + 1.0) * 0.5
    gt_j_t_01 = (image_j_t + 1.0) * 0.5

    # ---------------- Encode latents ----------------
    z_inv_i_t, _, z_dep_i_t, _, _ = vae.encode(
        image_i_t,
        deterministic_invariant=False,
        deterministic_dependent=False,
    )
    z_inv_j_t, _, z_dep_j_t, _, _ = vae.encode(
        image_j_t,
        deterministic_invariant=False,
        deterministic_dependent=False,
    )
    z_inv_i_t1, _, z_dep_i_t1, _, _ = vae.encode(
        image_i_t1,
        deterministic_invariant=False,
        deterministic_dependent=False,
    )

    # Small helper to render a single sample
    def render_single_view(pc_b, world_view_b, K_b):
        camera_center_b = camera_center_from_viewmatrix(world_view_b)  # (3,)
        full_proj_b = build_full_proj_matrix(
            K_b,
            H,
            W,
            splatter_cfg.data.znear,
            splatter_cfg.data.zfar,
            device=device,
        )
        out = render_predicted(
            pc=pc_b,
            world_view_transform=world_view_b,
            full_proj_transform=full_proj_b,
            camera_center=camera_center_b,
            bg_color=bg,
            cfg=splatter_cfg,
            intrinsics=K_b.unsqueeze(0),
            scaling_modifier=1.0,
            override_color=None,
        )
        return out["render"]  # (1,3,H,W)

    # =========================
    # ORIGINAL latents (from i)
    # =========================
    splatter_i_t = vae.decode(z_inv_i_t, z_dep_i_t)  # (B,C_s,H,W)

    source_cameras_view_to_world = (
        torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    )
    source_cv2wT_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .view(1, 4)
        .repeat(B, 1)
    )

    gaussian_pc = splatter_to_gaussians(
        splatter_i_t,
        source_cameras_view_to_world=source_cameras_view_to_world,
        source_cv2wT_quat=source_cv2wT_quat,
        intrinsics=K_i,
        activate_output=True,
    )

    # i_t → i-view (self) and j-view (cross)
    eye_4 = torch.eye(4, device=device)

    renders_i_from_i = []
    renders_j_from_i = []
    for b in range(n_vis):
        pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc.items()}

        # self-view
        world_view_i = eye_4
        renders_i_from_i.append(
            render_single_view(pc_b, world_view_i, K_i[b])
        )

        # cross-view
        world_view_j = T_ij[b].contiguous()
        renders_j_from_i.append(
            render_single_view(pc_b, world_view_j, K_j[b])
        )

    renders_i_from_i = torch.cat(renders_i_from_i, dim=0)  # (n_vis,3,H,W)
    renders_j_from_i = torch.cat(renders_j_from_i, dim=0)

    # =========================
    # SHUFFLED latents
    #    (z_inv_i_t, z_dep_j_t)
    # =========================
    splatter_shuffle_j = vae.decode(z_inv_i_t, z_dep_j_t)  # (B,C_s,H,W)

    source_cameras_view_to_world_j = (
        torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
    )
    source_cv2wT_quat_j = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .view(1, 4)
        .repeat(B, 1)
    )

    gaussian_pc_shuffle_j = splatter_to_gaussians(
        splatter_shuffle_j,
        source_cameras_view_to_world=source_cameras_view_to_world_j,
        source_cv2wT_quat=source_cv2wT_quat_j,
        intrinsics=K_j,
        activate_output=True,
    )

    T_ji = torch.linalg.inv(T_ij)  # (B,4,4)

    renders_j_from_shuf = []
    renders_i_from_shuf = []
    for b in range(n_vis):
        pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc_shuffle_j.items()}

        # j-view (world == cam_j, view == cam_j)
        world_view_j = eye_4
        renders_j_from_shuf.append(
            render_single_view(pc_b, world_view_j, K_j[b])
        )

        # i-view from j-world (cam_j -> cam_i)
        world_view_i_from_j = T_ji[b].contiguous()
        renders_i_from_shuf.append(
            render_single_view(pc_b, world_view_i_from_j, K_i[b])
        )

    renders_j_from_shuf = torch.cat(renders_j_from_shuf, dim=0)  # (n_vis,3,H,W)
    renders_i_from_shuf = torch.cat(renders_i_from_shuf, dim=0)

    # =========================
    # Prepare GT images (0-1)
    # =========================
    gt_i_vis = gt_i_t_01[:n_vis].detach()
    gt_j_vis = gt_j_t_01[:n_vis].detach()

    # =========================
    # Log images to TensorBoard
    # =========================
    writer.add_images("val/render_i_from_i", renders_i_from_i, global_step)
    writer.add_images("val/render_j_from_i", renders_j_from_i, global_step)
    writer.add_images("val/render_i_from_shuf", renders_i_from_shuf, global_step)
    writer.add_images("val/render_j_from_shuf", renders_j_from_shuf, global_step)
    writer.add_images("val/gt_i_t", gt_i_vis, global_step)
    writer.add_images("val/gt_j_t", gt_j_vis, global_step)
    writer.flush()


# -------------------------------------------------------------------------
# Main training loop (now with TensorBoard + validation renders)
# -------------------------------------------------------------------------

def train_splatter_vae(
    vae: InvariantDependentSplatterVAE,
    splatter_cfg: SplatterConfig,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    cfg_train: TrainConfig,
    writer: SummaryWriter,
):
    """
    Training loop:

    For each batch we now do:

      A) Cross-view reconstruction:
         - Encode image_i_t, image_j_t, image_i_t1
         - Decode image_i_t -> splatter_i_t
         - Treat camera-i coords as "world" (c2w = I)
         - Render from camera-j view using T_ij, K_j
         - Loss: rec_loss_cross

      B) Self reconstruction of image_i_t:
         - Re-use splatter_i_t and its Gaussians (world == cam_i)
         - Render back into camera-i view (view matrix = I, K_i)
         - Loss: rec_loss_self_i

      C) Shuffle loss:
         - Swap dependent embeddings between i_t and j_t:
               z_inv_i_t (content from i), z_dep_j_t (view from j)
         - Decode to splatter_shuffle_j in camera-j coordinates
           (world == cam_j, c2w = I, intrinsics K_j)
         - From these shuffled Gaussians (world == cam_j), render:
             1) j-view → j image      => rec_loss_shuffle_j
             2) i-view → i image      => rec_loss_shuffle_i_from_j
         - Combine these two into rec_loss_shuffle

      D) Contrastive losses:
         - inv_contrastive_loss over invariant latents
         - dep_contrastive_loss over dependent latents

      E) VQ losses over 3 images (i_t, j_t, i_t1)

    All scalar losses are logged to TensorBoard.
    Every cfg_train.eval_every steps, validation renders are also logged.
    """

    device = torch.device(cfg_train.device)
    vae.to(device)

    # Build Splatter -> Gaussian converter
    splatter_data_cfg = splatter_cfg.data
    splatter_to_gaussians = VAESplatterToGaussians(splatter_cfg).to(device)

    # Optimizer: VAE + splatter_to_gaussians
    params = list(vae.parameters()) + list(splatter_to_gaussians.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg_train.lr)

    # Background color (white or black)
    if splatter_data_cfg.white_background:
        bg = torch.ones(3, dtype=torch.float32, device=device)
    else:
        bg = torch.zeros(3, dtype=torch.float32, device=device)

    global_step = 0

    for epoch in range(cfg_train.num_epochs):
        vae.train()
        splatter_to_gaussians.train()

        for step, batch in enumerate(train_dataloader):
            # ---------------------------------------------------------
            # 1. Move batch to device
            # ---------------------------------------------------------
            image_i_t  = batch["image_i_t"].to(device)   # (B,3,H,W)
            image_j_t  = batch["image_j_t"].to(device)   # (B,3,H,W)
            image_i_t1 = batch["image_i_t1"].to(device)  # (B,3,H,W)

            T_ij = batch["T_ij"].to(device)              # (B,4,4) cam_i -> cam_j view
            K_i  = batch["K_i"].to(device)               # (B,3,3)
            K_j  = batch["K_j"].to(device)               # (B,3,3)

            B, _, H, W = image_i_t.shape

            # Images are assumed in [-1,1]; prepare [0,1] targets
            gt_i_t_01 = (image_i_t + 1.0) * 0.5          # (B,3,H,W)
            gt_j_t_01 = (image_j_t + 1.0) * 0.5          # (B,3,H,W)

            # ---------------------------------------------------------
            # Small helper: render a single sample from given view
            # ---------------------------------------------------------
            def render_single_view(
                pc_b: dict,
                world_view_b: torch.Tensor,  # (4,4)
                K_b: torch.Tensor,           # (3,3)
            ) -> torch.Tensor:
                """
                Render one Gaussian point cloud from a specific camera.
                Returns:
                    render: (1,3,H,W) image in [0,1]
                """
                camera_center_b = camera_center_from_viewmatrix(world_view_b)  # (3,)

                full_proj_b = build_full_proj_matrix(
                    K_b,
                    H,
                    W,
                    splatter_data_cfg.znear,
                    splatter_data_cfg.zfar,
                    device=device,
                )

                out = render_predicted(
                    pc=pc_b,
                    world_view_transform=world_view_b,
                    full_proj_transform=full_proj_b,
                    camera_center=camera_center_b,
                    bg_color=bg,
                    cfg=splatter_cfg,
                    intrinsics=K_b.unsqueeze(0),  # (1,3,3)
                    scaling_modifier=1.0,
                    override_color=None,
                )
                return out["render"]  # (1,3,H,W)

            # ---------------------------------------------------------
            # 2. Encode three images into invariant + view-dependent latents
            # ---------------------------------------------------------
            # image_i_t
            z_inv_i_t, inv_loss_i, z_dep_i_t, dep_loss_i, _ = vae.encode(
                image_i_t,
                deterministic_invariant=False,
                deterministic_dependent=False,
            )
            # image_j_t
            z_inv_j_t, inv_loss_j, z_dep_j_t, dep_loss_j, _ = vae.encode(
                image_j_t,
                deterministic_invariant=False,
                deterministic_dependent=False,
            )
            # image_i_t1
            z_inv_i_t1, inv_loss_i1, z_dep_i_t1, dep_loss_i1, _ = vae.encode(
                image_i_t1,
                deterministic_invariant=False,
                deterministic_dependent=False,
            )

            # ---------------------------------------------------------
            # 3. Decode image_i_t latents into Splatter Image (per-pixel params)
            # ---------------------------------------------------------
            splatter_i_t = vae.decode(z_inv_i_t, z_dep_i_t)  # (B, C_s, H, W)

            # ---------------------------------------------------------
            # 4. Splatter Image -> 3D Gaussians (camera i frame == world)
            # ---------------------------------------------------------
            source_cameras_view_to_world = (
                torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
            )  # (B,4,4)
            source_cv2wT_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                .view(1, 4)
                .repeat(B, 1)
            )  # (B,4)

            gaussian_pc = splatter_to_gaussians(
                splatter_i_t,
                source_cameras_view_to_world=source_cameras_view_to_world,
                source_cv2wT_quat=source_cv2wT_quat,
                intrinsics=K_i,
                activate_output=True,
            )

            # ---------------------------------------------------------
            # 5A. Cross-view reconstruction: render i_t → j_t  (original)
            # ---------------------------------------------------------
            rendered_from_j = []
            for b in range(B):
                pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc.items()}
                world_view_b = T_ij[b].contiguous()  # cam_i -> cam_j
                rendered_from_j.append(
                    render_single_view(pc_b, world_view_b, K_j[b])
                )

            rendered_from_j = torch.cat(rendered_from_j, dim=0)  # (B,3,H,W)
            rec_loss_cross = F.mse_loss(rendered_from_j, gt_j_t_01)

            # ---------------------------------------------------------
            # 5B. Self reconstruction: render i_t → i_t (same viewpoint)
            # ---------------------------------------------------------
            rendered_self_i = []
            eye_4 = torch.eye(4, device=device)

            for b in range(B):
                pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc.items()}
                world_view_i = eye_4  # world == cam_i, view == cam_i
                rendered_self_i.append(
                    render_single_view(pc_b, world_view_i, K_i[b])
                )

            rendered_self_i = torch.cat(rendered_self_i, dim=0)  # (B,3,H,W)
            rec_loss_self_i = F.mse_loss(rendered_self_i, gt_i_t_01)

            # ---------------------------------------------------------
            # 5C. Shuffle loss (z_inv_i_t, z_dep_j_t):
            #      Gaussians in camera-j coords (world == cam_j)
            #      Render:
            #         (1) j-view  → j_t
            #         (2) i-view  → i_t  via T_ji
            # ---------------------------------------------------------
            splatter_shuffle_j = vae.decode(z_inv_i_t, z_dep_j_t)  # (B, C_s, H, W)

            source_cameras_view_to_world_j = (
                torch.eye(4, device=device).view(1, 4, 4).repeat(B, 1, 1)
            )
            source_cv2wT_quat_j = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
                .view(1, 4)
                .repeat(B, 1)
            )

            gaussian_pc_shuffle_j = splatter_to_gaussians(
                splatter_shuffle_j,
                source_cameras_view_to_world=source_cameras_view_to_world_j,
                source_cv2wT_quat=source_cv2wT_quat_j,
                intrinsics=K_j,
                activate_output=True,
            )

            # (1) j-view (world == cam_j, view == cam_j)
            rendered_shuffle_j = []
            for b in range(B):
                pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc_shuffle_j.items()}
                world_view_j = eye_4
                rendered_shuffle_j.append(
                    render_single_view(pc_b, world_view_j, K_j[b])
                )

            rendered_shuffle_j = torch.cat(rendered_shuffle_j, dim=0)  # (B,3,H,W)
            rec_loss_shuffle_j = F.mse_loss(rendered_shuffle_j, gt_j_t_01)

            # (2) i-view from cam_j world: T_ji = inv(T_ij)
            T_ji = torch.linalg.inv(T_ij)  # (B,4,4)

            rendered_shuffle_i_from_j = []
            for b in range(B):
                pc_b = {k: v[b:b+1].contiguous() for k, v in gaussian_pc_shuffle_j.items()}
                world_view_i_from_j = T_ji[b].contiguous()
                rendered_shuffle_i_from_j.append(
                    render_single_view(pc_b, world_view_i_from_j, K_i[b])
                )

            rendered_shuffle_i_from_j = torch.cat(
                rendered_shuffle_i_from_j, dim=0
            )  # (B,3,H,W)
            rec_loss_shuffle_i_from_j = F.mse_loss(
                rendered_shuffle_i_from_j, gt_i_t_01
            )

            rec_loss_shuffle = 0.5 * (rec_loss_shuffle_j + rec_loss_shuffle_i_from_j)

            # ---------------------------------------------------------
            # Combine reconstruction losses
            # ---------------------------------------------------------
            rec_loss = (rec_loss_cross + rec_loss_self_i + rec_loss_shuffle) / 3.0

            # ---------------------------------------------------------
            # 6. Contrastive losses (view-invariant + view-dependent)
            # ---------------------------------------------------------
            inv_neg_keys = torch.cat(
                [
                    z_inv_i_t1.view(B, -1).unsqueeze(1),
                    z_dep_i_t.view(B, -1).unsqueeze(1),
                    z_dep_j_t.view(B, -1).unsqueeze(1),
                ],
                dim=1,
            )
            inv_contrastive_loss = infonce_loss(
                query=z_inv_i_t.view(B, -1),
                positive_keys=z_inv_j_t.view(B, -1),
                negative_keys=inv_neg_keys,
                temperature=cfg_train.temperature,
                negative_mode="mixed",
            )

            dep_contrastive_loss = infonce_loss(
                query=z_dep_i_t.view(B, -1),
                positive_keys=z_dep_i_t1.view(B, -1),
                negative_keys=z_dep_j_t.view(B, -1).unsqueeze(1),
                temperature=cfg_train.temperature,
                negative_mode="paired",
            )

            # ---------------------------------------------------------
            # 7. VQ losses from three encodings
            # ---------------------------------------------------------
            inv_vq_loss = (inv_loss_i + inv_loss_j + inv_loss_i1) / 3.0
            dep_vq_loss = (dep_loss_i + dep_loss_j + dep_loss_i1) / 3.0
            vq_loss = inv_vq_loss + dep_vq_loss

            # ---------------------------------------------------------
            # 8. Total loss: reconstruction + VQ + contrastive
            # ---------------------------------------------------------
            total_loss = (
                cfg_train.rec_weight * rec_loss
                + cfg_train.vq_weight * vq_loss
                + cfg_train.inv_contrastive_weight * inv_contrastive_loss
                + cfg_train.dep_contrastive_weight * dep_contrastive_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---------------------------------------------------------
            # 9. TensorBoard logging (scalars)
            # ---------------------------------------------------------
            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("train/rec_loss", rec_loss.item(), global_step)
            writer.add_scalar("train/rec_cross", rec_loss_cross.item(), global_step)
            writer.add_scalar("train/rec_self_i", rec_loss_self_i.item(), global_step)
            writer.add_scalar("train/rec_shuffle", rec_loss_shuffle.item(), global_step)
            writer.add_scalar("train/rec_shuffle_j", rec_loss_shuffle_j.item(), global_step)
            writer.add_scalar(
                "train/rec_shuffle_i_from_j",
                rec_loss_shuffle_i_from_j.item(),
                global_step,
            )
            writer.add_scalar("train/vq_loss", vq_loss.item(), global_step)
            writer.add_scalar("train/inv_vq_loss", inv_vq_loss.item(), global_step)
            writer.add_scalar("train/dep_vq_loss", dep_vq_loss.item(), global_step)
            writer.add_scalar(
                "train/inv_contrastive_loss",
                inv_contrastive_loss.item(),
                global_step,
            )
            writer.add_scalar(
                "train/dep_contrastive_loss",
                dep_contrastive_loss.item(),
                global_step,
            )

            if step % 50 == 0:
                print(
                    f"[Epoch {epoch+1} | Step {step} | Global {global_step}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(rec={rec_loss.item():.4f}, "
                    f"vq={vq_loss.item():.4f}, "
                    f"inv_con={inv_contrastive_loss.item():.4f}, "
                    f"dep_con={dep_contrastive_loss.item():.4f}, "
                    f"rec_cross={rec_loss_cross.item():.4f}, "
                    f"rec_self_i={rec_loss_self_i.item():.4f}, "
                    f"rec_shuffle_j={rec_loss_shuffle_j.item():.4f}, "
                    f"rec_shuffle_i_from_j={rec_loss_shuffle_i_from_j.item():.4f})"
                )

            # ---------------------------------------------------------
            # 10. Periodic validation renders
            # ---------------------------------------------------------
            if (
                cfg_train.eval_every > 0
                and global_step > 0
                and global_step % cfg_train.eval_every == 0
            ):
                log_validation_images(
                    vae=vae,
                    splatter_cfg=splatter_cfg,
                    splatter_to_gaussians=splatter_to_gaussians,
                    valid_dataloader=valid_dataloader,
                    device=device,
                    bg=bg,
                    writer=writer,
                    global_step=global_step,
                )

            global_step += 1

    print("Training finished.")


# -------------------------------------------------------------------------
# main(): wiring everything together
# -------------------------------------------------------------------------

def main():
    """
    Entry point for training the splatter VAE.

    - Builds train/valid dataloaders from manifests + cameras.json
    - Configures Splatter (camera / Gaussian) parameters
    - Instantiates a ReViWo-style InvariantDependentSplatterVAE
      whose decoder outputs a Splatter Image instead of RGB
    - Runs training with TensorBoard logging
    """
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, default="train.json")
    parser.add_argument("--valid_manifest", type=str, default="valid.json")
    parser.add_argument("--cameras_json", type=str, default="cameras.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="./runs/exp_splatter_vae")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=1000)
    args = parser.parse_args()

    # -------------------------------------------------
    # 1) Build train / valid dataloaders
    # -------------------------------------------------
    train_manifest_path = (
        os.path.join(args.dataset_root, args.train_manifest)
        if not os.path.isabs(args.train_manifest) else args.train_manifest
    )
    valid_manifest_path = (
        os.path.join(args.dataset_root, args.valid_manifest)
        if not os.path.isabs(args.valid_manifest) else args.valid_manifest
    )
    cameras_json_path = (
        os.path.join(args.dataset_root, args.cameras_json)
        if not os.path.isabs(args.cameras_json) else args.cameras_json
    )

    train_loader, valid_loader = build_train_valid_loaders(
        dataset_root=args.dataset_root,
        train_manifest=train_manifest_path,
        valid_manifest=valid_manifest_path,
        cameras_json_path=cameras_json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Peek a batch to get input resolution (H, W)
    sample_batch = next(iter(train_loader))
    _, _, H, W = sample_batch["image_i_t"].shape
    print(f"[Info] Training image resolution: H={H}, W={W}")

    # -------------------------------------------------
    # 2) Build SplatterConfig (camera / Gaussian setup)
    # -------------------------------------------------
    splatter_data_cfg = SplatterDataConfig(
        znear=0.1,
        zfar=2.0,
        white_background=False,
        inverted_x=False,
        inverted_y=False,
        category="generic",
    )
    # Attach resolution attributes needed by VAESplatterToGaussians
    splatter_data_cfg.img_height = H
    splatter_data_cfg.img_width = W

    splatter_model_cfg = SplatterModelConfig(
        max_sh_degree=1,
        isotropic=True,
        depth_scale=1.0,
        depth_bias=0.0,
        xyz_scale=1.0,
        xyz_bias=0.0,
        opacity_scale=1.0,
        opacity_bias=0.0,
        scale_scale=1.0,
        scale_bias=1.0,
    )

    splatter_cfg = SplatterConfig(
        data=splatter_data_cfg,
        model=splatter_model_cfg,
    )

    # Number of channels required by the splatter decoder output
    tmp_converter = VAESplatterToGaussians(splatter_cfg)
    splatter_channels = tmp_converter.num_splatter_channels()
    del tmp_converter

    # -------------------------------------------------
    # 3) Build ReViWo-style VAE
    #    - 8x8 tokens (like original) for 240x320
    # -------------------------------------------------
    # For 240x320, we want 8×8 tokens, so:
    #   patch_h = H / 8 = 30
    #   patch_w = W / 8 = 40
    # More generally, we enforce an 8×8 grid if divisible.
    assert H % 8 == 0 and W % 8 == 0, \
        f"Expected H,W divisible by 8 (for 8x8 tokens), got H={H}, W={W}"

    patch_size = (H // 8, W // 8)  # (30, 40) for 240x320

    # Base transformer configs (ReViWo-ish)
    base_trans_cfg = STTransConfig(
        block_size=4 * 4,        # will be overridden to n_tokens_per_frame
        n_tokens_per_frame=4 * 4,
        n_layer=8,
        n_head=8,
        n_embed=256,
        dropout=0.1,
        bias=False,
        mask_rate=None,
    )

    invariant_encoder_config = base_trans_cfg
    dependent_encoder_config = base_trans_cfg
    decoder_config = base_trans_cfg

    inv_cb_cfg = CodebookConfig()   # K=512, D=64, beta=0.25
    dep_cb_cfg = CodebookConfig()   # same size for dependent branch

    vae = InvariantDependentSplatterVAE(
        invariant_encoder_config=invariant_encoder_config,
        dependent_encoder_config=dependent_encoder_config,
        decoder_config=decoder_config,
        invariant_cb_config=inv_cb_cfg,
        dependent_cb_config=dep_cb_cfg,
        img_height=H,
        img_width=W,
        patch_size=patch_size,
        splatter_channels=splatter_channels,
        fusion_style="cat",      # concatenate z_inv and z_dep (ReViWo style)
        use_dependent_vq=False,  # dependent branch: simple AE head
        is_dependent_ae=True,    # deterministic dependent latent
        use_invariant_vq=True,   # invariant branch uses VQ
        is_invariant_ae=True,    # deterministic invariant latent (VQ-AE)
    )

    # IMPORTANT: make sure VectorQuantizer is not trying to KMeans-init
    # with too few samples (see section 1.2)
    # (If you changed init_kmeans in the VAE as shown above, you can ignore this.)

    # -------------------------------------------------
    # 4) Training config + TensorBoard writer
    # -------------------------------------------------
    cfg_train = TrainConfig(
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        eval_every=args.eval_every,
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"[TensorBoard] Logging to: {args.log_dir}")

    # -------------------------------------------------
    # 5) Run training
    # -------------------------------------------------
    train_splatter_vae(
        vae=vae,
        splatter_cfg=splatter_cfg,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        cfg_train=cfg_train,
        writer=writer,
    )

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
