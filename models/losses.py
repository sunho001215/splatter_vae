import torch
import torch.nn.functional as F

from fused_ssim import fused_ssim


def compute_reconstruction_loss(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    ssim_weight: float = 0.2,
):
    """
    Compute combined MSE + SSIM reconstruction loss.

    predicted, ground_truth: (B,3,H,W), values in [0,1]
    ssim_weight: weight for SSIM term in [0,1]
    """
    mse_loss = F.mse_loss(predicted, ground_truth)

    if ssim_weight <= 0.0:
        return mse_loss

    ssim_map = fused_ssim(predicted, ground_truth)  # (B,H,W)
    ssim_loss = 1.0 - ssim_map.mean()

    total_loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
    return total_loss


def flatten_state(z: torch.Tensor) -> torch.Tensor:
    """Flatten compact states or token grids to [B, D] for representation losses."""
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.reshape(z.shape[0], -1)


def vicreg_variance_loss(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg variance term: keep each feature dimension above the target std."""
    z = flatten_state(z)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + float(eps))
    loss = F.relu(float(gamma) - std).mean()
    return loss, std.mean(), std.min()


def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: decorrelate dimensions within one representation."""
    z = flatten_state(z)
    if z.shape[0] <= 1 or z.shape[1] <= 1:
        return z.new_zeros(())

    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / float(z.shape[0] - 1)
    off_diag = cov.flatten()[:-1].view(z.shape[1] - 1, z.shape[1] + 1)[:, 1:].flatten()
    return off_diag.pow(2).sum() / float(z.shape[1])


def cross_covariance_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Penalize second-order dependence between two deterministic branches."""
    z_a = flatten_state(z_a)
    z_b = flatten_state(z_b)
    if z_a.shape[0] <= 1 or z_a.shape[1] == 0 or z_b.shape[1] == 0:
        return z_a.new_zeros(())

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)
    cross_cov = (z_a.T @ z_b) / float(z_a.shape[0] - 1)
    # Average over the full D_inv x D_dep matrix so the loss scale remains
    # comparable when the dependent branch dimension is reduced.
    return cross_cov.pow(2).mean()


def vicreg_pair_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    std_gamma: float = 1.0,
    eps: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Compute VICReg on a positive pair, with no projector inside the loss."""
    z_a = flatten_state(z_a)
    z_b = flatten_state(z_b)

    invariance_loss = F.mse_loss(z_a, z_b)
    var_a, std_a_mean, std_a_min = vicreg_variance_loss(z_a, gamma=std_gamma, eps=eps)
    var_b, std_b_mean, std_b_min = vicreg_variance_loss(z_b, gamma=std_gamma, eps=eps)
    variance_loss = 0.5 * (var_a + var_b)
    covariance_loss = 0.5 * (vicreg_covariance_loss(z_a) + vicreg_covariance_loss(z_b))
    loss = (
        float(sim_coeff) * invariance_loss
        + float(std_coeff) * variance_loss
        + float(cov_coeff) * covariance_loss
    )

    return {
        "loss": loss,
        "invariance_loss": invariance_loss,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "std_mean": 0.5 * (std_a_mean + std_b_mean),
        "std_min": torch.minimum(std_a_min, std_b_min),
    }


def paired_infonce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE with explicit per-query positives and negatives."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    query = F.normalize(flatten_state(query), dim=-1, eps=1.0e-6)
    positive = F.normalize(flatten_state(positive), dim=-1, eps=1.0e-6)
    negative = torch.nan_to_num(negative, nan=0.0, posinf=0.0, neginf=0.0)
    if negative.dim() == 2:
        negative = negative.unsqueeze(1)
    if negative.dim() != 3:
        raise ValueError(f"negative must have shape [B,D] or [B,M,D], got {tuple(negative.shape)}")
    negative = F.normalize(negative.reshape(negative.shape[0], negative.shape[1], -1), dim=-1, eps=1.0e-6)

    pos_logit = (query * positive).sum(dim=-1, keepdim=True)
    neg_logits = torch.einsum("bd,bmd->bm", query, negative)
    logits = torch.cat((pos_logit, neg_logits), dim=1) / float(temperature)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def dependent_view_infonce_loss(
    z_dep_i_t: torch.Tensor,
    z_dep_j_t: torch.Tensor,
    z_dep_i_tk: torch.Tensor,
    z_dep_j_tk: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrast dependent states by camera viewpoint.

    Positives are same-camera states from the same demo at a different timestep.
    Negatives are the paired different-camera states at the same timestep.
    """
    z_i_t = flatten_state(z_dep_i_t)
    z_j_t = flatten_state(z_dep_j_t)
    z_i_tk = flatten_state(z_dep_i_tk)
    z_j_tk = flatten_state(z_dep_j_tk)

    query = torch.cat((z_i_t, z_i_tk, z_j_t, z_j_tk), dim=0)
    positive = torch.cat((z_i_tk, z_i_t, z_j_tk, z_j_t), dim=0)
    negative = torch.cat((z_j_t, z_j_tk, z_i_t, z_i_tk), dim=0)
    return paired_infonce_loss(query, positive, negative, temperature=temperature)


def compute_splattervae_representation_losses(
    latents: dict[str, torch.Tensor],
    *,
    vicreg_sim_coeff: float,
    vicreg_std_coeff: float,
    vicreg_cov_coeff: float,
    vicreg_std_gamma: float,
    vicreg_eps: float,
    dep_infonce_temperature: float,
) -> dict[str, torch.Tensor]:
    """Compute SplatterVAE representation losses from deterministic latent means.

    The invariant branch receives full VICReg on same-time cross-view pairs.
    The dependent branch receives InfoNCE: same camera and different timestep is
    positive, while different camera at the same timestep is negative. Cross-
    covariance discourages shared second-order information between branches.
    """
    inv_t = vicreg_pair_loss(
        latents["z_inv_mu_i_t"],
        latents["z_inv_mu_j_t"],
        sim_coeff=vicreg_sim_coeff,
        std_coeff=vicreg_std_coeff,
        cov_coeff=vicreg_cov_coeff,
        std_gamma=vicreg_std_gamma,
        eps=vicreg_eps,
    )
    inv_tk = vicreg_pair_loss(
        latents["z_inv_mu_i_tk"],
        latents["z_inv_mu_j_tk"],
        sim_coeff=vicreg_sim_coeff,
        std_coeff=vicreg_std_coeff,
        cov_coeff=vicreg_cov_coeff,
        std_gamma=vicreg_std_gamma,
        eps=vicreg_eps,
    )

    dep_infonce_loss = dependent_view_infonce_loss(
        latents["z_dep_mu_i_t"],
        latents["z_dep_mu_j_t"],
        latents["z_dep_mu_i_tk"],
        latents["z_dep_mu_j_tk"],
        temperature=dep_infonce_temperature,
    )

    z_inv_all = torch.cat(
        (
            flatten_state(latents["z_inv_mu_i_t"]),
            flatten_state(latents["z_inv_mu_j_t"]),
            flatten_state(latents["z_inv_mu_i_tk"]),
            flatten_state(latents["z_inv_mu_j_tk"]),
        ),
        dim=0,
    )
    z_dep_all = torch.cat(
        (
            flatten_state(latents["z_dep_mu_i_t"]),
            flatten_state(latents["z_dep_mu_j_t"]),
            flatten_state(latents["z_dep_mu_i_tk"]),
            flatten_state(latents["z_dep_mu_j_tk"]),
        ),
        dim=0,
    )
    _, z_inv_std_mean, z_inv_std_min = vicreg_variance_loss(
        z_inv_all,
        gamma=vicreg_std_gamma,
        eps=vicreg_eps,
    )

    inv_vicreg_loss = 0.5 * (inv_t["loss"] + inv_tk["loss"])
    return {
        "inv_vicreg_loss": inv_vicreg_loss,
        "inv_vicreg_invariance_loss": 0.5 * (
            inv_t["invariance_loss"] + inv_tk["invariance_loss"]
        ),
        "inv_vicreg_variance_loss": 0.5 * (
            inv_t["variance_loss"] + inv_tk["variance_loss"]
        ),
        "inv_vicreg_covariance_loss": 0.5 * (
            inv_t["covariance_loss"] + inv_tk["covariance_loss"]
        ),
        "dep_infonce_loss": dep_infonce_loss,
        "cross_cov_loss": cross_covariance_loss(z_inv_all, z_dep_all),
        "z_inv_std_mean": z_inv_std_mean,
        "z_inv_std_min": z_inv_std_min,
    }
