from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_blocks import EncoderBlock, DecoderBlock


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for continuous t in [0,1],
    followed by an MLP to d_model.
    """
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float in [0,1]
        returns: (B, out_dim)
        """
        half = self.embed_dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10_000.0), half, device=device)
        )
        args = t[:, None] * freqs[None, :] * 2.0 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


@dataclass
class PolicyConfig:
    proprio_dim: int
    action_dim: int
    pred_horizon: int

    d_model: int
    dropout: float

    enc_layers: int
    enc_heads: int
    enc_ff_mult: int

    dec_layers: int
    dec_heads: int
    dec_ff_mult: int

    t_embed_dim: int
    proprio_token: bool = True


class AlohaUnleashedFlowPolicy(nn.Module):
    """
    ALOHA Unleashed-style encoder-decoder, but trained with Flow Matching.

    Observation pathway (paper):
      - per-view CNN tokens -> flatten -> append proprio token -> Transformer Encoder 

    Action pathway (paper):
      - decoder operates on action-token sequence and cross-attends to obs latents

    Here, the decoder predicts velocity v_theta(x_t, t, obs).
    """

    def __init__(self, cfg: PolicyConfig, vision: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.vision = vision

        # Project *vision token dim* -> d_model
        self.vision_to_model = nn.Linear(self.vision.out_channels, cfg.d_model)

        # Proprio projection -> one token
        self.proprio_mlp = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Positional embeddings for obs tokens
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, 4096, cfg.d_model))
        nn.init.trunc_normal_(self.obs_pos_emb, std=0.02)

        # Transformer Encoder
        self.encoder = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.enc_heads, cfg.enc_ff_mult, cfg.dropout)
            for _ in range(cfg.enc_layers)
        ])
        self.enc_ln = nn.LayerNorm(cfg.d_model)

        # Action token embedding: project action_dim -> d_model
        self.action_in = nn.Linear(cfg.action_dim, cfg.d_model)

        # Learned positional embeddings for action tokens (length = pred_horizon)
        self.act_pos_emb = nn.Parameter(torch.zeros(1, cfg.pred_horizon, cfg.d_model))
        nn.init.trunc_normal_(self.act_pos_emb, std=0.02)

        # Timestep embedding (continuous t in [0,1])
        self.t_embed = SinusoidalTimestepEmbedding(cfg.t_embed_dim, cfg.d_model)
        # Project timestep embedding to d_model if needed
        self.t_token_proj = nn.Linear(cfg.d_model, cfg.d_model) 

        # Transformer Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(cfg.d_model, cfg.dec_heads, cfg.dec_ff_mult, cfg.dropout)
            for _ in range(cfg.dec_layers)
        ])
        self.dec_ln = nn.LayerNorm(cfg.d_model)

        # Project decoder output -> velocity in action space
        self.action_out = nn.Linear(cfg.d_model, cfg.action_dim)


    def encode_obs(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        proprio: (B, P)
        returns memory: (B, N_obs, d_model)
        """
        vis_tokens = self.vision(images)              # (B, N, C_vis)
        x = self.vision_to_model(vis_tokens)          # (B, N, d_model)

        if self.cfg.proprio_token:
            p = self.proprio_mlp(proprio).unsqueeze(1)  # (B,1,d_model)
            x = torch.cat([x, p], dim=1)

        # Add learned positions
        N = x.shape[1]
        pos = self.obs_pos_emb[:, :N, :]
        x = x + pos

        # Bidirectional encoder stack
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_ln(x)
        return x

    def predict_velocity(
        self,
        x_t: torch.Tensor,            # (B, T, action_dim)
        t: torch.Tensor,              # (B,)
        obs_memory: torch.Tensor,     # (B, N, d_model)
    ) -> torch.Tensor:
        """
        Returns velocity v_theta: (B, T, action_dim)
        """
        h = self.action_in(x_t) + self.act_pos_emb     # (B,T,d_model)

        # Build timestep conditioning token (B, 1, d_model)
        te = self.t_embed(t)                         # (B, d_model)
        t_tok = self.t_token_proj(te).unsqueeze(1)   # (B, 1, d_model)

        # Cross-attention memory includes obs + timestep token
        memory = torch.cat([obs_memory, t_tok], dim=1)  # (B, N+1, d_model)

        for blk in self.decoder:
            h = blk(h, memory=memory)

        h = self.dec_ln(h)
        v = self.action_out(h)                         # (B,T,action_dim)
        return v
