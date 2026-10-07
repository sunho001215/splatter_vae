"""SinCro masked ViT image encoder and state encoder.

Copied verbatim from SinCro (github.com/sunho001215/sincro @ 5c7c06d373233e7bc7d7c240fdc27b212e2535f4, file
``MV_mae_encoder.py``); only the imports differ. MIT License, Copyright (c) 2025 Seungyeon Yoo. Only the definitions that the
splatter4d SinCro trainer and RL wrapper call are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

from s4d.baselines.sincro.vit_block import Transformer

def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    H, W = grid_h_size, grid_w_size

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, device, embed_dim=1024, depth=24, num_heads=16, num_view=6,
                 time_interval = None,
                 decoder_depth = 2,
                 decoder_num_heads = 2,
                 decoder_output_dim = 64,
                 batch_size = 8,
                 vit_encoder_mlp_dim = 1024,
                 vit_decoder_mlp_dim = 1024,
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_view = num_view
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_patches = int((img_size // patch_size) ** 2)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.device = device
        self.time_interval = time_interval
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_output_dim = decoder_output_dim
        self.vit_encoder_mlp_dim = vit_encoder_mlp_dim
        self.vit_decoder_mlp_dim = vit_decoder_mlp_dim
        self.pos_embed = torch.tensor(get_2d_sincos_pos_embed(self.embed_dim, int(img_size // patch_size), int(img_size // patch_size))[None], 
                                    dtype=torch.float32, device=self.device)
        self.decoder_pos_embed = torch.tensor(get_2d_sincos_pos_embed(self.embed_dim, int(img_size // patch_size), int(img_size // patch_size))[None], 
                                    dtype=torch.float32, device=self.device)
        self.cls_pos_embed = torch.tensor(np.zeros([1, self.embed_dim]), dtype=torch.float32, device=self.device)

        self.create_model()

    def random_view_masking(self, x, mask_ratio, T, ncams):
        # If T=1, random masking
        # [B, TVH'W', embed]
        N, L, D = x.shape   # L = time_interval * ncams * 36 embeded_dim

        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand([N, L], device=self.device)
        if mask_ratio == 0.0:
            # important to avoid shuffling when m == 0
            noise, indices = torch.sort(noise)

        # sort noise for each sample
        # keep small, remove large
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, axis=1)

        row_ids = torch.ones_like(ids_shuffle, device=self.device) * torch.unsqueeze(torch.arange(end=N, device=self.device), 1)
        _ids_shuffle = torch.stack([row_ids, ids_shuffle], -1)  # [N, L, 2]
        _ids_restore = torch.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        # keep the first subset
        ids_keep = _ids_shuffle[:, :len_keep]
        x_masked = x[np.arange(N)[:,None],ids_keep[:,:,1]]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.cat([torch.zeros(N, len_keep, device=self.device), torch.ones(N, L - len_keep, device=self.device)], dim=1)
        # unshuffle to get ther binary mask
        mask_cat = torch.transpose(mask[np.arange(N)[:,None],_ids_restore[:,:,1]], 0, 1)

        return x_masked, mask_cat, ids_restore

    def SinCro_image_encoder(self, x, mask_ratio, T, is_ref):
        # embed patches       
        B, Time, H, VW, C = x.shape

        assert T == Time
        x = x.reshape(B*Time, H,VW,C)
        W = self.img_size
        if W == VW:
            V = 1
        else:
            V = int(VW/W)
            x = torch.split(x, W, dim = 2) # [BT, H, W, C] * V
            x = torch.cat(x, dim = 0) # [VBT,H,W,C]
        
        if not is_ref:
            self.B, self.T, self.V = B, T, V
                
        x = x.permute(0,3,1,2) # [VBT,C,H,W]
        
        x = self.convs(x) # [VBT, embed , H', W']
        h_, w_ = x.shape[2], x.shape[3]
        self.h_, self.w_ = h_, w_
        x = x.reshape(B*T*V, self.embed_dim, -1) # [VBT, embed , H'W']
        if V!=1:
            x = torch.split(x, B*T, dim = 0) # [BT, embed, H'W'] * V
            x = torch.cat(x, dim = 2) # [BT, embed, VH'W']
        
        x = x.reshape(B,T, self.embed_dim, -1) # [B,T, embed, VH'W']
        x = x.permute(0,1,3,2).reshape(B, -1, self.embed_dim) # [B, T, VH'W',embed] -> [B, TVH'W', embed]

        # add pos embed w/o cls token
        _pos_embed = []
        for t in range(T):
            for v in range(V):
                if is_ref:
                    if T == 1:
                        t = -1
                    else:
                        assert T == len(self.encoder_time_tokens)
                cam_pos_embed = self.pos_embed
                # [1, H'W', embed]
                cam_time_token = torch.tile(self.encoder_time_tokens[t], [1, cam_pos_embed.shape[1], 1]) 
                    
                _pos_embed.append(cam_pos_embed + cam_time_token)
        
        pos_embed = torch.cat(_pos_embed, axis=1) # [1, TVH'W', embed] or [1, TVH'W', 2 * embed]
            
        x = x + pos_embed # [B, TVH'W', embed] or [B, TVH'W', 2 * embed]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_view_masking(x, mask_ratio, T, V)

        # append class token
        cls_token = self.encoder_cls_token + self.cls_pos_embed
        cls_tokens = repeat(cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1) # class token -> [B, 1+reduced length, embed] or [B, 1+reduced length, 2 * embed]

        # apply Transformer blocks
        x = self.ViT(x)
        latent = self.ViT_encoder_norm_layer(x)
        
        return latent, mask, ids_restore
        
    def SinCro_state_encoder(self, latent, ref, mask, ids_restore):
        # embed tokens
        x = self.mlp1(latent)
        ref = self.mlp1(ref)
        
        # ref = ref[:,1:,:]   # [B, ref_TVH'W', embed] when B, T=1, V=(1/3)*total_V
        ref_V = int(ref.shape[1]/(self.h_*self.w_)) # ref_V=(1/3)*total_V
        
        N = ids_restore.shape[0]
        row_ids = torch.ones_like(ids_restore, device=self.device) * torch.unsqueeze(torch.arange(end=N, device=self.device), 1)
        ids_restore = torch.stack([row_ids, ids_restore], -1)  # [N, L, 2]

        mask_tokens = torch.tile(self.mask_token, [x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_restore = x_[np.arange(N)[:,None],ids_restore[:,:,1]]
        
        camera_size = x_restore.shape[1] # TVH'W'
        x = x_restore   # [B, TVH'W', embed]

        decoder_pos_embed = []
        for t in range(self.T):
            for v in range(self.V):
                decoder_cam_pos_embed = self.decoder_pos_embed
                decoder_cam_time_token = torch.tile(self.decoder_time_tokens[t], [1, decoder_cam_pos_embed.shape[1], 1])
                decoder_img_token = torch.tile(self.decoder_img_tokens[v], [1, decoder_cam_pos_embed.shape[1], 1])
                decoder_pos_embed.append(decoder_cam_pos_embed + decoder_cam_time_token + decoder_img_token)
                
        # tokens for references
        for v in range(ref_V):
            assert (ref_V + 1) == len(self.decoder_img_tokens)
            t = self.T-1
            decoder_cam_pos_embed = self.decoder_pos_embed
            decoder_cam_time_token = torch.tile(self.decoder_time_tokens[t], [1, self.pos_embed.shape[1], 1]) # [1, H'W', embed]
            decoder_img_token = torch.tile(self.decoder_img_tokens[v+1], [1, decoder_cam_pos_embed.shape[1], 1])
            decoder_pos_embed.append(decoder_cam_pos_embed + decoder_cam_time_token + decoder_img_token)
            
        decoder_pos_embed = torch.cat(decoder_pos_embed, dim=1) # [1, TVH'W'+ref_VH'W', embed]     
        
        x = x + decoder_pos_embed[:,:(self.T*self.V*self.h_*self.w_),:]  # [B, TVH'W', Embed] 
        ref = ref + decoder_pos_embed[:,(self.T*self.V*self.h_*self.w_):,:]   # [B, ref_VH'W', embed]

        # append class token
        dec_cls_token = self.decoder_cls_token + self.cls_pos_embed
        dec_cls_tokens = repeat(dec_cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat([dec_cls_tokens, x], dim=1) # class token -> [B, 1+TVH'W', embed] or [B, 1+TVH'W', 2 * embed]
        
        x = torch.cat([x, ref], dim = 1)    # [B, TVH'W'+ref_VH'W', Embed] or [B, 1+TVH'W'+ref_VH'W', Embed] 
        x = self.ViT_CatBlock(x)
        y = x[:, 1+camera_size:]
        x = x[:, 1:1+camera_size]   # [B, TVH'W', embed_dim] 
        x = self.ViT_decoder_norm_layer(x)  # [B, TVH'W', embed_dim] 
        assert ref_V == 2
        y1 = self.ViT_decoder_norm_layer(y[:,:y.shape[1]//2])
        y2 = self.ViT_decoder_norm_layer(y[:,y.shape[1]//2:])
        assert y1.shape == y2.shape
        
        # average_right_after ViT
        vit_dec_output = x[:, :camera_size] # [B, TVH'W', embed_dim] 
        vit_dec_output = vit_dec_output.reshape(self.B, self.T, self.V, -1, self.embed_dim) # [B,T,V, H'W',embed_dim] 
        y1 = y1.reshape(self.B, 1, 1, -1, self.embed_dim)   # [B, T(1), V(1), H_ref'W_ref', embed_dim]
        y2 = y2.reshape(self.B, 1, 1, -1, self.embed_dim)
        vit_dec_output = vit_dec_output.reshape(self.B, self.T, self.V,-1) # [B,T,V,H'W'embed_dim] 
        vit_dec_output = vit_dec_output.reshape(self.B, -1, vit_dec_output.shape[-1]) # [B,TV,H'W'embed_dim] 
        y1 = y1.reshape(self.B, 1, 1, -1)
        y2 = y2.reshape(self.B, 1, 1, -1)

        # self.vit_dec_output = vit_dec_output
        latent = self.fc(vit_dec_output) # [B,TV,feat_dim]
        y1 = self.fc(y1)
        y2 = self.fc(y2)
        self.ref_feature = y1[:,0,0]   # [B, feat_dim]
        self.input_feature = latent[:,-1]    # [B, feat_dim]
        latent = latent.reshape(self.B, self.T, self.V, -1)
        latent_last_time = latent[:,-1:]    # [B, T(1), V(1), feat_dim]
        latent_last_time = torch.cat((latent_last_time, y1, y2), dim=2).mean(2, keepdim=True)   # [B, T(1), V(1), feat_dim]
        latent_last_time = self.state_mlp(latent_last_time) # [B, T(1), V(1), feat_dim]
        latent[:,-1:] = latent_last_time
        latent = latent.reshape(self.B, -1, self.decoder_output_dim)    # [B,TV,feat_dim]
            
        latent = F.normalize(latent, dim=-1)

        return latent, mask, ids_restore

    def create_model(self):
        def conv2d_size_out(size, kernel_size, padding, stride):
            return (size - kernel_size + 2*padding) // stride + 1

        size_w = self.img_size
        size_h = self.img_size

        self.conv_num = int(np.log2(self.patch_size))
        self.convs = []

        self.conv_num = 1
        self.convs.append(nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size))
        self.size_w = conv2d_size_out(size_w, kernel_size=self.patch_size, padding=0, stride=self.patch_size)
        self.size_h = conv2d_size_out(size_h, kernel_size=self.patch_size, padding=0, stride=self.patch_size)
        assert (self.size_w == int(self.img_size // self.patch_size)) and (self.size_h == int(self.img_size // self.patch_size))
        self.convs = nn.Sequential(*self.convs)

        self.ViT = Transformer(dim=self.embed_dim, depth=self.depth, heads=self.num_heads, dim_head=64, mlp_dim=self.vit_encoder_mlp_dim, dropout=0.0)
        self.ViT_encoder_norm_layer = nn.LayerNorm(self.embed_dim)

        self.encoder_time_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, self.embed_dim)) for i in range(self.time_interval)])
        self.encoder_cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.mlp1 = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, device=self.device)

        # for decoding part (despite no full decoding)
        self.ViT_CatBlock = Transformer(dim=self.embed_dim, depth=self.decoder_depth, heads=self.decoder_num_heads, dim_head=64, mlp_dim=self.vit_decoder_mlp_dim, dropout=0.0)
        self.decoder_img_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, self.embed_dim)) for i in range(self.num_view // 3 + 1)])
        self.ViT_decoder_norm_layer = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(in_features=self.embed_dim*(int(self.img_size // self.patch_size)**2), 
                            out_features=self.decoder_output_dim)
        
        self.fc_layer_norm = nn.LayerNorm(self.decoder_output_dim)
        self.fc_tanh = nn.Tanh()
        
        self.decoder_time_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, self.embed_dim)) for i in range(self.time_interval)])

        self.decoder_cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        self.state_mlp = nn.Sequential(
                        nn.Linear(in_features=self.decoder_output_dim, out_features=self.decoder_output_dim), nn.ReLU(),
                        nn.Linear(in_features=self.decoder_output_dim, out_features=self.decoder_output_dim)
                    )
