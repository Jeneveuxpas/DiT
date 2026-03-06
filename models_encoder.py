# DiT with Encoder KV Distillation
# Adapted from SIT's encoder KV approach for DiT's DDPM framework.
# References:
#   - DiT: https://github.com/facebookresearch/DiT
#   - SIT: https://github.com/sihyun-yu/SiT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from timm.models.vision_transformer import Mlp

from models import (
    TimestepEmbedder,
    LabelEmbedder,
    FinalLayer,
    PatchEmbed,
    get_2d_sincos_pos_embed,
    modulate,
)
from encoder_adapter import EncoderKVProjection


class AttentionWithEncoderKV(nn.Module):
    """
    Multi-head self-attention with optional encoder K/V injection.

    Two-stage training:
      Stage 1: Use encoder K/V for attention (replace DiT's own K/V).
      Stage 2: Use DiT's own K/V + MSE distillation loss on attention output
               L_distill = ||SDPA(Q,K,V) - SDPA(Q,h(K*),h(V*))||_2^2
    """
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, enc_k=None, enc_v=None, stage=2):
        """
        Args:
            x: Input tensor (B, N, D).
            enc_k: Encoder key (B, num_heads, N_enc, head_dim) or None.
            enc_v: Encoder value (B, num_heads, N_enc, head_dim) or None.
            stage: 1 = use encoder KV, 2 = use own KV + distill loss.

        Returns:
            output: Attention output (B, N, D).
            distill_loss: Scalar distillation loss (0 if stage 1 or no encoder KV).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_head)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D_head)

        distill_loss = torch.tensor(0.0, device=x.device)

        has_enc_kv = enc_k is not None and enc_v is not None

        if has_enc_kv and stage == 1:
            # Stage 1: Use encoder K/V
            # Handle sequence length mismatch by interpolating encoder KV
            enc_k_use, enc_v_use = self._align_seq_len(enc_k, enc_v, N)
            attn = (q * self.scale) @ enc_k_use.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x_out = attn @ enc_v_use
        elif has_enc_kv and stage == 2:
            # Stage 2: Use own K/V, compute distillation loss on attention output
            enc_k_use, enc_v_use = self._align_seq_len(enc_k, enc_v, N)

            attn_w_own = (q * self.scale @ k.transpose(-2, -1)).softmax(dim=-1)
            x_out = attn_w_own @ v

            # o* = SDPA(Q, h(K*), h(V*))  — scaffold output, enc_k/v already no_grad
            attn_w_enc = (q.detach() * self.scale @ enc_k_use.transpose(-2, -1)).softmax(dim=-1)
            o_star = (attn_w_enc @ enc_v_use).detach()

            distill_loss = F.mse_loss(x_out, o_star)
        else:
            # No encoder KV: standard self-attention
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x_out = attn @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        return x_out, distill_loss

    def _align_seq_len(self, enc_k, enc_v, target_n):

        """Interpolate encoder KV if sequence lengths differ."""
        N_enc = enc_k.shape[2]
        if N_enc == target_n:
            return enc_k, enc_v

        # Reshape to spatial grid, interpolate, reshape back
        B, H, _, D = enc_k.shape
        h_enc = w_enc = int(math.sqrt(N_enc))
        h_tgt = w_tgt = int(math.sqrt(target_n))

        # (B*H, D, h_enc, w_enc)
        enc_k = enc_k.reshape(B * H, N_enc, D).permute(0, 2, 1).reshape(B * H, D, h_enc, w_enc)
        enc_v = enc_v.reshape(B * H, N_enc, D).permute(0, 2, 1).reshape(B * H, D, h_enc, w_enc)

        enc_k = F.interpolate(enc_k, size=(h_tgt, w_tgt), mode='bilinear', align_corners=False)
        enc_v = F.interpolate(enc_v, size=(h_tgt, w_tgt), mode='bilinear', align_corners=False)

        enc_k = enc_k.reshape(B, H, D, target_n).permute(0, 1, 3, 2)
        enc_v = enc_v.reshape(B, H, D, target_n).permute(0, 1, 3, 2)
        return enc_k, enc_v


class DiTBlockWithEncoderKV(nn.Module):
    """
    DiT block with adaLN-Zero conditioning and encoder KV support.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithEncoderKV(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, enc_k=None, enc_v=None, stage=2):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        attn_out, distill_loss = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            enc_k=enc_k, enc_v=enc_v, stage=stage
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, distill_loss


class ProjectionLayer(nn.Module):
    """
    Projection layer for REPA loss.
    Projects DiT hidden states to encoder feature dimension for alignment.
    Adapted from SIT's sit.py.
    """
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(nn.Linear(out_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class DiTWithEncoderKV(nn.Module):
    """
    Diffusion Transformer with Encoder KV Distillation.

    Extends the DiT architecture with:
    1. Encoder KV injection in attention layers (two-stage)
    2. REPA projector for representation alignment loss
    3. Side-output storage for distill_loss and zs (encoder-aligned features)

    Compatible with DiT's existing diffusion framework:
    - forward() returns (N, 2*C, H, W) tensor for diffusion training_losses()
    - Side outputs stored as model attributes (_distill_loss, _zs)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        # Encoder KV params
        enc_dim=1024,
        enc_num_heads=16,
        num_enc_kv_layers=0,
        kv_proj_type="linear",
        kv_norm_type="layer",
        # REPA params
        encoder_depth=8,
        repa_out_dim=1024,
        repa_proj_layers=2,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.encoder_depth = encoder_depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlockWithEncoderKV(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Encoder KV projection (only if we have encoder KV layers)
        self.num_enc_kv_layers = num_enc_kv_layers
        if num_enc_kv_layers > 0:
            self.kv_projection = EncoderKVProjection(
                enc_dim=enc_dim,
                enc_num_heads=enc_num_heads,
                dit_dim=hidden_size,
                dit_num_heads=num_heads,
                num_layers=num_enc_kv_layers,
                proj_type=kv_proj_type,
                norm_type=kv_norm_type,
            )
        else:
            self.kv_projection = None

        # REPA projector: projects DiT features at encoder_depth to encoder dimension
        self.repa_projector = ProjectionLayer(
            in_dim=hidden_size,
            out_dim=repa_out_dim,
            num_layers=repa_proj_layers,
        )

        # Side outputs (set during forward)
        self._distill_loss = torch.tensor(0.0)
        self._zs = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, enc_kv_list=None, stage=2):
        """
        Forward pass of DiT with Encoder KV.

        Args:
            x: (N, C, H, W) noisy latent input.
            t: (N,) diffusion timesteps.
            y: (N,) class labels.
            enc_kv_list: List of (K, V) from EncoderKVExtractor, or None.
            stage: 1 or 2 for two-stage training.

        Returns:
            output: (N, 2*C, H, W) for diffusion compatibility (epsilon + variance).

        Side effects:
            self._distill_loss: Accumulated distillation loss from attention layers.
            self._zs: Projected features at encoder_depth for REPA loss. (N, T, repa_out_dim)
        """
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb

        # Project encoder KV if provided
        # In Stage 2, kv_projection output only feeds o_star (no_grad target),
        # so wrap in no_grad to avoid unnecessary computation graph overhead.
        projected_kv = None
        if enc_kv_list is not None and self.kv_projection is not None:
            if stage == 2:
                with torch.no_grad():
                    projected_kv = self.kv_projection(enc_kv_list)
            else:
                projected_kv = self.kv_projection(enc_kv_list)

        total_distill_loss = torch.tensor(0.0, device=x.device)
        zs = None

        for i, block in enumerate(self.blocks):
            # Determine encoder K/V for this block
            enc_k, enc_v = None, None
            if projected_kv is not None and i < self.num_enc_kv_layers:
                enc_k, enc_v = projected_kv[i]

            x, distill_loss = block(x, c, enc_k=enc_k, enc_v=enc_v, stage=stage)
            total_distill_loss = total_distill_loss + distill_loss

            # Extract features at encoder_depth for REPA projector
            if i == self.encoder_depth - 1:
                zs = self.repa_projector(x)  # (N, T, repa_out_dim)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        # Store side outputs
        self._distill_loss = total_distill_loss
        self._zs = zs

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance for sampling.
        At inference, no encoder KV is needed.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                           DiT EncoderKV Configs                               #
#################################################################################

def DiT_XL_2_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8_EncoderKV(**kwargs):
    return DiTWithEncoderKV(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_EncoderKV_models = {
    'DiT-XL/2': DiT_XL_2_EncoderKV,  'DiT-XL/4': DiT_XL_4_EncoderKV,  'DiT-XL/8': DiT_XL_8_EncoderKV,
    'DiT-L/2':  DiT_L_2_EncoderKV,   'DiT-L/4':  DiT_L_4_EncoderKV,   'DiT-L/8':  DiT_L_8_EncoderKV,
    'DiT-B/2':  DiT_B_2_EncoderKV,   'DiT-B/4':  DiT_B_4_EncoderKV,   'DiT-B/8':  DiT_B_8_EncoderKV,
    'DiT-S/2':  DiT_S_2_EncoderKV,   'DiT-S/4':  DiT_S_4_EncoderKV,   'DiT-S/8':  DiT_S_8_EncoderKV,
}
