# Encoder KV Adapter for DiT
# Adapted from SIT (https://github.com/sihyun-yu/SiT)
# Extracts K/V from frozen encoders and projects them to DiT dimensions.

import torch
import torch.nn as nn
import math


def zscore_norm(x, dim=-1, eps=1e-6):
    """Z-score normalization along a given dimension."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return (x - mean) / (std + eps)


class ZScoreNorm(nn.Module):
    """Z-score normalization as a module."""
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return zscore_norm(x, dim=self.dim, eps=self.eps)


def build_kv_norm(norm_type, dim):
    """Build normalization layer for K/V projections."""
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    elif norm_type == "zscore":
        return ZScoreNorm(dim=-1)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown kv_norm_type: {norm_type}")


def build_kv_mlp(in_dim, out_dim, proj_type="linear"):
    """Build projection MLP for K/V."""
    if proj_type == "linear":
        return nn.Linear(in_dim, out_dim)
    elif proj_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    else:
        raise ValueError(f"Unknown kv_proj_type: {proj_type}")


class EncoderKVExtractor(nn.Module):
    """
    Extract Q/K/V from frozen encoder (e.g., DINOv2) attention layers via hooks.

    Registers forward hooks on specified encoder layers to capture the K/V
    projections from each attention head during the forward pass.
    """
    def __init__(self, encoder, layer_indices):
        """
        Args:
            encoder: Frozen encoder model (e.g., DINOv2 from timm).
            layer_indices: List of layer indices to extract K/V from.
        """
        super().__init__()
        self.encoder = encoder
        self.layer_indices = layer_indices
        self._kv_cache = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on encoder attention layers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        blocks = self.encoder.blocks
        for idx in self.layer_indices:
            block = blocks[idx]
            hook = block.attn.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        """Create a hook function that captures K/V from an attention layer."""
        num_prefix = getattr(self.encoder, 'num_prefix_tokens', 0)

        def hook_fn(module, input, output):
            # timm's Attention module: input[0] is x after norm
            x = input[0]
            B, N, C = x.shape
            # Compute QKV using the attention module's qkv projection
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
            q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)
            # Strip prefix tokens (e.g. CLS) to keep only patch tokens
            self._kv_cache[layer_idx] = (k[:, :, num_prefix:, :].detach(),
                                         v[:, :, num_prefix:, :].detach())
        return hook_fn

    @torch.no_grad()
    def forward(self, x):
        """
        Run encoder forward pass and return extracted K/V pairs.

        Args:
            x: Input images preprocessed for the encoder. (B, 3, H, W)

        Returns:
            List of (K, V) tuples, one per layer index.
            Each K, V has shape (B, num_heads, N, head_dim).
        """
        self._kv_cache = {}
        _ = self.encoder(x)
        kv_list = []
        for idx in self.layer_indices:
            if idx in self._kv_cache:
                kv_list.append(self._kv_cache[idx])
            else:
                raise RuntimeError(f"K/V not captured for layer {idx}")
        self._kv_cache = {}
        return kv_list

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class EncoderKVProjection(nn.Module):
    """
    Project encoder K/V to DiT's dimensions.

    The encoder may have different hidden_size and num_heads than DiT,
    so we need linear projections to align them.
    """
    def __init__(
        self,
        enc_dim,
        enc_num_heads,
        dit_dim,
        dit_num_heads,
        num_layers,
        proj_type="linear",
        norm_type="layer",
    ):
        """
        Args:
            enc_dim: Encoder hidden dimension.
            enc_num_heads: Number of attention heads in encoder.
            dit_dim: DiT hidden dimension.
            dit_num_heads: Number of attention heads in DiT.
            num_layers: Number of layers to project.
            proj_type: Type of projection ("linear" or "mlp").
            norm_type: Type of normalization ("layer", "zscore", or "none").
        """
        super().__init__()
        self.enc_dim = enc_dim
        self.enc_num_heads = enc_num_heads
        self.dit_dim = dit_dim
        self.dit_num_heads = dit_num_heads
        enc_head_dim = enc_dim // enc_num_heads
        dit_head_dim = dit_dim // dit_num_heads

        # Per-layer K and V projections
        self.k_projs = nn.ModuleList([
            build_kv_mlp(enc_head_dim, dit_head_dim, proj_type)
            for _ in range(num_layers)
        ])
        self.v_projs = nn.ModuleList([
            build_kv_mlp(enc_head_dim, dit_head_dim, proj_type)
            for _ in range(num_layers)
        ])
        # Per-layer normalization
        self.k_norms = nn.ModuleList([
            build_kv_norm(norm_type, dit_head_dim)
            for _ in range(num_layers)
        ])
        self.v_norms = nn.ModuleList([
            build_kv_norm(norm_type, dit_head_dim)
            for _ in range(num_layers)
        ])

    def forward(self, kv_list):
        """
        Project encoder K/V to DiT dimensions.

        Args:
            kv_list: List of (K, V) tuples from EncoderKVExtractor.
                     Each K, V: (B, enc_num_heads, N_enc, enc_head_dim)

        Returns:
            List of (K_proj, V_proj) tuples.
            Each K_proj, V_proj: (B, dit_num_heads, N_enc, dit_head_dim)
        """
        projected = []
        for i, (k, v) in enumerate(kv_list):
            B, H_enc, N, D_enc = k.shape
            # Merge heads into batch for projection: (B*H_enc, N, D_enc)
            k_flat = k.reshape(B * H_enc, N, D_enc)
            v_flat = v.reshape(B * H_enc, N, D_enc)

            # Project to dit head dim
            k_proj = self.k_projs[i](k_flat)  # (B*H_enc, N, dit_head_dim)
            v_proj = self.v_projs[i](v_flat)

            # Normalize
            k_proj = self.k_norms[i](k_proj)
            v_proj = self.v_norms[i](v_proj)

            dit_head_dim = k_proj.shape[-1]

            # Reshape: if enc_num_heads != dit_num_heads, we average/repeat heads
            k_proj = k_proj.reshape(B, H_enc, N, dit_head_dim)
            v_proj = v_proj.reshape(B, H_enc, N, dit_head_dim)

            if H_enc != self.dit_num_heads:
                # Interpolate heads via repeat + reshape
                # Simple approach: repeat and truncate/average
                if H_enc < self.dit_num_heads:
                    repeat_factor = math.ceil(self.dit_num_heads / H_enc)
                    k_proj = k_proj.repeat(1, repeat_factor, 1, 1)[:, :self.dit_num_heads]
                    v_proj = v_proj.repeat(1, repeat_factor, 1, 1)[:, :self.dit_num_heads]
                else:
                    # Average groups of encoder heads
                    group_size = H_enc // self.dit_num_heads
                    k_proj = k_proj[:, :self.dit_num_heads * group_size].reshape(
                        B, self.dit_num_heads, group_size, N, dit_head_dim
                    ).mean(dim=2)
                    v_proj = v_proj[:, :self.dit_num_heads * group_size].reshape(
                        B, self.dit_num_heads, group_size, N, dit_head_dim
                    ).mean(dim=2)

            projected.append((k_proj, v_proj))
        return projected
