# Encoder KV Adapter for DiT
# Adapted from SIT (https://github.com/sihyun-yu/SiT)
# Extracts K/V from frozen encoders and projects them to DiT dimensions.

import torch
import torch.nn as nn


def zscore_norm(x, dim=-1, alpha=1.0, eps=1e-6):
    """Z-score normalization along a given dimension."""
    input_dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True)
    return ((x - alpha * mean) / (std + eps)).to(input_dtype)


class ZScoreNorm(nn.Module):
    """Z-score normalization as a module."""
    def __init__(self, dim=-1, alpha=1.0, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        return zscore_norm(x, dim=self.dim, alpha=self.alpha, eps=self.eps)


def build_kv_norm(norm_type, dim, alpha=1.0):
    """Build normalization layer for K/V projections."""
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    elif norm_type == "zscore":
        return ZScoreNorm(dim=1, alpha=alpha)  # spatial norm (matching SIT)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown kv_norm_type: {norm_type}")


def build_kv_mlp(in_dim, out_dim, hidden_dim=None):
    """Build MLP for K/V projection: in_dim -> hidden_dim -> hidden_dim -> out_dim"""
    if hidden_dim is None:
        hidden_dim = max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


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
    Project encoder K/V to DiT's dimensions (matching SIT's implementation).

    Projects at full-dimension level (enc_dim -> dit_dim), allowing cross-head
    information mixing. Then reshapes to (B, dit_heads, N, dit_head_dim).
    Normalization is applied before projection (norm -> project).
    """
    def __init__(
        self,
        enc_dim,
        enc_num_heads,
        dit_dim,
        dit_num_heads,
        num_layers=1,
        proj_type="linear",
        norm_type="layer",
        kv_zscore_alpha=1.0,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.enc_num_heads = enc_num_heads
        self.dit_dim = dit_dim
        self.dit_num_heads = dit_num_heads
        self.dit_head_dim = dit_dim // dit_num_heads
        self.proj_type = proj_type
        self.num_layers = num_layers

        # Per-layer normalization (applied before projection)
        self.k_norms = nn.ModuleList([
            build_kv_norm(norm_type, enc_dim, alpha=kv_zscore_alpha)
            for _ in range(num_layers)
        ])
        self.v_norms = nn.ModuleList([
            build_kv_norm(norm_type, enc_dim, alpha=kv_zscore_alpha)
            for _ in range(num_layers)
        ])

        # Per-layer K and V projections (full-dimension: enc_dim -> dit_dim)
        if proj_type == "linear":
            self.k_projs = nn.ModuleList([
                nn.Linear(enc_dim, dit_dim, bias=False) for _ in range(num_layers)
            ])
            self.v_projs = nn.ModuleList([
                nn.Linear(enc_dim, dit_dim, bias=False) for _ in range(num_layers)
            ])
        elif proj_type == "mlp":
            self.k_projs = nn.ModuleList([
                build_kv_mlp(enc_dim, dit_dim) for _ in range(num_layers)
            ])
            self.v_projs = nn.ModuleList([
                build_kv_mlp(enc_dim, dit_dim) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}. Choose from: linear, mlp")

    def _project_component(self, enc_tensor, norm, proj):
        """
        Project a single K or V component.

        Args:
            enc_tensor: (B, enc_heads, N, enc_head_dim)
            norm: normalization module
            proj: projection module

        Returns:
            (B, dit_heads, N, dit_head_dim)
        """
        B, H_enc, N, D_enc = enc_tensor.shape
        # Merge heads back to full dim: (B, N, enc_dim)
        flat = enc_tensor.transpose(1, 2).reshape(B, N, self.enc_dim)
        # Norm then project: (B, N, enc_dim) -> (B, N, dit_dim)
        projected = proj(norm(flat).reshape(B * N, self.enc_dim)).reshape(B, N, self.dit_dim)
        # Reshape to multi-head: (B, dit_heads, N, dit_head_dim)
        return projected.reshape(B, N, self.dit_num_heads, self.dit_head_dim).transpose(1, 2)

    def forward(self, kv_list, stage=1):
        """
        Project encoder K/V to DiT dimensions.

        Args:
            kv_list: List of (K, V) tuples from EncoderKVExtractor.
                     Each K, V: (B, enc_num_heads, N_enc, enc_head_dim)
            stage: 1 = trainable projection, 2 = detached (no gradient).

        Returns:
            List of (K_proj, V_proj) tuples.
            Each K_proj, V_proj: (B, dit_num_heads, N_enc, dit_head_dim)
        """
        projected = []
        for i, (k, v) in enumerate(kv_list):
            k_proj = self._project_component(k, self.k_norms[i], self.k_projs[i])
            v_proj = self._project_component(v, self.v_norms[i], self.v_projs[i])

            if stage == 2:
                k_proj = k_proj.detach()
                v_proj = v_proj.detach()

            projected.append((k_proj, v_proj))
        return projected
