# Projection Loss functions for REPA (REPresentation Alignment)
# Adapted from SIT (https://github.com/sihyun-yu/SiT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder_adapter import zscore_norm


# Registry for projection loss functions
PROJECTION_LOSSES = {}


def register_projection_loss(name):
    """Decorator to register a projection loss function."""
    def decorator(cls):
        PROJECTION_LOSSES[name] = cls
        return cls
    return decorator


def get_projection_loss(name, **kwargs):
    """Get a projection loss function by name."""
    if name not in PROJECTION_LOSSES:
        raise ValueError(f"Unknown projection loss: {name}. Available: {list(PROJECTION_LOSSES.keys())}")
    return PROJECTION_LOSSES[name](**kwargs)


@register_projection_loss("cosine")
class CosineProjectionLoss(nn.Module):
    """
    Cosine similarity projection loss from REPA paper.
    Maximizes cosine similarity between projected DiT features and encoder features.
    Matches SIT's implementation: zscore only on model output, along spatial dim (dim=1).
    """
    def __init__(self, zscore_alpha=0.6, **kwargs):
        super().__init__()
        self.zscore_alpha = zscore_alpha

    def forward(self, dit_features, enc_features):
        """
        Args:
            dit_features: Projected DiT hidden states. (B, N, D)
            enc_features: Encoder features (e.g., DINOv2). (B, N, D)

        Returns:
            Scalar loss value.
        """
        dit_features = dit_features.float()
        enc_features = enc_features.float()

        # Zscore normalize only model output, along spatial dim (matching SIT)
        dit_features = zscore_norm(dit_features, dim=1, alpha=0.6)

        # L2 normalize both for cosine similarity
        dit_features = F.normalize(dit_features, dim=-1)
        enc_features = F.normalize(enc_features, dim=-1)
        loss = -torch.mean(torch.sum(dit_features * enc_features, dim=-1))
        return loss


@register_projection_loss("mse")
class MSEProjectionLoss(nn.Module):
    """
    MSE projection loss between DiT and encoder features.
    Matches SIT: zscore only on model output, along spatial dim (dim=1).
    """
    def __init__(self, zscore_alpha=0.6, **kwargs):
        super().__init__()
        self.zscore_alpha = zscore_alpha

    def forward(self, dit_features, enc_features):
        """
        Args:
            dit_features: Projected DiT hidden states. (B, N, D)
            enc_features: Encoder features (e.g., DINOv2). (B, N, D)

        Returns:
            Scalar loss value.
        """
        dit_features = dit_features.float()
        enc_features = enc_features.float()

        # Zscore normalize only model output, along spatial dim (matching SIT)
        dit_features = zscore_norm(dit_features, dim=1, alpha=self.zscore_alpha)

        loss = F.mse_loss(dit_features, enc_features)
        return loss
