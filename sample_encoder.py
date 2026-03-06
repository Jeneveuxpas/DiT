# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT with Encoder KV Distillation.
At inference time, no encoder is needed — only DiT's own K/V are used.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models_encoder import DiT_EncoderKV_models
import argparse


def find_model(model_path):
    """Load a checkpoint, handling both full checkpoints and EMA-only state dicts."""
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    return state_dict


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_EncoderKV_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        # At inference, KV projection weights are loaded but not used
        # We still need to match the checkpoint's architecture
        num_enc_kv_layers=args.num_enc_kv_layers,
        enc_dim=args.enc_dim,
        enc_num_heads=args.enc_num_heads,
        encoder_depth=args.encoder_depth,
        repa_out_dim=args.enc_dim,
    ).to(device)

    # Load checkpoint:
    assert args.ckpt is not None, "Must provide --ckpt path for DiT-EncoderKV models."
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with:
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images (no encoder needed at inference):
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample_encoder.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_EncoderKV_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a DiT-EncoderKV checkpoint from train_encoder.py.")
    # Architecture args (must match training checkpoint):
    parser.add_argument("--num-enc-kv-layers", type=int, default=4,
                        help="Number of encoder KV layers (must match training)")
    parser.add_argument("--enc-dim", type=int, default=1024,
                        help="Encoder hidden dimension (must match training)")
    parser.add_argument("--enc-num-heads", type=int, default=16,
                        help="Encoder number of attention heads (must match training)")
    parser.add_argument("--encoder-depth", type=int, default=8,
                        help="DiT layer for REPA projector (must match training)")
    args = parser.parse_args()
    main(args)
