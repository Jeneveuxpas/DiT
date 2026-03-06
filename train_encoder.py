# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for DiT with Encoder KV Distillation.
Based on DiT's train.py with additions for:
  - Frozen DINOv2 encoder for K/V extraction and REPA features
  - Two-stage training (stage 1: encoder KV, stage 2: own KV + distillation)
  - REPA projection loss for representation alignment
  - SIT-compatible dataset: precomputed VAE latents + raw images
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models_encoder import DiT_EncoderKV_models
from encoder_adapter import EncoderKVExtractor
from projection_loss import get_projection_loss
from diffusion import create_diffusion

try:
    from datasets import load_from_disk
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(image_arr, image_size):
    """
    Center cropping implementation from ADM.
    Takes a numpy array, returns a numpy array.
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """
    Sample from VAE posterior given precomputed (mean, std) latents.
    Same as SIT's sample_posterior.
    """
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z


ENC_TYPE_ALIASES = {
    # SIT-style shorthand -> timm model name
    "dinov2-s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2-b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-l": "vit_large_patch14_dinov2.lvd142m",
    "dinov2-g": "vit_giant_patch14_dinov2.lvd142m",
}


def build_encoder(enc_type, device, img_size=224):
    """
    Build frozen encoder (DINOv2) via timm.
    Supports SIT-style shorthand (e.g. 'dinov2-b') or full timm names.

    Returns:
        encoder: Frozen encoder model.
        enc_dim: Encoder hidden dimension.
        enc_num_heads: Number of attention heads in encoder.
    """
    timm_name = ENC_TYPE_ALIASES.get(enc_type, enc_type)
    encoder = timm.create_model(timm_name, pretrained=True, img_size=img_size)
    encoder = encoder.to(device)
    encoder.eval()
    requires_grad(encoder, False)

    enc_dim = encoder.embed_dim
    enc_num_heads = encoder.blocks[0].attn.num_heads

    return encoder, enc_dim, enc_num_heads


def get_enc_layer_indices(layer_indices_str, num_kv_layers):
    """Parse encoder layer indices (1-based, like SIT) and convert to 0-based."""
    if layer_indices_str:
        return [int(x) - 1 for x in layer_indices_str.split(",")]
    return list(range(num_kv_layers))


def parse_layer_indices(indices_str):
    """Parse DiT layer indices (1-based, like SIT) and convert to 0-based. Returns None if not set."""
    if indices_str:
        return [int(x) - 1 for x in indices_str.split(",")]
    return None


#################################################################################
#              SIT-compatible Datasets (precomputed latents + raw images)       #
#################################################################################

class HFImgLatentDataset(Dataset):
    """
    HuggingFace-based dataset with both raw images and precomputed VAE latents.
    Same format as SIT's dataset.
    Returns: (raw_image [0-255 uint8 CHW], latent_moments, label)
    """
    PRECOMPUTED = ["sdvae-ft-mse-f8d4"]

    def __init__(self, vae_name, data_dir, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        split_str = "val" if split == "val" else ""
        self.img_dataset = load_from_disk(os.path.join(data_dir, "imagenet-latents-images", split_str))
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", split_str))
        assert len(self.img_dataset) == len(self.latent_dataset), \
            "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)  # (3, H, W) uint8
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


class ImageFolderLatentDataset(Dataset):
    """
    Fallback dataset using ImageFolder for images and HuggingFace for latents.
    Same format as SIT's dataset.
    Returns: (raw_image [0-255 uint8 CHW], latent_moments, label)
    """
    PRECOMPUTED = ["sdvae-ft-mse-f8d4"]

    def __init__(self, vae_name, data_dir, resolution=256, split="train"):
        assert vae_name in self.PRECOMPUTED, f"VAE {vae_name} not found in {self.PRECOMPUTED}"
        vae_split = "val" if split == "val" else ""
        self.img_dataset = ImageFolder(os.path.join(data_dir, "imagenet", split))
        self.resolution = resolution
        self.latent_dataset = load_from_disk(os.path.join(data_dir, f"imagenet-latents-{vae_name}", vae_split))
        assert len(self.img_dataset) == len(self.latent_dataset), \
            "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        image, label = self.img_dataset[idx]
        image = center_crop_arr(np.array(image.convert("RGB")), self.resolution)
        image = image.transpose(2, 0, 1)  # (3, H, W) uint8
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)


def encoder_preprocess(raw_image, resolution=256):
    """
    Preprocess raw uint8 images [0-255] for DINOv2 encoder.
    Matches SIT's encoder.preprocess(): normalize to [0,1], ImageNet norm, interpolate.

    Args:
        raw_image: (B, 3, H, W) uint8 or float tensor with values in [0, 255].
        resolution: Target resolution for the encoder.

    Returns:
        Preprocessed image tensor for encoder. (B, 3, resolution, resolution)
    """
    x = raw_image.float() / 255.0
    # ImageNet normalization
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    # Interpolate to encoder resolution if needed
    if x.shape[-1] != resolution or x.shape[-2] != resolution:
        x = torch.nn.functional.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
    return x


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Trains a DiT model with Encoder KV Distillation."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-EncoderKV"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Build frozen encoder (DINOv2):
    if args.use_kv or args.proj_coeff > 0:
        encoder, enc_dim, enc_num_heads = build_encoder(args.enc_type, device, img_size=args.enc_resolution)
        logger.info(f"Encoder: {args.enc_type}, dim={enc_dim}, heads={enc_num_heads}")

        # Parse layer indices for KV extraction
        enc_layer_indices = get_enc_layer_indices(args.enc_layer_indices, args.num_kv_layers)
        logger.info(f"Encoder KV layer indices: {enc_layer_indices}")

        # Parse DiT injection layer indices
        dit_kv_layer_indices = parse_layer_indices(args.dit_layer_indices)
        if dit_kv_layer_indices is not None:
            assert len(dit_kv_layer_indices) == len(enc_layer_indices), (
                f"--dit-layer-indices length ({len(dit_kv_layer_indices)}) must match "
                f"--enc-layer-indices length ({len(enc_layer_indices)})"
            )
            logger.info(f"DiT KV injection layer indices: {dit_kv_layer_indices}")
        else:
            logger.info(f"DiT KV injection layer indices: first {len(enc_layer_indices)} blocks (default)")

        # Build KV extractor
        kv_extractor = EncoderKVExtractor(encoder, enc_layer_indices)
    else:
        encoder = None
        kv_extractor = None
        enc_dim = 1024
        enc_num_heads = 16
        enc_layer_indices = []
        dit_kv_layer_indices = None

    # Load precomputed VAE latent statistics (SIT-style normalization):
    latents_stats_path = args.latents_stats_path
    if latents_stats_path and os.path.exists(latents_stats_path):
        latents_stats = torch.load(latents_stats_path, map_location=device, weights_only=False)
        latents_scale = latents_stats['latents_scale'].to(device).view(1, -1, 1, 1)
        latents_bias = latents_stats['latents_bias'].to(device).view(1, -1, 1, 1)
        logger.info(f"Loaded latent stats from {latents_stats_path}")
    else:
        latents_scale = 1.0
        latents_bias = 0.0
        logger.info("No latent stats file provided, using identity normalization")

    # Create model:
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    model = DiT_EncoderKV_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        enc_dim=enc_dim,
        enc_num_heads=enc_num_heads,
        num_enc_kv_layers=len(enc_layer_indices) if args.use_kv else 0,
        dit_kv_layer_indices=dit_kv_layer_indices if args.use_kv else None,
        kv_proj_type=args.kv_proj_type,
        kv_norm_type=args.kv_norm_type,
        encoder_depth=args.encoder_depth,
        repa_out_dim=enc_dim,
        repa_proj_type=args.repa_proj_type,
        repa_projector_dim=args.repa_projector_dim,
        repa_proj_kernel_size=args.repa_proj_kernel_size,
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)
    if args.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    model = DDP(model, device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision == "fp16" else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, None)
    logger.info(f"Mixed precision: {args.mixed_precision}")

    # Setup REPA projection loss:
    proj_loss_fn = None
    if args.proj_coeff > 0:
        proj_loss_fn = get_projection_loss(args.repa_loss)
        logger.info(f"REPA loss: {args.repa_loss}, coeff={args.proj_coeff}")

    # Setup optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data (SIT-compatible: precomputed latents + raw images):
    if HAS_HF_DATASETS:
        try:
            train_dataset = HFImgLatentDataset(args.vae_name, args.data_dir, split="train")
            logger.info("Using HFImgLatentDataset")
        except Exception as e:
            logger.info(f"HFImgLatentDataset failed ({e}), falling back to ImageFolderLatentDataset")
            train_dataset = ImageFolderLatentDataset(
                args.vae_name, args.data_dir, resolution=args.image_size, split="train"
            )
    else:
        train_dataset = ImageFolderLatentDataset(
            args.vae_name, args.data_dir, resolution=args.image_size, split="train"
        )
        logger.info("Using ImageFolderLatentDataset (HuggingFace datasets not installed)")

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Training variables:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_denoise_loss = 0
    running_proj_loss = 0
    running_distill_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"Stage 1 steps: {args.stage1_steps}")
    logger.info(f"Distill coeff: {args.distill_coeff}, Proj coeff: {args.proj_coeff}")

    # Initialize wandb (rank 0 only)
    use_wandb = args.wandb and HAS_WANDB and rank == 0
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
        )
    elif args.wandb and not HAS_WANDB:
        logger.warning("wandb not installed, skipping. Run: pip install wandb")

    total_steps = args.epochs * len(loader)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in pbar:
            # Unpack SIT-style batch: (raw_image, latent_moments, label)
            if len(batch) == 3:
                raw_image, x, y = batch
                raw_image = raw_image.to(device)
            else:
                x, y = batch
                raw_image = None
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)

            with torch.no_grad():
                # Sample from VAE posterior and normalize (SIT-style)
                x_latent = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

                # Extract encoder K/V and features from raw images
                enc_kv_list = None
                enc_features = None

                if encoder is not None and raw_image is not None:
                    # Preprocess raw uint8 images for DINOv2
                    raw_image_enc = encoder_preprocess(raw_image, resolution=args.enc_resolution)

                    if args.use_kv and args.proj_coeff > 0:
                        # Joint path: single forward extracts both KV (via hooks) and features
                        kv_extractor._kv_cache = {}
                        features = encoder.forward_features(raw_image_enc)
                        # Get patch tokens (remove CLS if present)
                        if hasattr(encoder, 'num_prefix_tokens') and encoder.num_prefix_tokens > 0:
                            enc_features = features[:, encoder.num_prefix_tokens:]
                        else:
                            enc_features = features
                        # Get captured KV from hooks
                        enc_kv_list = []
                        for idx in enc_layer_indices:
                            if idx in kv_extractor._kv_cache:
                                enc_kv_list.append(kv_extractor._kv_cache[idx])
                        kv_extractor._kv_cache = {}
                    elif args.use_kv:
                        enc_kv_list = kv_extractor(raw_image_enc)
                    elif args.proj_coeff > 0:
                        features = encoder.forward_features(raw_image_enc)
                        if hasattr(encoder, 'num_prefix_tokens') and encoder.num_prefix_tokens > 0:
                            enc_features = features[:, encoder.num_prefix_tokens:]
                        else:
                            enc_features = features

            # Determine stage:
            stage = 1 if train_steps < args.stage1_steps else 2

            # Freeze projection heads at the moment of Stage 2 transition
            if train_steps == args.stage1_steps and model.module.kv_projection is not None:
                requires_grad(model.module.kv_projection, False)
                logger.info(f"Step {train_steps}: transitioned to Stage 2, frozen kv_projection")

            # Forward pass through diffusion:
            t = torch.randint(0, diffusion.num_timesteps, (x_latent.shape[0],), device=device)
            model_kwargs = dict(
                y=y,
                enc_kv_list=enc_kv_list if args.use_kv else None,
                stage=stage,
            )
            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                loss_dict = diffusion.training_losses(model, x_latent, t, model_kwargs)
                denoise_loss = loss_dict["loss"].mean()

                # Get side outputs from model:
                model_module = model.module if hasattr(model, 'module') else model
                distill_loss = model_module._distill_loss
                zs = model_module._zs

                # Compute REPA projection loss:
                proj_loss = torch.tensor(0.0, device=device)
                if proj_loss_fn is not None and enc_features is not None and zs is not None:
                    if enc_features.shape[1] != zs.shape[1]:
                        B, N_enc, D_enc = enc_features.shape
                        N_dit = zs.shape[1]
                        h_enc = w_enc = int(N_enc ** 0.5)
                        h_dit = w_dit = int(N_dit ** 0.5)
                        enc_features = enc_features.permute(0, 2, 1).reshape(B, D_enc, h_enc, w_enc)
                        enc_features = torch.nn.functional.interpolate(
                            enc_features, size=(h_dit, w_dit), mode='bilinear', align_corners=False
                        )
                        enc_features = enc_features.reshape(B, D_enc, N_dit).permute(0, 2, 1)
                    proj_loss = proj_loss_fn(zs, enc_features.detach())

                # Compute total loss:
                distill_coeff = args.distill_coeff if stage == 2 else 0.0
                loss = denoise_loss + args.proj_coeff * proj_loss + distill_coeff * distill_loss

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            update_ema(ema, model.module)

            # Logging:
            running_loss += loss.item()
            running_denoise_loss += denoise_loss.item()
            running_proj_loss += proj_loss.item() if isinstance(proj_loss, torch.Tensor) else proj_loss
            running_distill_loss += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_denoise = torch.tensor(running_denoise_loss / log_steps, device=device)
                avg_proj = torch.tensor(running_proj_loss / log_steps, device=device)
                avg_distill = torch.tensor(running_distill_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_denoise, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_proj, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_distill, op=dist.ReduceOp.SUM)
                ws = dist.get_world_size()
                avg_loss = avg_loss.item() / ws
                avg_denoise = avg_denoise.item() / ws
                avg_proj = avg_proj.item() / ws
                avg_distill = avg_distill.item() / ws
                logger.info(
                    f"(step={train_steps:07d}, stage={stage}) "
                    f"Loss: {avg_loss:.4f}, Denoise: {avg_denoise:.4f}, "
                    f"Proj: {avg_proj:.4f}, Distill: {avg_distill:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0:
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", stage=stage, step=train_steps)
                if use_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "denoise_loss": avg_denoise,
                        "proj_loss": avg_proj,
                        "distill_loss": avg_distill,
                        "steps_per_sec": steps_per_sec,
                        "stage": stage,
                        "epoch": epoch,
                    }, step=train_steps)
                running_loss = 0
                running_denoise_loss = 0
                running_proj_loss = 0
                running_distill_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    if use_wandb:
        wandb.finish()
    if kv_extractor is not None:
        kv_extractor.remove_hooks()
    cleanup()


if __name__ == "__main__":
    # ── Config file support ───────────────────────────────────────────────────
    # First pass: extract --config path if present, then load YAML defaults.
    # Command-line args always override config values.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()

    _config_defaults = {}
    if _pre_args.config:
        import yaml
        with open(_pre_args.config) as _f:
            _yaml = yaml.safe_load(_f) or {}
        # Convert hyphen-keys to underscore-keys (argparse dest format)
        _config_defaults = {k.replace("-", "_"): v for k, v in _yaml.items()}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (values overridden by CLI args)")
    # Data args (SIT-compatible):
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root data directory containing imagenet-latents-images/ and imagenet-latents-sdvae-ft-mse-f8d4/")
    parser.add_argument("--vae-name", type=str, default="sdvae-ft-mse-f8d4",
                        help="Precomputed VAE latent name (must match SIT's preprocessing)")
    parser.add_argument("--latents-stats-path", type=str, default="pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt",
                        help="Path to precomputed latent statistics (latents_scale, latents_bias)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (used for wandb run name)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="DiT-EncoderKV", help="wandb project name")
    parser.add_argument("--model", type=str, choices=list(DiT_EncoderKV_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--mixed-precision", type=str, default="none", choices=["none", "fp16", "bf16"],
                        help="Mixed precision training (fp16/bf16 recommended for speed)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model for speed")
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    # Encoder KV distillation args:
    parser.add_argument("--use-kv", action="store_true", help="Enable encoder KV injection")
    parser.add_argument("--enc-type", type=str, default="vit_large_patch14_dinov2.lvd142m",
                        help="Encoder model name from timm")
    parser.add_argument("--enc-resolution", type=int, default=224,
                        help="Encoder input resolution (DINOv2 default: 224)")
    parser.add_argument("--enc-layer-indices", type=str, default=None,
                        help="Comma-separated encoder layer indices for KV extraction, 1-based like SIT (e.g., '20,23')")
    parser.add_argument("--num-kv-layers", type=int, default=4,
                        help="Number of encoder layers to use for KV (used if --enc-layer-indices not set)")
    parser.add_argument("--dit-layer-indices", type=str, default=None,
                        help="Comma-separated DiT block indices to inject encoder KV into, 1-based like SIT (e.g., '4,8'). "
                             "Must match length of enc-layer-indices. Default: first N DiT blocks.")
    parser.add_argument("--kv-proj-type", type=str, choices=["linear", "mlp"], default="linear",
                        help="Projection type for encoder KV")
    parser.add_argument("--kv-norm-type", type=str, choices=["layer", "zscore", "none"], default="layer",
                        help="Normalization type for projected KV")

    # Two-stage training args:
    parser.add_argument("--stage1-steps", type=int, default=50000,
                        help="Number of training steps for stage 1 (encoder KV)")
    parser.add_argument("--distill-coeff", type=float, default=1.0,
                        help="Distillation loss coefficient (stage 2 only)")

    # REPA projection args:
    parser.add_argument("--proj-coeff", type=float, default=0.5,
                        help="REPA projection loss coefficient (0 to disable)")
    parser.add_argument("--encoder-depth", type=int, default=8,
                        help="DiT layer index at which to extract features for REPA projector")
    parser.add_argument("--repa-loss", type=str, choices=["cosine", "mse"], default="cosine",
                        help="REPA projection loss type")
    parser.add_argument("--repa-proj-type", type=str, choices=["linear", "mlp", "conv"], default="linear",
                        help="REPA projector architecture: linear (1-layer), mlp (3-layer SiLU), or conv (Conv2d on H×W grid)")
    parser.add_argument("--repa-projector-dim", type=int, default=2048,
                        help="Hidden dim for mlp projector (ignored for linear/conv)")
    parser.add_argument("--repa-proj-kernel-size", type=int, default=1, choices=[1, 3, 5, 7],
                        help="Kernel size for conv projector (ignored for linear/mlp)")

    args = parser.parse_args()

    # Apply config values on top of CLI args (config has highest priority)
    if _config_defaults:
        for k, v in _config_defaults.items():
            if hasattr(args, k):
                setattr(args, k, v)

    main(args)
