"""
Multi-GPU DDP sampling for DiT with Encoder KV Distillation.
At inference, no encoder is needed — only DiT's own K/V are used.
Saves a .npz file compatible with ADM/guided-diffusion evaluator.

Usage:
    torchrun --standalone --nproc_per_node=8 sample_encoder_ddp.py \
        --ckpt results/000-DiT-XL-2-EncoderKV/checkpoints/0400000.pt \
        --model DiT-XL/2 --cfg-scale 1.5
"""
import torch
import torch.distributed as dist
from models_encoder import DiT_EncoderKV_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse


def find_model(model_path):
    """Load EMA weights preferentially from a train_encoder.py checkpoint."""
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    if "ema" in checkpoint:
        return checkpoint["ema"], checkpoint.get("args", None)
    elif "model" in checkpoint:
        return checkpoint["model"], checkpoint.get("args", None)
    return checkpoint, None


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load checkpoint and recover training args for architecture reconstruction
    state_dict, train_args = find_model(args.ckpt)

    # Use saved training args to reconstruct the exact same architecture
    latent_size = args.image_size // 8
    enc_dim       = getattr(train_args, 'enc_dim',            args.enc_dim)       if train_args else args.enc_dim
    enc_num_heads = getattr(train_args, 'enc_num_heads',      args.enc_num_heads) if train_args else args.enc_num_heads
    enc_layer_indices = getattr(train_args, 'enc_layer_indices', None)            if train_args else None
    num_enc_kv_layers = len(enc_layer_indices.split(',')) if enc_layer_indices else args.num_enc_kv_layers
    kv_proj_type  = getattr(train_args, 'kv_proj_type',       'linear')           if train_args else 'linear'
    kv_norm_type  = getattr(train_args, 'kv_norm_type',       'layer')            if train_args else 'layer'
    encoder_depth = getattr(train_args, 'encoder_depth',      args.encoder_depth) if train_args else args.encoder_depth
    repa_proj_type       = getattr(train_args, 'repa_proj_type',       'linear')  if train_args else 'linear'
    repa_projector_dim   = getattr(train_args, 'repa_projector_dim',   2048)      if train_args else 2048
    repa_proj_kernel_size = getattr(train_args, 'repa_proj_kernel_size', 1)       if train_args else 1

    model = DiT_EncoderKV_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_enc_kv_layers=num_enc_kv_layers,
        enc_dim=enc_dim,
        enc_num_heads=enc_num_heads,
        kv_proj_type=kv_proj_type,
        kv_norm_type=kv_norm_type,
        encoder_depth=encoder_depth,
        repa_out_dim=enc_dim,
        repa_proj_type=repa_proj_type,
        repa_projector_dim=repa_projector_dim,
        repa_proj_kernel_size=repa_proj_kernel_size,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0
    using_cfg = args.cfg_scale > 1.0

    # Output folder
    model_str = args.model.replace("/", "-")
    ckpt_str = os.path.basename(args.ckpt).replace(".pt", "")
    folder_name = f"{model_str}-{ckpt_str}-size{args.image_size}-cfg{args.cfg_scale}-seed{args.global_seed}"
    sample_folder = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder, exist_ok=True)
        print(f"Saving samples to {sample_folder}")
    dist.barrier()

    # Sampling loop
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_per_gpu = total_samples // dist.get_world_size()
    iterations = samples_per_gpu // n

    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0

    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for i, sample in enumerate(samples):
            idx = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder}/{idx:06d}.png")
        total += global_batch_size

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model", type=str, choices=list(DiT_EncoderKV_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    # Architecture (must match training)
    parser.add_argument("--num-enc-kv-layers", type=int, default=1)
    parser.add_argument("--enc-dim", type=int, default=1024)
    parser.add_argument("--enc-num-heads", type=int, default=16)
    parser.add_argument("--encoder-depth", type=int, default=8)
    # Sampling
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--sample-dir", type=str, default="samples")
    args = parser.parse_args()
    main(args)
