# References:
# https://github.com/facebookresearch/DiT
# https://github.com/chuanyangjin/fast-DiT

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
# from models_hyena import DiT_models
#from models import DiT_models
# from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

import gc

from utils import *
from itertools import islice
from graft import graft_dit


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
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
    """
    Sample images.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Read config file
    config_filepath = args.config_filepath
    config = load_yaml_file(config_filepath)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = int(config['sample_config']['global_seed']) * dist.get_world_size() + rank
    assert int(config['sample_config']['global_seed']) == 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # if args.ckpt is None:
    #     assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
    #     assert args.image_size in [256, 512]
    #     assert args.num_classes == 1000

    # Get arguments
    # config_filepath = args.config_filepath
    # config = load_yaml_file(config_filepath)
    dit_model_name = config['sample_config']['dit_model_name']
    dit_ckpt_path = config['sample_config']['grafted_dit_ckpt_path']
    dit_ckpt_path = None if dit_ckpt_path=="None" else dit_ckpt_path
    image_size = config['sample_config']['image_size']
    graft_indexes = config['sample_config']['graft_indexes']
    graft_weights = config['sample_config']['graft_weights']
    graft_weights = {} if graft_weights==None else graft_weights
    operator_name = config['operator']['name']
    operator_type = config['operator']['type']
    operator_config_filepath = config['operator']['config_filepath']
    #print(dit_ckpt_path, dit_ckpt_path == None)
    print(f'training grafted {dit_model_name}')

    # Load model:
    latent_size = image_size // 8
    print(f'Graft indexes = {graft_indexes}, Graft weights initialization = {graft_weights}')
    model = graft_dit(dit_model_name, dit_ckpt_path, image_size, operator_type, operator_name, operator_config_filepath, 
                    graft_indexes, graft_weights, run_all_unit_tests=False).to(device)
    print(model)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = dit_ckpt_path
    if ckpt_path is not None:
        print(f'Using ckpt at {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location='cuda')['ema']
        print(f'using ema checkpoint....')
        #state_dict = find_model(ckpt_path)
        print(state_dict.keys())
        model.load_state_dict(state_dict, strict=True)
    
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Save folder name
    folder_name = config['sample_config']['samples_save_dir']
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(args.num_fid_samples , global_batch_size)
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    for pbar_index in pbar:
        print(pbar_index, model.in_channels, latent_size )
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

        # Delete the samples
        del samples
        torch.cuda.empty_cache()
        gc.collect()


    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    #parser.add_argument("--num-fid-samples", type=int, default=50)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--config-filepath", type=str, required=True)
    args = parser.parse_args()
    main(args)
