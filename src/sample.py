# References:
# https://github.com/facebookresearch/DiT
# https://github.com/chuanyangjin/fast-DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
from torchsummary import summary

from utils import *
from itertools import islice
from graft import graft_dit


def main(args):
    """
    Sample images from a pretrained (and optionally grafted) Diffusion Transformer (DiT) model.

    This script loads a DiT model whose architecture and performs grafting (based on the config file)
    It then generates class-conditional images (ImageNet-1K).

    Config and checkpoint loading, grafting, and sampling behavior are controlled via a YAML
    configuration file.

    References:
    - DiT base code: https://github.com/facebookresearch/DiT
    - Fast-DiT fork: https://github.com/chuanyangjin/fast-DiT
    - Our method: https://arxiv.org/abs/2506.05340

    Usage:
        python sample.py --config-filepath path/to/config.yaml

    Args:
        --vae: Which pretrained VAE to use ('mse' or 'ema')
        --image-size: Image resolution (256 or 512)
        --num-classes: Number of classes (typically 1000 for ImageNet)
        --cfg-scale: Classifier-free guidance scale
        --num-sampling-steps: Number of diffusion steps
        --seed: Random seed for reproducibility
        --config-filepath: Path to YAML file specifying model, graft, and operator settings

    Output:
        - Saves a grid of sampled images as `sample.png` in the current directory.
    """

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read config file
    config_filepath = args.config_filepath
    config = load_yaml_file(config_filepath)

    dit_model_name = config['sample_config']['dit_model_name']
    dit_ckpt_path = config['sample_config']['grafted_dit_ckpt_path']
    dit_ckpt_path = None if dit_ckpt_path=="None" else dit_ckpt_path
    image_size = config['sample_config']['image_size']
    graft_indexes = config['sample_config']['graft_indexes']
    graft_weights = config['sample_config']['graft_weights']
    graft_weights = {} if graft_weights==None else graft_weights
    operator_type = config['operator']['type']
    operator_name = config['operator']['name']
    operator_config_filepath = config['operator']['config_filepath']

    # Load model:
    latent_size = image_size // 8
    print(f'Graft indexes = {graft_indexes}, Graft weights initialization = {graft_weights}')
    model = graft_dit(dit_model_name, dit_ckpt_path, image_size, 
                    operator_type, operator_name, operator_config_filepath, 
                    graft_indexes, graft_weights, 
                    run_all_unit_tests=False).to(device)
    print(model)


    ckpt_path = dit_ckpt_path
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cuda')
        state_dict = state_dict['ema']
        model.load_state_dict(state_dict, strict=True)
        print(f'Using ckpt at {ckpt_path}')
    
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
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

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-filepath", type=str, default=None)    
    args = parser.parse_args()
    main(args)