# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# References:
# https://github.com/facebookresearch/DiT
# https://github.com/chuanyangjin/fast-DiT


"""
Script to extract Scion Features for training/ grafting Latent Diffusion Transformers
"""
import io
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os, sys
from accelerate import Accelerator

sys.path.append(f"{os.getcwd()}/src/") # Append paths
from models.dit import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.download import find_model

from webdataset import ShardWriter
from imagenet_vae_dataset import ImageNet_VAE_WebDataset
from utils import *


"""
This code extracts all intermediate features for attention operators in DiT for training Scions.
With a single A100 GPU, it takes about 150 mins to extract all intermediate features for 50k datapoints. 
"""
#################################################################################
#                             Scion Feature Extraction Code                     #
#################################################################################

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


import re
def extract_shard_index(filename):
    match = re.search(r'split_(\d+)\.tar', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Filename does not match the expected pattern")


# Function to register multiple hooks and collect activations by block indices
def get_activations_by_indices(model, input_data, layer_indices):
    activations = {}
    hooks = []

    def make_forward_hook(layer_path):
        def forward_hook(module, input, output):
            activations[layer_path] = {'input': input[0].detach().cpu().to(torch.float16), 'output': output.detach().cpu().to(torch.float16)}
            #print(f"Layer: {layer_path} Output 0 Shape: {output.size()}")

        return forward_hook

    # Register hooks for all specified layers
    for layer_path in layer_indices:

        # Hook for attention inputs/outputs
        layer = eval(f'model.{layer_path}')
        hooks.append(layer.register_forward_hook(make_forward_hook(layer_path))) # layer_path doesn't have "model."

        # Hook for modulation inputs/outputs
        modulation_layer_path = layer_path.replace(".attn", ".adaLN_modulation") # This is the modulation layer name
        layer_adaln = eval(f'model.{modulation_layer_path}') 
        hooks.append(layer_adaln.register_forward_hook(make_forward_hook(modulation_layer_path)))


    # Pass the input through the model
    with torch.no_grad():
        model(*input_data)

    # Remove all hooks to avoid memory leaks
    for hook in hooks:
        hook.remove()

    return activations


def aggregate_image_counts(image_counts, rank, world_size):
    # Convert counts to tensor for aggregation
    count_tensors = {k: torch.tensor(v, device='cuda') for k, v in image_counts.items()}

    # Aggregate counts across all GPUs
    aggregated_counts = {}
    for k, v in count_tensors.items():
        aggregated_count = torch.zeros_like(v)
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        aggregated_counts[k] = v.item()

    # Print aggregated counts on rank 0
    if rank == 0:
        print("Aggregated Image Counts:", aggregated_counts)


def main(args): 
    """
    Extracts intermediate features for grafting. 
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")
    torch.set_grad_enabled(False)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        experiment_dir = './logs/'
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Load pre-trained model for fine-tuning 
    ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    # logger.info(f'Pretrained model at {ckpt_path} loaded successfully.')

    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f'Pretrained model at {ckpt_path} loaded successfully.')

    config_filepath = args.config_filepath
    config_original = load_yaml_file(config_filepath)
    sha_key_verify = config_original['sha_key']
    data_split = config_original['split']
    config = remove_keys(config_original, ['sha_key', 'split'])

    reader = ImageNet_VAE_WebDataset(**config)
    loader = reader.get_dataloader()

    model.eval()  # important!

    # Print trainable parameters (for sanity check)
    trainable_params = [ n for (n, p) in model.named_parameters() if p.requires_grad ]
    print(f'List of trainable parameters')
    for element in trainable_params:
        print(element)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {total_trainable_params}')

    # Variables for monitoring/logging purposes:
    train_steps = 0
    start_time = time()

    #sink = ShardWriter(writer_pattern, maxcount=50000, start_shard=0)
    if accelerator.is_main_process:
        scion_fts_path = os.path.join(args.feature_path, f'scion_fts_mha/{data_split}')
        os.makedirs(scion_fts_path, exist_ok=True)
        [os.makedirs(f'{scion_fts_path}/block_{i}_mha', exist_ok=True) for i in args.dit_block_indexes]
        start_shard_index = extract_shard_index(args.tar_str_indexes)
        print(f'Start shard index = {start_shard_index}')
        sinks = [ShardWriter(f'{scion_fts_path}/block_{i}_mha/split_%03d.tar', maxcount=args.shard_size, start_shard=start_shard_index, maxsize=10e9) for i in args.dit_block_indexes] # use for val
    
    # Save the initialized checkpoint to track hyena weights
    if accelerator.is_main_process:
        logger.info(f"Extracting features ...")

    if accelerator.is_main_process:
        logger.info(f"Beginning activation extractions...")

    image_counts = {}
    block_names = [f'blocks[{i}].attn' for i in args.dit_block_indexes]

    for block_name in block_names:
        image_counts[block_name] = 0
    print(image_counts)
    for index, (x, y, class_names, human_readable_class_names, image_paths) in enumerate(loader):
        #print(keys)
        x = x.to(device)
        y = y.to(device)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

        # Extract activations from specific blocks
        activations = get_activations_by_indices(model, (x,y,t), block_names)
        print(activations.keys())
        torch.cuda.synchronize()
        
        # Print results for each layer
        all_activations = {}
        for layer_name, layer_activations in activations.items():
            if layer_name not in all_activations:
                all_activations[layer_name] = { 'input': layer_activations['input'], 'output': layer_activations['output'] } 

        if accelerator.is_main_process:
            layer_idx = 0

            # Looping through keys (do not loop through modulation as you need to extract them and save them in the same sample)
            for layer_n in all_activations:
                #print(layer_idx)

                if '.adaLN_modulation' in layer_n:
                    print(f'skipping {layer_n}')
                    continue

                layer_n_adaln_name = layer_n.replace('.attn', '.adaLN_modulation')
                print(f'Saving {layer_n} and {layer_n_adaln_name} features now...')

                for i in range(all_activations[layer_n]['input'].size(0)):
                    
                    # Attention features
                    in_fts = io.BytesIO()
                    out_fts = io.BytesIO()

                    # AdaLN features for each attention block
                    adaln_in_fts = io.BytesIO()
                    adaln_out_fts = io.BytesIO()

                    # Additional features
                    latent_buffer = io.BytesIO()
                    label_buffer = io.BytesIO()
                    t_buffer = io.BytesIO()

                    # print(i)
                    # print(layer_n, all_activations[layer_n]['input'][i].size())
                    # print(layer_n, all_activations[layer_n]['output'][i].size())
                    # print(layer_n_adaln_name, all_activations[layer_n_adaln_name]['input'][i].size())
                    # print(layer_n_adaln_name, all_activations[layer_n_adaln_name]['output'][i].size())


                    #print(all_activations[layer_n]['shift_scale_gate'].size())
                    # Serialization Complexity: When serializing a tensor view, PyTorch's torch.save serializes the entire storage to 
                    # maintain consistency and avoid losing data that might be needed by other views or the original tensor.
                    # Fix: Clone when creating memory buffers.
                    torch.save(all_activations[layer_n]['input'][i].clone().detach(), in_fts) # save attn input features
                    torch.save(all_activations[layer_n]['output'][i].clone().detach(), out_fts) # save attn output features
                    torch.save(all_activations[layer_n_adaln_name]['input'][i].clone().detach(), adaln_in_fts) # adaln input features (for each block)
                    torch.save(all_activations[layer_n_adaln_name]['output'][i].clone().detach(), adaln_out_fts) # adaln output features (for each block)

                    torch.save(x[i].cpu().clone().detach(), latent_buffer)
                    torch.save(y[i].cpu().clone().detach(), label_buffer)
                    torch.save(t[i].cpu().clone().detach(), t_buffer)
                    # torch.save(all_activations[layer_n]['output'][i], out_fts) # This is incorrect, thus commented.
                    
                    #print(i)
                    sample = {
                        "__key__": f"{image_counts[layer_n]:08d}",
                        "in_fts.pth": in_fts.getvalue(), # use .pth extension for saving tensors.
                        "out_fts.pth": out_fts.getvalue(), # use .pth extension for saving tensors.
                        "adaln_in_fts.pth": adaln_in_fts.getvalue(), # use .pth extension for saving tensors.
                        "adaln_out_fts.pth": adaln_out_fts.getvalue(), # use .pth extension for saving tensors.
                        "latent.pth": latent_buffer.getvalue(),
                        "label.pth": label_buffer.getvalue(),
                        "t.pth": t_buffer.getvalue(),
                        "class_name": class_names[i].decode('utf-8'),
                        "human_readable_class_name": human_readable_class_names[i].decode('utf-8'),
                        "image_path": image_paths[i].decode('utf-8'),
                        "layer_name": layer_n,
                    }
                    sinks[layer_idx].write(sample)

                    image_counts[layer_n] += 1
                    #print(image_counts)

                    # Clear buffers
                    #in_fts.close()
                    out_fts.close()
                    latent_buffer.close()
                    label_buffer.close()
                    t_buffer.close()


                layer_idx += 1

            #print(image_count)
            print("saved index", index)
            torch.cuda.empty_cache()
        
            train_steps += 1
    
    # if accelerator.is_main_process:
    #     print(image_counts)

    # Aggregate image counts from all GPUs
    aggregate_image_counts(image_counts, rank, world_size)
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Custom
    def parse_int_list(value):
        return [int(i) for i in value.split(',') if i.strip().isdigit()]

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="/data/")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--dit-block-indexes", type=parse_int_list, default=[])
    parser.add_argument("--config-filepath", type=str, required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--tar-str-indexes", type=str, required=True)
    
    args = parser.parse_args()
    print(args)
    # sys.exit()
    main(args)

   
