# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# References:
# https://github.com/facebookresearch/DiT
# https://github.com/chuanyangjin/fast-DiT
 
"""
Script to extract VAE features for training/ grafting Latent Diffusion Transformers
"""
import torch, io
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
import argparse
import os, math
from webdataset import ShardWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Printing
from rich.console import Console
from rich.text import Text

# Import custom modules
from diffusers.models import AutoencoderKL
from imagenet_dataset import ImageNetDataset


def str_or_int(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for num_samples: {value}")


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    # pil_image = pil_image.resize(
    #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    # )

    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Feature Extractio                            #
#################################################################################

def extract_vae_features(args):
    """
    Extract VAE features.
    """
    assert torch.cuda.is_available(), "Extracting features currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageNetDataset(args.data_path, num_samples_per_class=args.num_samples, transform=transform, verbose=False)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size//world_size, 
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # print(dataset.__len__())

    # Rank 0 initialization for writing
    if rank == 0:
        console = Console()
        message = f'Dataset contains {dataset.__len__()} images.'
        console.print(message, Text("✔", style="bold green"))
        writer_pattern = f"{args.features_path}/split_%06d.tar"
        sink = ShardWriter(writer_pattern, maxcount=args.max_count_per_split, start_shard=0)


    image_count = 0 # After 10 -> 00503315
    
    for x, y, class_names, human_readable_class_name, img_paths in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        # Gather objects across all ranks
        class_names_list = [None for _ in range(world_size)]
        class_names_hr_list = [None for _ in range(world_size)]
        img_paths_list = [None for _ in range(world_size)]
        
        #print(img_paths_list)
        dist.all_gather_object(class_names_list, class_names)
        dist.all_gather_object(class_names_hr_list, human_readable_class_name)
        dist.all_gather_object(img_paths_list, img_paths)

        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        # Prepare the lists to gather tensors from all processes
        latents_list = [torch.zeros_like(x) for _ in range(world_size)]
        labels_list = [torch.zeros_like(y) for _ in range(world_size)]

        # Gather tensors from all processes
        dist.all_gather(latents_list, x)
        dist.all_gather(labels_list, y)

        # Only rank 0 writes to the webdataset
        if rank == 0:
            torch.cuda.synchronize()
            # Concatenate all gathered tensors along the batch dimension
            all_latents = torch.cat(latents_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)

            #print(all_latents.size(), all_labels.size())

            # Flatten class_names_list and img_paths_list
            flat_class_names = [name for sublist in class_names_list for name in sublist]
            flat_class_names_hr = [name for sublist in class_names_hr_list for name in sublist]
            flat_img_paths = [path for sublist in img_paths_list for path in sublist]

            # Process each sample in the concatenated batch
            for i in range(all_latents.size(0)):
                latent_buffer = io.BytesIO()
                label_buffer = io.BytesIO()
                torch.save(all_latents[i].cpu().clone().detach(), latent_buffer) # Clone to create new memory mapping
                torch.save(all_labels[i].cpu().clone().detach(), label_buffer) # Clone to create new memory mapping

                # Write to webdataset
                sample = {
                    "__key__": f"{image_count:08d}",
                    "latent.pth": latent_buffer.getvalue(), # use .pth extension for saving tensors.
                    "label.pth": label_buffer.getvalue(), # # use .pth extension for saving tensors.
                    "class_name": flat_class_names[i],
                    "human_readable_class_name": flat_class_names_hr[i],
                    "image_path": flat_img_paths[i]
                }
                sink.write(sample)
                image_count += 1

                # Close
                latent_buffer.close()
                label_buffer.close()


    if rank == 0:
        sink.close()
        message = f'Completed extracting VAE features for ImageNet (resolution={args.image_size}, total images={image_count})'
        console.print(message, Text("✔", style="bold green"))
        # print(f'Completed extracting VAE features for ImageNet (resolution={args.image_size}, total images={image_count})')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="vae_features_256")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--max_count_per_split', type=int, default=100000, help="Max count per tar file for webdataset")
    parser.add_argument("--num-samples", type=str_or_int, default=None, help="Number of samples, can be an integer or 'None'")
    args = parser.parse_args()
    extract_vae_features(args)
