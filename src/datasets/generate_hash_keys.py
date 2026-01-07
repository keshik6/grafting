# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# Import base libaries
import os, sys, math
import argparse
import pickle

# Import scientific computing libraries
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist


# Import visualization/ data processing libraries
import matplotlib.pyplot as plt
import pandas as pd
import webdataset as wds

# Import utils
sys.path.append(f"{os.getcwd()}/src/") # Append paths
from utils import *
from collections import defaultdict, Counter
from imagenet_vae_dataset import ImageNet_VAE_WebDataset

from itertools import islice



def gather_paths_and_generate_key(image_paths, rank, world_size, args):
    # Serialize image paths
    serialized_image_paths = pickle.dumps(image_paths)

    # Gather serialized data from all GPUs
    gathered_serialized_data = [None] * world_size
    dist.all_gather_object(gathered_serialized_data, serialized_image_paths)

    #Deserialize and aggregate data on rank 0
    if rank == 0:
        all_image_paths = []
        for data in gathered_serialized_data:
            all_image_paths.extend(pickle.loads(data))

        # print(len(all_image_paths))
        # print(all_image_paths[:10])

        all_image_paths = [path.decode('utf-8') for path in all_image_paths]
        sha_key = generate_sha_hash(all_image_paths)
        save_hash(sha_key, file_name=args.save_filepath)
        load_hash(file_name=args.save_filepath)
        #assert verify_hash(sha_key_verify, sha_key)
        print(sha_key)
        print(f'Total #image_paths = {len(all_image_paths)}')

        # Save the aggregated image paths to a file
        files_savepath = args.save_filepath.replace('.txt', '_image_paths.pkl')
        with open(files_savepath, 'wb') as f:
            pickle.dump(all_image_paths, f)
        print("Saved all image paths to aggregated_image_paths.pkl")


"""
Generate Hash Keys for Webdatasets
"""
# Generate hash key
def generate_hash_key(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Test 1
    # config = {
    # "tar_dir": "/data/vae_features/imagenet_256/train/",
    # "tar_str_indexes":'split_{000000..00004}.tar',
    # #"tar_str_indexes":'split_000000.tar',
    # "return_tuple_keys": ("__key__", "latent.pth", "label.pth", "class_name", "human_readable_class_name", "image_path"),
    # "batch_size": 256,
    # "num_workers": 5, # Basically 0-4 tars will be used.
    # "num_samples_per_class": 5,
    # "num_samples": 50000,
    # "train_scions": True,
    # }

    #config["num_samples"] = config["num_workers"]*config["num_samples_per_class"]*1000
    #print(config["num_samples"] )
    
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # config_filepath = 'configs/datasets/imagenet/vae_features/generate_sha_key_50k_set_val.yaml'
    config_filepath = args.config_filepath
    config = load_yaml_file(config_filepath)
    # sha_key_verify = config_original['sha_key']
    # config = remove_keys(config_original, ['sha_key'])
    print(config)

    global_class_sample_counts = torch.zeros(1000, dtype=torch.int).cuda(local_rank)  # Assuming 1000 classes for ImageNet
    reader = ImageNet_VAE_WebDataset(**config)
    dataloader = reader.get_dataloader()
    
    # Iterate through the DataLoader
    all_class_names = []
    all_image_paths = []
    num = 0

    for batch in dataloader:
        keys, latents, labels, class_names, class_names_hr, img_paths = batch
        num += latents.size(0)

        # Append the keys
        all_class_names.extend(class_names)
        all_image_paths.extend(img_paths)
        
        #print(num)
    
    # Gather all image paths from all GPUs
    gather_paths_and_generate_key(all_image_paths, local_rank, world_size, args)
    # print(num)
    #torch.cuda.synchronize()
    # Plot histogram

    # do using a single gpu
    if config['train_scions'] ==False:
        occurrences = Counter(all_class_names)
        result_dict = dict(occurrences)
        plot_name = args.save_filepath.split('/')[-1].split('.txt')[0]
        #print(result_dict)
        os.makedirs('./plots', exist_ok=True)
        plot_histogram(result_dict, output_filename=f'plots/{plot_name}.png')

        # all_image_paths = [path.decode('utf-8') for path in all_image_paths]
        # sha_key = generate_sha_hash(all_image_paths)
        # #assert verify_hash(sha_key_verify, sha_key)
        # print(sha_key)
        # # save_hash(sha_key, file_name='assets/imagenet/hash_key_50k_set_val_256.txt')
        # # load_hash(file_name='assets/imagenet/hash_key_50k_set_val_256.txt')
        # save_hash(sha_key, file_name=args.save_filepath)
        # load_hash(file_name=args.save_filepath)


    # Finalize the distributed environment
    dist.destroy_process_group()

    #return sha_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-filepath", type=str, required=True)
    parser.add_argument("--save-filepath", type=str, required=True)
    args = parser.parse_args()
    generate_hash_key(args)
