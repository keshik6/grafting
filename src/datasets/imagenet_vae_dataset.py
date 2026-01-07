# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# Import base libaries
import os, sys, math
import pickle


# Import scientific computing libraries
import torch
from torch.utils.data import DataLoader

# Import visualization/ data processing libraries
import matplotlib.pyplot as plt
import pandas as pd
import webdataset as wds

# Import utils
sys.path.append(f"{os.getcwd()}/src/") # Append paths
from utils import *
from collections import defaultdict, Counter
import torch.distributed as dist

"""
VAE Features Webdataset Class for training Latent Diffusion Transformers
"""
class ImageNet_VAE_WebDataset:
    def __init__(self, tar_dir, tar_str_indexes='split_{000000..000024}.tar', 
                return_tuple_keys=("__key__", "latent.pth", "label.pth", "class_name", "human_readable_class_name", "image_path"),
                num_samples=None, num_samples_per_class=10, train_scions=True,
                batch_size=256, num_workers=8, selected_image_paths_file=None, shuffle=False ):
        self.tar_dir = tar_dir
        self.tar_str_indexes = tar_str_indexes
        self.return_tuple_keys = return_tuple_keys
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scions = train_scions
        self.shuffle=shuffle

        # Load selected image paths
        if self.train_scions:
            print(f'Loading {selected_image_paths_file}')
            image_paths_filepath = selected_image_paths_file
            with open(image_paths_filepath, 'rb') as f:
                selected_image_paths = pickle.load(f)
            self.selected_image_paths = selected_image_paths
            print(f'Number of samples: {len(self.selected_image_paths)}')
        
        if train_scions:
            assert self.num_samples % self.num_workers == 0, "Stratified Split requires number of samples to be divisible by number of workers"

    def get_dataset(self):
        # Create a class_sample_counts dictionary within the get_dataset scope
        class_sample_counts = defaultdict(int)
        
        # Define the filter function within get_dataset so it has access to class_sample_counts
        def filter_samples_generate_set(sample):
            class_label = sample["label.pth"].item()
            if class_sample_counts[class_label] < self.num_samples_per_class:
                class_sample_counts[class_label] += 1
                return True
            return False

        def filter_samples_train(sample):
            #print(sample["image_path"].decode('utf-8'), self.selected_image_paths[0])
            return sample["image_path"].decode('utf-8') in self.selected_image_paths

        filter_fnc = filter_samples_generate_set if not self.train_scions else filter_samples_train
        print(filter_fnc, self.shuffle)
        
        if self.shuffle:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(str(os.path.join(self.tar_dir, self.tar_str_indexes))),
                # add wds.split_by_node here if you are using multiple nodes
                wds.split_by_node,     
                wds.split_by_worker,         

                # Optional: Log shards being processed by each worker
                wds.map(log_worker_shards),
        
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),

                # Get the first samples
                wds.slice(int(math.ceil(self.num_samples_per_class*1000))+1000),
                # this decodes the images and json
                wds.decode("torch"),

                # Filter samples to only get 10 samples per class (Do this later for ablation studies)
                wds.select(filter_fnc),

                # this shuffles the samples in memory
                wds.shuffle(1000),

                wds.to_tuple(*self.return_tuple_keys),
            )

        else:
            dataset = wds.DataPipeline(
                wds.SimpleShardList(str(os.path.join(self.tar_dir, self.tar_str_indexes))),
                # add wds.split_by_node here if you are using multiple nodes
                wds.split_by_node,     
                wds.split_by_worker,         

                # Optional: Log shards being processed by each worker
                wds.map(log_worker_shards),
        
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),

                # Get the first samples
                wds.slice(int(math.ceil(self.num_samples_per_class*1000))+1000),
                # this decodes the images and json
                wds.decode("torch"),

                # Filter samples to only get 10 samples per class (Do this later for ablation studies)
                wds.select(filter_fnc),

                wds.to_tuple(*self.return_tuple_keys),
            )


        return dataset


    def get_dataloader(self):
        # Create the DataLoader
        dataset = self.get_dataset()
        dataloader = wds.WebLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False)
        return dataloader



if __name__ == "__main__":
    pass
    #generate_hash_key()
