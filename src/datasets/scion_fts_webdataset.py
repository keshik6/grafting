# Grafting Diffusion Transformers (NeurIPS 2025 Oral)
# Authors: Keshik
# https://grafting.stanford.edu

# Import base libaries
import os, sys, math
import pickle
from collections import defaultdict, Counter

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


class Scion_Training_WebDataset:
    def __init__(self, tar_dir, block_index, tar_str_indexes='split_{000000..000024}.tar', 
                return_tuple_keys=("__key__", "latent.pth", "label.pth", "class_name", "human_readable_class_name", "image_path"),
                num_samples=100000, selected_image_paths_file=None, train_subset=False, operator_type="mha",
                batch_size=256, num_workers=8 ):
        self.tar_dir = tar_dir
        self.tar_str_indexes = tar_str_indexes
        self.return_tuple_keys = return_tuple_keys
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_name = str(block_index)
        self.train_subset = train_subset
        self.operator_type = operator_type
        # self.num_samples_per_class = num_samples_per_class

        if self.train_subset:
            print(f'Loading {selected_image_paths_file}')
            image_paths_filepath = selected_image_paths_file
            with open(image_paths_filepath, 'rb') as f:
                selected_image_paths = pickle.load(f)
            self.selected_image_paths = selected_image_paths
            print(len(self.selected_image_paths))
        
        assert self.num_samples % self.num_workers == 0

    def get_dataset(self):
        def filter_samples_train(sample):
            #print(sample["image_path"].decode('utf-8'), self.selected_image_paths[0])
            if self.train_subset:
                return sample["image_path"].decode('utf-8') in self.selected_image_paths
            return True

        # filter_fnc = filter_samples_generate_set if not self.train_subset else filter_samples_train
        filter_fnc = filter_samples_train
        print(filter_fnc)

        dataset = wds.DataPipeline(
            wds.SimpleShardList(str(os.path.join(self.tar_dir, f'block_{self.block_name}_{self.operator_type}', self.tar_str_indexes))),
            # add wds.split_by_node here if you are using multiple nodes
            wds.split_by_node,     
            wds.split_by_worker,                     

            # Optional: Log shards being processed by each worker
            wds.map(log_worker_shards),
    
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),

            # this decodes the images and json
            wds.decode("torch"),

            # Filter samples to only get 10 samples per class (Do this later for ablation studies)
            wds.select(filter_samples_train),

            #wds.rsample(),
            # this shuffles the samples in memory
            wds.shuffle(1000), #uncomment for training, comment for analysis
            wds.to_tuple(*self.return_tuple_keys),
        )


        #dataset = wds.repeatedly(dataset, nsamples=10000)
        return dataset


    def get_dataloader(self):
        # Create the DataLoader
        dataset = self.get_dataset()
        #dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        dataloader = wds.WebLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, pin_memory=True)
        return dataloader


# Usage example 
def test_loader():
    
    # Test 1
    config_filepath = 'configs/train_scions/imagenet/debug.yaml'
    split = 'train'
    config_original = load_yaml_file(config_filepath)[split]
    # print(config_original)
    sha_key_verify = config_original['sha_key']
    config = remove_keys(config_original, ['sha_key'])
    # print(config)

    # Create Webdataset
    reader = Scion_Training_WebDataset(**config)
    dataloader = reader.get_dataloader()

    # Iterate through the DataLoader
    all_class_names = []
    all_image_paths = []

    num = 0
    for batch in dataloader:
        keys, in_fts, out_fts, latents, labels, t, class_names, class_names_hr, img_paths, layer_name = batch
        #keys, latents, labels = batch
        #print(out_fts.sum(dim=2).sum(dim=1), out_fts.size())

        num+=latents.size(0)
        all_image_paths.extend(img_paths)
        all_class_names.extend(class_names)
    
        # print(num)
    
    #print(num)
    # Calculate hash for verification
    occurrences = Counter(all_class_names)
    result_dict = dict(occurrences)
    #print(result_dict)
    all_image_paths = [path.decode('utf-8') for path in all_image_paths]
    
    # Get set for resampling
    all_image_paths = list(set(all_image_paths))
    print(len(all_image_paths))

    sha_key = generate_sha_hash(all_image_paths)
    assert verify_hash(sha_key_verify, sha_key)
    print(f'> Hash key verification successful for {split} split, total images={num} ')
    #print(sha_key)


if __name__ == "__main__":
    test_loader()
