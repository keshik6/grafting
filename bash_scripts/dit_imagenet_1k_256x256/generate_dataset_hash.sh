#!/bin/bash

echo "Grafting Diffusion Transformers!"

SPLIT=train
NUM_SAMPLES=128k
CONFIG_FILEPATH=configs/datasets/imagenet_1k_256x256/vae_fts/generate_sha_key_$NUM_SAMPLES\_$SPLIT\_set.yaml
SAVE_FILEPATH=assets/imagenet/splits/hash_key_$NUM_SAMPLES\_set_$SPLIT\_256.txt
NUM_GPUS=1

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS  src/datasets/generate_hash_keys.py \
  --config-filepath "$CONFIG_FILEPATH" \
  --save-filepath "$SAVE_FILEPATH"