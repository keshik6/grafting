#!/bin/bash

echo "Grafting Diffusion Transformers!"

# This is agnostic to image resolution since the compression rate is 8 for SD-VAE.
SPLIT=train # Set this to train or val
DATA_PATH="/data/imagenet/$SPLIT"
FEATURES_PATH="/data/vae_features/imagenet_256/$SPLIT/"
IMAGE_SIZE=256
GLOBAL_BATCH_SIZE=128
VAE="ema"
NUM_WORKERS=8
MAX_COUNT_PER_SPLIT=72000 # Dataset is generated in a roundrobin fashion (minimum is 722 images per class.)
NUM_GPUS=1
SEED=0
NUM_SAMPLES=None # Use None to select all samples.

torchrun --nnodes=1 --rdzv_endpoint=127.0.0.0:29501 --nproc_per_node=$NUM_GPUS src/datasets/extract_vae_features.py \
  --data-path "$DATA_PATH" \
  --features-path "$FEATURES_PATH" \
  --image-size "$IMAGE_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --global-seed "$SEED" \
  --num-samples "$NUM_SAMPLES" \
  --max_count_per_split "$MAX_COUNT_PER_SPLIT" \
  --vae "$VAE" \
  --num-workers "$NUM_WORKERS"