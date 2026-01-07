#!/bin/bash

echo "Grafting Diffusion Transformers!"

# Train Scion Features
SPLIT=val # Set this to train or val
SHARD_SIZE=1000 # 8000 for training
# CONFIG_FILEPATH=configs/datasets/imagenet/scion_features/extract_scion_fts_$SPLIT.yaml
CONFIG_FILEPATH=configs/datasets/imagenet_1k_256x256/scion_fts/extract_$SPLIT.yaml
FEATURE_PATH="/data/"
IMAGE_SIZE=256
NUM_GPUS=1
SEED=0

# DIT_INDEXES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
DIT_INDEXES=6,16,27 # demo

TAR_STR_INDEXES=(split_000000.tar split_000001.tar split_000002.tar split_000003.tar split_000004.tar split_000005.tar split_000006.tar split_000007.tar)

if [[ "$SPLIT" == "train" ]]; then
  tar_indexes=("${TAR_STR_INDEXES[@]}")   # all
else
  tar_indexes=("${TAR_STR_INDEXES[0]}")   # first only (val case)
fi
echo "${tar_indexes[@]}"

for tar in "${tar_indexes[@]}"; do
  echo "Running on $tar ..."
  CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:27002 \
    src/datasets/extract_scion_fts_mha.py \
      --feature-path "$FEATURE_PATH" \
      --image-size "$IMAGE_SIZE" \
      --global-seed "$SEED" \
      --dit-block-indexes "$DIT_INDEXES" \
      --config-filepath "$CONFIG_FILEPATH" \
      --shard-size "$SHARD_SIZE" \
      --tar-str-indexes "$tar" \
      --shard-size "$SHARD_SIZE"
done
