#!/bin/bash

# Define default values for the arguments
echo "Grafting Diffusion Transformers!"
RESULTS_DIR="./results/stage1/" # All results for this study go here.

#RESULTS_DIR="./tmp/"
IMAGE_SIZE=256
NUM_CLASSES=1000
NUM_GPUS=1
LOG_EVERY=50 # This is for saving checkpoints (save every 25 epoch)


# cd /workspace/
GPUS=$NUM_GPUS
echo "Number of GPUs: $GPUS"

# Layer selection
# BLOCK_INDEXES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)
BLOCK_INDEXES=(6 16 27)


for BLOCK_INDEX in ${BLOCK_INDEXES[*]}; do
  CONFIG_FILEPATH=configs/train_stage1/imagenet_1k_256x256/hyena_y.yaml
  
  # Supports multi-gpu
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes $NUM_GPUS \
    --main_process_port 29514 --mixed_precision bf16 src/train_stage1.py \
    --result-dir "$RESULTS_DIR" \
    --image-size $IMAGE_SIZE \
    --num-classes $NUM_CLASSES \
    --config-filepath $CONFIG_FILEPATH \
    --block-index $BLOCK_INDEX \
    --log-every $LOG_EVERY \
    --loss "l1"

    # --scaled_predictor --loss "huber" --huber_delta 1.0 (Use this for Huber)
done

