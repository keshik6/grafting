#!/bin/bash

# Define default values for the arguments
echo "Grafting Diffusion Transformers!"
NUM_SAMPLES=128k
INIT=graft
RESULTS_DIR="./results/stage2/$NUM_SAMPLES/$INIT/"
IMAGE_SIZE=256
NUM_CLASSES=1000
NUM_GPUS=8


# 50% Hyena-Y Hybrid
# CONFIG_FILEPATH=configs/train_stage2/imagenet_1k_256x256/graft_mha_50p_hyena_y.yaml
CONFIG_FILEPATH=configs/train_stage2/imagenet_1k_256x256/demo_graft_mha_hyena_y_3_swaps.yaml
echo $CONFIG_FILEPATH


#cd /workspace/
GPUS=$NUM_GPUS
echo "Number of GPUs: $GPUS"


accelerate launch --multi-gpu --num_processes $NUM_GPUS \
  --main_process_port 29512 --mixed_precision bf16 src/train_stage2.py \
  --result-dir "$RESULTS_DIR" \
  --image-size $IMAGE_SIZE \
  --num-classes $NUM_CLASSES \
  --config-filepath $CONFIG_FILEPATH




