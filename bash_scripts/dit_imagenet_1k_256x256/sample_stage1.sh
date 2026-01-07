#!/bin/bash

echo "Grafting Diffusion Transformers!"

NUM_GPUS=1

# Demo (3 swaps: MHA -> Hyena-Y)
CONFIG_FILEPATH=configs/evals/demo_stage1.yaml
echo $CONFIG_FILEPATH


#cd /workspace/
GPUS=$NUM_GPUS
echo "Number of GPUs: $GPUS"

# Stage 1: Qualitative evals
python src/sample.py \
  --config-filepath $CONFIG_FILEPATH \
  --cfg-scale 4.0 \
  --num-sampling-steps 250