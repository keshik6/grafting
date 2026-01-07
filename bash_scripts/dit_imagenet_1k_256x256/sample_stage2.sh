

echo "Grafting Diffusion Transformers!"

# Qualitative
NUM_GPUS=1
# Demo (3 swaps: MHA -> Hyena-Y)
CONFIG_FILEPATH=configs/evals/demo_stage2.yaml
echo $CONFIG_FILEPATH


#cd /workspace/
GPUS=$NUM_GPUS
echo "Number of GPUs: $GPUS"

# Stage 1: Qualitative evals
python src/sample.py \
  --config-filepath $CONFIG_FILEPATH \
  --cfg-scale 4.0 \
  --num-sampling-steps 250

# Sample 50k samples for FID, sFID, Precision and Recall calculation
GPUS=8
CONFIG_FILEPATH=configs/evals/demo_stage2.yaml
echo $CONFIG_FILEPATH
torchrun --rdzv_endpoint=localhost:29500 --nnodes=1 --nproc_per_node=$GPUS src/sample_ddp.py \
        --num-fid-samples 50000 --per-proc-batch-size=64 --config-filepath $CONFIG_FILEPATH