#!/bin/bash
# Gate 2: VRAM/data feasibility tryrun for stage-2 flow transformer fine-tune.
# --tryrun = build dataset + model + trainer without training (train.py's own flag).
# Fresh init (no finetune weights yet): tryrun tests memory + data pipeline;
# released-checkpoint loading is wired only if this passes.
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/gate2_%j.out
#SBATCH --time=01:00:00

eval "$(/vol/gpudata/tes25-thesis/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
export HF_TOKEN=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
python train.py \
  --config configs/gen/ft_gate2.json \
  --output_dir /vol/gpudata/tes25-thesis/checkpoints/ft_tryrun \
  --data_dir datasets/OXL_proof \
  --tryrun --num_gpus 1
echo "=== GATE 2 TRYRUN DONE ==="