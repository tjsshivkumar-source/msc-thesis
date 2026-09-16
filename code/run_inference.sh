#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/inference_%j.out
#SBATCH --time=01:00:00

ENV=/vol/gpudata/tes25-thesis/miniconda3/envs/trellis2
export CUDA_HOME=/vol/cuda/12.4.0
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
mkdir -p $HF_HOME

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
export HF_TOKEN=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
$ENV/bin/python test_inference.py