#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/batch_%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=64G

ENV=/vol/gpudata/tes25-thesis/miniconda3/envs/trellis2
export CUDA_HOME=/vol/cuda/12.4.0
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
export HF_TOKEN=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

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
$ENV/bin/python batch_patents.py