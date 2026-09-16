#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=64G
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/sheets_%j.out
#SBATCH --time=01:30:00

ENV=/vol/gpudata/tes25-thesis/miniconda3/envs/trellis2
export CUDA_HOME=/vol/cuda/12.4.0
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
export HF_TOKEN=hf_YOURTOKENHERE

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
$ENV/bin/python make_sheets.py