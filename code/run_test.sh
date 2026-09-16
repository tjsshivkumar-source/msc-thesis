#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/%j.out

source /vol/gpudata/tes25-thesis/thesis_env/bin/activate
source /vol/cuda/12.0.0/setup.sh

cd /vol/gpudata/tes25-thesis/code
python test_gpu.py