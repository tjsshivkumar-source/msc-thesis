#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/patch_%j.out
#SBATCH --time=01:00:00

set -x

ENV=/vol/gpudata/tes25-thesis/miniconda3/envs/trellis2
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

export CUDA_HOME=/vol/cuda/12.4.0
export PATH=$CUDA_HOME/bin:$PATH
export PIP_CACHE_DIR=/vol/gpudata/tes25-thesis/.pip_cache
export TMPDIR=/vol/gpudata/tes25-thesis/.tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

echo "=== Installing psutil + ninja ==="
$PIP install psutil ninja

echo "=== Installing flash-attn ==="
$PIP install flash-attn==2.7.3 --no-build-isolation 2>&1 | tail -20

echo "=== Installing nvdiffrec from GitHub ==="
rm -rf /tmp/extensions/nvdiffrec
mkdir -p /tmp/extensions
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
$PIP install /tmp/extensions/nvdiffrec --no-build-isolation 2>&1 | tail -20

$PIP install git+https://github.com/EasternJournalist/utils3d.git easydict pandas open3d rembg onnxruntime

echo "=== Final Verification ==="
cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
$PYTHON -c "import torch; print(f'torch OK: {torch.cuda.is_available()}, CUDA {torch.version.cuda}')"
$PYTHON -c "import o_voxel; print('o_voxel OK')"
$PYTHON -c "import flash_attn; print('flash_attn OK')"
$PYTHON -c "import cumesh; print('cumesh OK')"
$PYTHON -c "import utils3d; print('utils3d OK')"
$PYTHON -c "import nvdiffrast; print('nvdiffrast OK')"
$PYTHON -c "from trellis2.utils import render_utils; print('render_utils OK')"
$PYTHON -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('trellis2 pipeline OK')"
echo "=== ALL DONE ==="