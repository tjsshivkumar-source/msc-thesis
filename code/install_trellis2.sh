#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/install_clean_%j.out
#SBATCH --time=02:00:00

set -x

CONDA_DIR=/vol/gpudata/tes25-thesis/miniconda3
BASEDIR=/vol/gpudata/tes25-thesis/code/TRELLIS.2
$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Remove old environment
$CONDA_DIR/bin/conda env remove -n trellis2 -y 2>/dev/null

# Create new environment
$CONDA_DIR/bin/conda create -n trellis2 python=3.10 -y

# Use the environment's pip and python directly by full path
ENV=$CONDA_DIR/envs/trellis2
PIP=$ENV/bin/pip
PYTHON=$ENV/bin/python

echo "=== Python version ==="
$PYTHON --version

# Install PyTorch
$PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== PyTorch CUDA check ==="
$PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.is_available()}')"

export CUDA_HOME=/vol/cuda/12.4.0
export PATH=$CUDA_HOME/bin:$PATH

# Basic deps
$PIP install pillow imageio imageio-ffmpeg numpy scipy tqdm opencv-python-headless transformers huggingface_hub safetensors einops jaxtyping kornia timm

echo "=== Installing flash-attn ==="
$PIP install flash-attn==2.7.3 --no-build-isolation 2>&1 | tail -30

echo "=== Installing nvdiffrast ==="
$PIP install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation 2>&1 | tail -30

echo "=== Installing nvdiffrec ==="
$PIP install ${BASEDIR}/trellis2/renderers/nvdiffrec_render --no-build-isolation 2>&1 | tail -30

echo "=== Installing o-voxel ==="
cd ${BASEDIR}/o-voxel
$PIP install -e . --no-build-isolation 2>&1 | tail -30

echo "=== Installing cumesh ==="
$PIP install ${BASEDIR}/trellis2/modules/cumesh --no-build-isolation 2>&1 | tail -30

echo "=== Installing flexgemm ==="
$PIP install ${BASEDIR}/trellis2/modules/flexgemm --no-build-isolation 2>&1 | tail -30

echo "=== Final Verification ==="
cd ${BASEDIR}
$PYTHON -c "import torch; print(f'torch OK: {torch.cuda.is_available()}, CUDA {torch.version.cuda}')"
$PYTHON -c "import o_voxel; print('o_voxel OK')"
$PYTHON -c "import flash_attn; print('flash_attn OK')"
$PYTHON -c "import cumesh; print('cumesh OK')"
$PYTHON -c "import nvdiffrast; print('nvdiffrast OK')"
$PYTHON -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('trellis2 pipeline OK')"
echo "=== ALL DONE ==="