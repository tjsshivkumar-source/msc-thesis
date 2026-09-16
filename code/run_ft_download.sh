#!/bin/bash
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/ftdl_%j.out
#SBATCH --time=02:00:00

eval "$(/vol/gpudata/tes25-thesis/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
export HF_TOKEN=hf_YOURTOKEN

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
python data_toolkit/download.py ObjaverseXL --root datasets/OXL_proof --world_size 8000
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/OXL_proof
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/OXL_proof
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/OXL_proof
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/OXL_proof
python data_toolkit/dual_grid.py ObjaverseXL --root datasets/OXL_proof --resolution 512
python data_toolkit/voxelize_pbr.py ObjaverseXL --root datasets/OXL_proof --resolution 512
echo "=== GATE 1 CPU STAGES DONE ==="
