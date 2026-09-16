#!/bin/bash

# ---------------------------------------------------------------------------
# Fine-tuning data pipeline, stage: conditioning-image rendering
# Renders 4 views per processed Objaverse asset (the fine-tuning corpus, ~63
# meshes in datasets/OXL_proof) using the toolkit's renderer, so images
# match the format/cameras the TRELLIS.2 trainer expects. These  are
# the INPUT half of each training pair (ground truth = dual-grid/voxel data
# already built); ft_canny.py will convert them to line-drawing style for
# the sparse input adaptation
#
# Second command updates metadata.csv so
# downstream stages (latent encoding) know which assets have renders.
# Partition: a16 (idle nodes, instant start; job is CPU/Blender-dominated,
# GPU slot unused but required by cluster config). Patent evaluation set
# (outputs/patent_baseline) is untouched (train/test separation).
# ---------------------------------------------------------------------------

#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/ftrc_%j.out
#SBATCH --time=02:00:00
eval "$(/vol/gpudata/tes25-thesis/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
cd /vol/gpudata/tes25-thesis/code/TRELLIS.2
python data_toolkit/render_cond.py ObjaverseXL --root datasets/OXL_proof --num_cond_views 4
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/OXL_proof
echo "=== RENDER_COND DONE ==="