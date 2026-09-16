#!/bin/bash
# ---------------------------------------------------------------------------
# Fine-tuning data pipeline, stage: latent encoding (GPU)
# Encodes the 84-asset corpus through the frozen SC-VAE: shape latents at 512
# (stage-2 training targets), then SS latents at 64 (stage-1 targets).
# encode_pbr_latent SKIPPED: fine-tune targets shape stages only.
# shape_latent_name verified from disk after shape encoding (ls line).
# ---------------------------------------------------------------------------
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/vol/gpudata/tes25-thesis/logs/ftenc_%j.out
#SBATCH --time=03:00:00

eval "$(/vol/gpudata/tes25-thesis/miniconda3/bin/conda shell.bash hook)"
conda activate trellis2
export HF_HOME=/vol/gpudata/tes25-thesis/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /vol/gpudata/tes25-thesis/code/TRELLIS.2

echo "=== ENCODE SHAPE LATENTS (512) ==="
python data_toolkit/encode_shape_latent.py --root datasets/OXL_proof --resolution 512

echo "=== METADATA UPDATE ==="
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/OXL_proof

echo "=== DISK CHECK: shape latent dir name ==="
ls datasets/OXL_proof/ | grep -i "shape"

echo "=== ENCODE SS LATENTS (64) ==="
python data_toolkit/encode_ss_latent.py --root datasets/OXL_proof --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64

echo "=== FINAL METADATA ==="
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/OXL_proof
echo "=== ENCODE DONE ==="