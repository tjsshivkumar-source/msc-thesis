# 3D-Mediated Visual Similarity Assessment for Design Patents

**Evaluating Reconstruction Feasibility Across Drawing Styles**

MSc Computing Individual Project — Imperial College London, September 2026
**Author:** Tejas Shivkumar · **Supervisor:** Dr Jiankang Deng

This repository contains all code, training and generation configurations, and evaluation data for my MSc thesis evaluating TRELLIS.2 (single-image-to-3D generation) on design-patent figures. As part of the thesis, I performed 26 reconstructions across 8 Locarno classes, scored against each filing's held-out views (150 comparisons, LPIPS/SSIM in raw and canny-converted edge modes). Additionally, I developed a qualitative failure taxonomy and a prepared fine-tuning pipeline targeting the line-drawing appearance gap.

The full report (56 pp.) documents the methodology, results, and findings; this README maps the repository to it (chapter/appendix references below point at the report).

---

## Repository map

```
.
├── code/
│   ├── run_renders.sh          # batch reconstruction + canonical/sweep rendering (§3.3–3.4)
│   ├── run_sheets.sh           # contact-sheet generation for orientation declaration (§3.5)
│   ├── run_ft_download.sh      # fine-tune corpus: Objaverse-XL asset download (§5.5)
│   ├── run_ft_render.sh        # fine-tune corpus: conditioning renders + Canny conversion
│   ├── run_ft_encode.sh        # fine-tune corpus: O-Voxel + latent encoding
│   ├── run_gate2.sh            # training dry-run: data interface + trainer construction
│   ├── thesis/
│   │   ├── evaluate.py         # normalization chain, LPIPS/SSIM (raw+canny-edge), self-tests (§3.6)
│   │   ├── manual_offsets.csv  # manually declared orientation offsets from ring sheets, locked pre-metrics (§3.5, §4.6)
│   │   └── stamp_metadata.py   # per-artifact metadata receipts for bookeeping amidst training pipeline issues (§5.5, App. A.2)
│   └── TRELLIS.2/              # fork of microsoft/TRELLIS.2 incl. repaired training toolkit
│                               #   (dataset modules transplanted from TRELLIS v1; App. A.2)
├── outputs/
│   └── patent_baseline/        # eval_results.csv (150 rows), per-item manifests, receipts
│                               #   render PNGs excluded from git (regenerable; see below)
├── requirements-frozen.txt     # full frozen environment (App. A.1)
└── .gitignore
```

## Environment

Experiments were ran on a single NVIDIA A40 (48 GB; 64 GB host RAM) under Slurm, Python 3.10 (conda). The load-bearing verson pins are listed below (each corresponds to a diagnosed incompatibility — report Appendix A.1):

- `transformers 4.57.6 (<5)` — v5 restructures `DINOv3ViTModel` used by the conditioning pathway
- `opencv-python-headless` — wheel ships without EXR; environment-map loads replaced with `imageio.v3` (+`OpenEXR`/`pyav` backends)
- `utils3d` pinned commit — later commits rename the intrinsics API
- full specification: `requirements-frozen.txt`

Setup commands:
```bash
conda create -n trellis2 python=3.10 -y && conda activate trellis2
pip install -r requirements-frozen.txt
cd code/TRELLIS.2 && . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

## Reproducing evaluation

1. **Reconstruct + render** — `code/run_renders.sh`: each design's conditioning view → TRELLIS.2 (fixed seed 42, deterministic) → textured mesh → six canonical views + 49-pose perspective sweep. Provenance copies of inputs and `.glb` meshes are written per item.
2. **Declare orientations** — `code/run_sheets.sh` produces contact sheets; declared (yaw, pitch) offsets live in `code/thesis/manual_offsets.csv` (locked before any metric computation).
3. **Evaluate** — `code/thesis/evaluate.py`: normalization (white-background standardisation, tight crop, pad-square, 512²), LPIPS + SSIM in raw and Canny-edge modes, metric self-tests. Output: `outputs/patent_baseline/eval_results.csv`.
4. **Fine-tuning pipeline (prepared; §5.5)** — `run_ft_download.sh` → `run_ft_render.sh` → `run_ft_encode.sh` builds the paired corpus from ObjaverseXL(84-asset proof corpus completed); `run_gate2.sh` verifies the data interface, 1.29 B-parameter denoiser, and trainer construction on a single A40.

Every thesis table is reproducible from `eval_results.csv` (e.g. `df.groupby('swept')[['lpips','ssim']].mean()`).

## Data availability

Render images and mesh binaries are excluded from git (size); they are regenerable deterministically from the fixed seed via step 1, and archived copies are available on request (locally stored). `eval_results.csv`, manifests, offsets, and receipts are included.

## Working history

Development was conducted on the departmental cluster with this repository serving as milestone snapshots; large files were stripped from history at submission (`git filter-repo`). The dated baseline commit ("Initial commit: baseline pipeline, 26/26 full-res reconstructions") marks the mid-project state; the full working tree is included, and further working history is preserved on the cluster.

## Acknowledgements & licensing

`code/TRELLIS.2/` derives from [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and retains its upstream license; modifications (dataset-module transplant from TRELLIS v1, interface adaptations, provenance stamping) are documented in report Appendix A.2. Fine-tune corpus assets derive from Objaverse-XL under their respective licenses.
