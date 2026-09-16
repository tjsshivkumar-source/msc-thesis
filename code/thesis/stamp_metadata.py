"""Stamp latent receipts into metadata.csv.
Encoders wrote latent .npz files to disk but receipts never merged into the
ledger (tes25-thesis/code/TRELLIS.2/datasets/metadata.csv) under 
the generic names the dataset class filters on."""
# The trainer's dataset class decides which assets are usable by reading columns in metadata.csv
#Pipeline:
#glob every .npz in each latent directory → 
# each filename is a sha256 → 
# mark that row *_encoded=True → 
# open each npz and count coords rows = the token count 
# the size filter needs (each latent is a sparse set of occupied-voxel tokens) → 
# fill blank aesthetic scores with 0.0 (config filter is 0.0, so nothing gets dropped) → 
# write to metadata.csv 
import os, glob
import numpy as np
import pandas as pd

ROOT = '/vol/gpudata/tes25-thesis/code/TRELLIS.2/datasets/OXL_proof'
meta = pd.read_csv(f'{ROOT}/metadata.csv') #metadata.csv defined

def stamp(latent_dir, flag_col, tokens_col):
    files = glob.glob(f'{ROOT}/{latent_dir}/*.npz')
    print(f'{latent_dir}: {len(files)} npz files')
    tokens = {}
    for f in files:
        sha = os.path.splitext(os.path.basename(f))[0]
        try:
            d = np.load(f)
            if 'coords' in d.files:
                tokens[sha] = len(d['coords'])          # sparse latents: token count
            else:
                tokens[sha] = int(np.prod(d['z'].shape)) # dense latents (ss): fixed size
        except Exception as e:
            print(f'  unreadable {sha}: {e}')
    meta[flag_col] = meta['sha256'].isin(tokens.keys())
    meta[tokens_col] = meta['sha256'].map(tokens)
    print(f'  stamped {meta[flag_col].sum()} rows -> {flag_col}, {tokens_col}')

#STAMP FUNC: 
# Arguments: 
#   (1) directory to scan ( encoder-named subfolder where .npz files lande)
#   (2) the boolean column to write (shape_latent_encoded, the eact name the dataset class filters on, from latent_key='shape_latent')
#   (3) the token-count column (shape_latent_tokens, for the max_tokens filter)

stamp('shape_latents/shape_enc_next_dc_f16c32_fp16_512', 'shape_latent_encoded', 'shape_latent_tokens')

#BELOW: same for the sparse-structure latents under generic names

stamp('ss_latents/ss_enc_conv3d_16l8_fp16_64',           'ss_latent_encoded',    'ss_latent_tokens')

meta['aesthetic_score'] = meta.get('aesthetic_score', 0.0)
meta['aesthetic_score'] = meta['aesthetic_score'].fillna(0.0)

# Conditioning-render receipts (ImageConditionedMixin filters 'cond_rendered');
# renders_cond/ holds one entry per asset named by sha256 — existence = receipt
rc = glob.glob(f'{ROOT}/renders_cond/*')
rc_shas = {os.path.splitext(os.path.basename(p))[0] for p in rc}
meta['cond_rendered'] = meta['sha256'].isin(rc_shas)
print(f"renders_cond: {len(rc_shas)} assets -> stamped {meta['cond_rendered'].sum()} rows")

meta.to_csv(f'{ROOT}/metadata.csv', index=False)
print('metadata.csv updated')