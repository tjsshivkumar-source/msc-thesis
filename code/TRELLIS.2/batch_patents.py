import os, sys, glob, re, csv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import OpenEXR, Imath
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

DATA_ROOT = '/vol/gpudata/tes25-thesis/data/patent_tests/THESIS_IMGS'
OUT_ROOT  = '/vol/gpudata/tes25-thesis/outputs/patent_baseline'

#Environment map (direct OpenEXR load; see lab notes on imageio nondeterminism)
f = OpenEXR.InputFile('assets/hdri/forest.exr')
dw = f.header()['dataWindow']
w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
rgb = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(h, w) for c in ('R','G','B')]
envmap = EnvMap(torch.tensor(np.ascontiguousarray(np.stack(rgb, -1)), dtype=torch.float32, device='cuda'))

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

IMG_EXT = ('.png', '.jpg', '.jpeg')

def pick_conditioning(files):
    """Fixed rule: PERSP1 > any PERSP > FRONT > first sorted."""
    def find(pat):
        m = sorted(f for f in files if re.search(pat, os.path.basename(f), re.I))
        return m[0] if m else None
    for pat in [r'PERSP1', r'PERSP\d*', r'FRONT']:
        hit = find(pat)
        if hit: return hit
    return sorted(files)[0]

# --- Discover leaf item dirs ---
jobs = []
for dirpath, dirnames, filenames in os.walk(DATA_ROOT):
    imgs = [os.path.join(dirpath, fn) for fn in filenames if fn.lower().endswith(IMG_EXT)]
    if not imgs: continue
    item = os.path.basename(dirpath)
    m = re.match(r'(DM\d+)(?:_(LINE|PHOTO))?$', item)
    dm  = m.group(1) if m else item
    sty = m.group(2) if (m and m.group(2)) else 'DEFAULT'
    jobs.append((dm, sty, imgs))

print(f'{len(jobs)} items discovered')

SKIP = {'DM068850_DEFAULT'}   # pendant: remesh hang >3h on degenerate thin-wire geometry

manifest = []
for dm, sty, files in sorted(jobs):
    name = f'{dm}_{sty}'
    if name in SKIP:
        print(f'SKIPLIST {name}'); continue
    out = os.path.join(OUT_ROOT, name)
    if os.path.exists(os.path.join(out, 'mesh.glb')):
        print(f'SKIP {name} (already done)'); continue
    os.makedirs(out, exist_ok=True)
    cond = pick_conditioning(files)
    print(f'=== {name} | conditioning: {os.path.basename(cond)} ===', flush=True)
    try:
        img = np.array(Image.open(cond).convert('RGB')) #open image 
        alpha = 255 - (np.all(img > 240, axis=-1).astype(np.uint8) * 255)
        image = Image.fromarray(np.dstack([img, alpha]))
        image.save(os.path.join(out, 'input.png'))

        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)

        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(os.path.join(out, 'render.mp4'), video, fps=15)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]], decimation_target=1000000,
            texture_size=2048, remesh=True, remesh_band=1, remesh_project=0, verbose=False
        )
        glb.export(os.path.join(out, 'mesh.glb'), extension_webp=True)
        manifest.append([name, dm, sty, os.path.basename(cond), 'OK'])
        print(f'DONE {name}', flush=True)
    except Exception as e:
        manifest.append([name, dm, sty, os.path.basename(cond), f'FAILED: {e}'])
        print(f'FAILED {name}: {e}', flush=True)

os.makedirs(OUT_ROOT, exist_ok=True)
with open(os.path.join(OUT_ROOT, 'manifest.csv'), 'w', newline='') as fh:
    csv.writer(fh).writerows([['item','dm','style','cond_view','status']] + manifest)
print(f'Batch complete. {sum(1 for m in manifest if m[4]=="OK")}/{len(manifest)} succeeded.')