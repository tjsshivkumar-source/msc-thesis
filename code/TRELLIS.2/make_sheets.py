import os, glob, re
# get PyTorch to use resizable meory segments so that feragmented space gets reused 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch
#Image for opening images, ImageDraw for writing on top of images
from PIL import Image, ImageDraw
from trellis2.pipelines import Trellis2ImageTo3DPipeline
#rendering util to get mesh
from trellis2.utils import render_utils
#Imath for math types (float) used by the OpenEXR API
import OpenEXR, Imath
from trellis2.renderers import EnvMap

f = OpenEXR.InputFile('assets/hdri/forest.exr'); dw = f.header()['dataWindow']
w, h = dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1; FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
rgb = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(h, w) for c in 'RGB']
envmap = EnvMap(torch.tensor(np.ascontiguousarray(np.stack(rgb,-1)), dtype=torch.float32, device='cuda'))

OUT_ROOT = '/vol/gpudata/tes25-thesis/outputs/patent_baseline'
#Yaw and Pitch list from which to render images for each mesh
YAWS = list(range(0, 360, 15)); PITCHES = [-45,-30,-15,0,15,30,45]; T = 192

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B"); pipeline.cuda()

#rendering function to render each mesh from the assigned yaw/pitch
def render(mesh, yaw, pitch):
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [np.deg2rad(yaw)], [np.deg2rad(pitch)], [2.0], [40.0])
    return render_utils.render_frames(mesh, extr, intr, {'resolution': T}, verbose=False, envmap=envmap)['shaded'][0]

#for all input pngs in the patent folders in patent_baseline
for inp in sorted(glob.glob(f'{OUT_ROOT}/*/input.png')):
    out = os.path.dirname(inp); name = os.path.basename(out)
    sheet_path = f'{out}/ring_sheet.png'
    if os.path.exists(sheet_path): continue
    print(f'=== {name} ===', flush=True)
    #create the mesh using the seed 42 that is used in the original generation
    mesh = pipeline.run(Image.open(inp), seed=42)[0]; mesh.simplify(16777216)
    #create the ring sheet 
    sheet = Image.new('RGB', (len(YAWS)*T, len(PITCHES)*T), 'white'); d = ImageDraw.Draw(sheet)
    #ITERATE THROUGH ALL PITCHES AND YAWS TO CREATE THE ANGLES AS SEEN IN THE RIGNT SHEET (use ImageDraw draw to draw on sheet)
    for r, p in enumerate(PITCHES):
        for c, y in enumerate(YAWS):
            sheet.paste(Image.fromarray(render(mesh, y, p)), (c*T, r*T))
            d.text((c*T+4, r*T+4), f'y{y} p{p}', fill='red')
    sheet.save(sheet_path); print(f'DONE {name}', flush=True)