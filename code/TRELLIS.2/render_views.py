import os, sys, glob, re, csv
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
import OpenEXR, Imath
from trellis2.renderers import EnvMap

# Environment map (PBR renderer requires one) 
f = OpenEXR.InputFile('assets/hdri/forest.exr') #open HDRI file for lighting
dw = f.header()['dataWindow'] #read image dims
w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT) #force HDRI channels to decode as float32 so that HDR range preserved
rgb = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(h, w) for c in ('R','G','B')] #RGB stored as separate planes by EXR, each channel returns raw bytes
                                                                                                  #reinterpret as float32 using np.frombuffer, and get in 2D shape with .reshape(w, h)
envmap = EnvMap(torch.tensor(np.ascontiguousarray(np.stack(rgb, -1)), dtype=torch.float32, device='cuda')) #stack planes along a new axis to get (512, 1024, 3) HWC image
                                                                                                           #use np.ascontiguousarray for row major memory for CUDA transfer
OUT_ROOT = '/vol/gpudata/tes25-thesis/outputs/patent_baseline'
#manually declared offsets that detail what the front view of an object is (based on the canonical front view as presented int he patent details via WIPO)
OFFSETS_CSV = '/vol/gpudata/tes25-thesis/code/thesis/manual_offsets.csv'
RES = 1024

# Canonical camera table; applied to the mesh AFTER it is rotated to be front-facing
CAMERAS = {'FRONT': (0, 0), 'BACK': (180, 0), 'LEFT': (-90, 0), 'RIGHT': (90, 0),
           'TOP': (0, 89), 'BOTTOM': (0, -89)}
#assumed 3/4 perspective pose
PERSP_NOMINAL = (45, 20)
#sweep steps
SWEEP = [(dy, dp) for dy in range(-15, 16, 5) for dp in range(-15, 16, 5)]

# read from csv manually declared orientation per item: the (yaw, pitch) at which the mesh appears front-on and level
OFFSETS = {r['item']: (float(r['yaw']), float(r['pitch']))
           for r in csv.DictReader(open(OFFSETS_CSV))}

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

def canon_transform(yaw, pitch):
    """4x4 rotation that maps the declared (yaw,pitch) viewing direction onto canonical FRONT (0,0).
    Built from the renderer's extrinsics."""
    #camera facing manually declared front view (in manual_offsets.csv, where yaw + pitch inputs come from)
    #2.0 = distance from center (2 frames), 40.0 = 40º view cone 
    e_decl, _ = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [np.deg2rad(yaw)], [np.deg2rad(pitch)], [2.0], [40.0])
    #CANONICAL front camera view (faces 0,0 where the front view is actually meant to be)
    e_can, _ = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [0.0], [0.0], [2.0], [40.0])

    # NB: THE ABOVE TWO FUNCTIONS USE THE INTRINSICS (yaw/pitch combos and center dist/view width)
    #      TO CREATE AN EXTRINSICS MATRIX IN 4X4 USING yaw_pitch_r_fov_to_extrinsics_intrinsics
    #      WHICH IS DEFINED IN render_utils.py
    #   [ r r r t ]     top-left 3×3 = rotation (camera orientation,
    #   [ r r r t ]                     computed FROM yaw/pitch via the
    #   [ r r r t ]                     sin/cos look-at construction)
    #   [ 0 0 0 1 ]     right column = translation (from r=2.0 position)
    # e_decl[0][:3,:3] = a 3×3 of sines and cosines encoding "which way the declared camera faces"

    #Extrinsics Matrices: CONVERT POINT'S WORLD COORDINATES TO CAMERA'S COORDINATES (Translate a point's "front" to the camera so camera faces the front based on points)
    #e_decl[0][:3, :3] is [WORLD -> CAMERA] converts the vertex from world coords to the declared cameras view coords
    #e_decl makes this camera view the FRONT view ; in its view, the WORLD is wrong

    #e_can[0][:3, :3]^T is transpose (inverse) so it is doing [CAMERA -> WORLD (0,0)] transformation for the canonical camera
    #e_can^T works on: "GIVEN correctly posed view, where should vertices sit in real world so that canonical camera gets that view?"
    #e_can^T takes straight in front of viewr -> straight in front of yaw 0 camera (looking in -Y dir)
    # KEY: FRONT = FACING + Y -> FRONT OF OBJECT
    #   Transpose of e_can -> writes the reference frame back out to the world so that 
    #   the (0,0) camera is the one seeing "front = straight ahead."

    #e_decl re-expresses every vertex from world-frame coordinates into the declared camera's frame — 
    # same points, new numbers, now measured along "toward-the-camera / camera's-right / camera's-up" axes

    #Because that camera faces the actual front, the mesh's coordinates in this frame 
    # have the property "front = toward viewer."

    #e_canᵀ interprets those same numbers as if they were coordinates 
    # in the canonical camera's frame, and converts them back to world coordinates

    #the numbers are handed from one camera's frame to a different camera's frame without 
    # change; the reinterpretation IS the rotation
    R = e_can[0][:3, :3].T @ e_decl[0][:3, :3]
        # matrix products apply right-to-left on a vector
        #[0] just unwraps the one-element list the function returns; 
        #[:3,:3] slices the rotation block out of the 4×4, discarding the translation part
    Tm = torch.eye(4, device=R.device, dtype=R.dtype)
        #Tm is what transformation=Tm hands the renderer, which multiplies every vertex by it before projecting
    Tm[:3, :3] = R
    return Tm

def render_pose(mesh, yaw_deg, pitch_deg, Tm):
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [np.deg2rad(yaw_deg)], [np.deg2rad(pitch_deg)], [2.0], [40.0])
    rets = render_utils.render_frames(mesh, extr, intr, {'resolution': RES},
                                      verbose=False, envmap=envmap, transformation=Tm)
    return rets['shaded'][0]

for inp in sorted(glob.glob(f'{OUT_ROOT}/*/input.png')):
    out = os.path.dirname(inp); name = os.path.basename(out)
    vdir = os.path.join(out, 'views')
    if os.path.exists(os.path.join(vdir, 'offset.txt')):
        print(f'SKIP {name}'); continue
    if name not in OFFSETS:
        print(f'NO OFFSET for {name} - skipping'); continue
    os.makedirs(vdir, exist_ok=True)
    yaw0, pitch0 = OFFSETS[name]
    print(f'=== {name} | declared front at yaw {yaw0} pitch {pitch0} ===', flush=True)
    try:
        mesh = pipeline.run(Image.open(inp), seed=42)[0]
        mesh.simplify(16777216)
        Tm = canon_transform(yaw0, pitch0)
        for view, (yaw, pitch) in CAMERAS.items():
            Image.fromarray(render_pose(mesh, yaw, pitch, Tm)).save(f'{vdir}/{view}.png')
        for dy, dp in SWEEP:
            Image.fromarray(render_pose(mesh, PERSP_NOMINAL[0] + dy, PERSP_NOMINAL[1] + dp, Tm)
                            ).save(f'{vdir}/PERSP_{dy:+03d}_{dp:+03d}.png')
        with open(f'{vdir}/offset.txt', 'w') as fh:
            fh.write(f'{yaw0}\n{pitch0}\nmanual\n')
        print(f'DONE {name}', flush=True)
    except Exception as e:
        print(f'FAILED {name}: {e}', flush=True)
print('Render pass complete.')