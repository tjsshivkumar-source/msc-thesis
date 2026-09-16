import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

print("Load environment map...")

import OpenEXR, Imath
import numpy as np

f = OpenEXR.InputFile('assets/hdri/forest.exr') # open the EXR file 
dw = f.header()['dataWindow'] #read EXR file dimensions directly
w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
FLOAT = Imath.PixelType(Imath.PixelType.FLOAT) # request pixel data as 32 bit floats to preserve HDR radiance values
rgb = [np.frombuffer(f.channel(c, FLOAT), dtype=np.float32).reshape(h, w) for c in ('R', 'G', 'B')] #pull each channel individually as raw bytes -> convert to float 32 array in numpy -> reshape to (h, w)
exr = np.stack(rgb, axis=-1) #combine the 3 RGB planes into one (h, w, 3) image
print("EXR loaded shape:", exr.shape, "dtype:", exr.dtype, "min/max:", exr.min(), exr.max()) # verify that run loaded float32 (32 bit decimals) with HDR range (NOT capping it due to how large HDR radiance values range)
envmap = EnvMap(torch.tensor(np.ascontiguousarray(exr), dtype=torch.float32, device='cuda')) #contiguous float32 CUDA tensor into the renderer's environment map

# the HDRI provides lighting for visualisation render, so RGB vals are radiance, alpha channel on lighting is meaningless
#np.ascontiguousarray guarantees clean memory layout by storing it in c-contigous (row-major order + one contiguous memory block)

print("Loading pipeline (downloads ~10GB on first run)...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

print("Run inference on example image...")
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)

print("Rendering video...")
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("/vol/gpudata/tes25-thesis/logs/test_output.mp4", video, fps=15)

glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces,
    attr_volume=mesh.attrs, coords=mesh.coords,
    attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000, texture_size=4096,
    remesh=True, remesh_band=1, remesh_project=0, verbose=True
)
glb.export("/vol/gpudata/tes25-thesis/outputs/test.glb", extension_webp=True)

print("Success; Video saved to logs/test_output.mp4; glb saved to outputs as test.glb")