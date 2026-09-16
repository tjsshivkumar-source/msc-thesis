import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
print("OpenCV version:", cv2.__version__)
print("EXR in build:", "EXR" in cv2.getBuildInformation() and "OpenEXR" in cv2.getBuildInformation())
img = cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED)
print("cv2 EXR load:", "SUCCESS" if img is not None else "FAILED")

import imageio.v3 as iio
try:
    exr = iio.imread('assets/hdri/forest.exr')
    print("imageio EXR load: SUCCESS, shape:", exr.shape, "dtype:", exr.dtype)
except Exception as e:
    print("imageio EXR load: FAILED:", e)