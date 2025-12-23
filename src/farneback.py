import cv2
import numpy as np
import os
from glob import glob
from skimage.metrics import structural_similarity as ssim

from utils.utils import evaluate

INPUT_DIR = "mickey_original"
OUTPUT_DIR = "output_farneback"
SIZE = (960, 720)

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.png")))

out_idx = 0

img_prev = cv2.imread(image_paths[0])
img_prev = cv2.resize(img_prev, SIZE)
cv2.imwrite(os.path.join(OUTPUT_DIR, f"img_{out_idx:04d}.png"), img_prev)
out_idx += 1

mse_list = []
psnr_list = []
ssim_list = []

for i in range(len(image_paths) - 1):
    print(f"Processing {i} -> {i+1}")

    img0 = img_prev
    img1 = cv2.imread(image_paths[i + 1])
    img1 = cv2.resize(img1, SIZE)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1,
        None,
        pyr_scale=0.7,
        levels=2,
        winsize=7,
        iterations=10,
        poly_n=5,
        poly_sigma=1.0,
        flags=0
    )

    u = flow[..., 0]
    v = flow[..., 1]

    mag = np.sqrt(u*u + v*v)

    # remove
    u[mag > 30.0] = 0
    v[mag > 30.0] = 0

    max_disp = 6.0
    scale = np.minimum(1.0, max_disp / (mag + 1e-6))
    u *= scale
    v *= scale
    
    # smoothing
    u = cv2.GaussianBlur(u, (3,3), 0)
    v = cv2.GaussianBlur(v, (3,3), 0)

    h, w = gray0.shape
    y, x = np.mgrid[0:h, 0:w]

    # interpolation
    mid = cv2.remap(
        img0,
        (x - u).astype(np.float32),
        (y - v).astype(np.float32),
        cv2.INTER_LINEAR,
    )


    cv2.imwrite(os.path.join(OUTPUT_DIR, f"img_{out_idx:04d}.png"), mid)
    out_idx += 1

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"img_{out_idx:04d}.png"), img1)
    out_idx += 1

    img_prev = img1

print("Done !")
