
import cv2
import numpy as np
import os
from glob import glob

from utils.utils import evaluate

INPUT_DIR = "mickey_original"
SIZE = (960, 720)

image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.png")))

mse_list = []
psnr_list = []
ssim_list = []

for i in range(0, len(image_paths) - 2, 2):
    print(f"Processing frames {i}, {i+1}, {i+2}")

    # --- observed frames ---
    img0 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i + 2])
    gt   = cv2.imread(image_paths[i + 1])  # GT intermédiaire

    img0 = cv2.resize(img0, SIZE)
    img2 = cv2.resize(img2, SIZE)
    gt   = cv2.resize(gt,   SIZE)

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- optical flow I0 -> I2 ---
    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray2,
        None,
        pyr_scale=0.7,
        levels=2,
        winsize=7,
        iterations=10,
        poly_n=5,
        poly_sigma=1.0,
        flags=0
    )

    u = flow[..., 0] * 0.5
    v = flow[..., 1] * 0.5

    mag = np.sqrt(u*u + v*v)

    # --- clamp large motions ---
    u[mag > 30.0] = 0
    v[mag > 30.0] = 0

    # --- smoothing ---
    u = cv2.GaussianBlur(u, (3,3), 0)
    v = cv2.GaussianBlur(v, (3,3), 0)

    h, w = gray0.shape
    y, x = np.mgrid[0:h, 0:w]

    # --- interpolation at t+0.5 ---
    mid = cv2.remap(
        img0,
        (x - u).astype(np.float32),
        (y - v).astype(np.float32),
        cv2.INTER_LINEAR,
    )

    # --- evaluation ---
    mse, psnr, ssim_score = evaluate(mid, gt)

    mse_list.append(mse)
    psnr_list.append(psnr)
    ssim_list.append(ssim_score)

    print(f"  MSE  : {mse:.6f}")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim_score:.4f}")

print("\n===== RESULTS =====")
print(f"MSE  mean : {np.mean(mse_list):.6f}")
print(f"PSNR mean : {np.mean(psnr_list):.2f} dB")
print(f"SSIM mean : {np.mean(ssim_list):.4f}")



