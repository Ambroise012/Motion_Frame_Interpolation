import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

from utils.utils import evaluate
from src.horn_schunck import horn_schunck_grad

INPUT_DIR = "mickey_original"
SIZE = (960, 720)

SCALE = 1.0

def main():
    image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.png")))

    out_idx = 0

    img_prev = cv2.imread(image_paths[0])
    img_prev = cv2.resize(img_prev, SIZE)
    out_idx += 1

    mse_list = []
    psnr_list = []
    ssim_list = []

    for i in range(0,len(image_paths) - 2,2):
        print(f"Processing frames {i}, {i+1}, {i+2}")

        # --- observed frames ---
        img0 = cv2.imread(image_paths[i])
        img2 = cv2.imread(image_paths[i + 2])
        gt   = cv2.imread(image_paths[i + 1])  # GT intermédiaire

        img0 = cv2.resize(img0, SIZE)
        img2 = cv2.resize(img2, SIZE)
        gt   = cv2.resize(gt,   SIZE)

        # flow i -> i+1
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        Ix = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
        It = gray1 - gray0

        u01, v01 = horn_schunck_grad(Ix, Iy, It)

        # flow i+1 -> i
        Ix = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        It = gray0 - gray1

        u10, v10 = horn_schunck_grad(Ix, Iy, It)

        # warping
        h, w = u01.shape
        y, x = np.mgrid[0:h, 0:w]

        alpha = 0.5
        mid0 = cv2.remap(
            img0,
            (x - alpha * SCALE * u01).astype(np.float32),
            (y - alpha * SCALE * v01).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        mid1 = cv2.remap(
            img2,
            (x - alpha * SCALE * u10).astype(np.float32),
            (y - alpha * SCALE * v10).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        img_mid = ((mid0.astype(np.float32) + mid1.astype(np.float32)) * 0.5).astype(np.uint8)

        img_prev = img2

        # --- evaluation ---
        mse, psnr, ssim_score = evaluate(img_mid, gt)

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

if __name__ == "__main__":
    main()