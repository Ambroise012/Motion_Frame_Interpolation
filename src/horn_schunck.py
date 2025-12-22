import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

from utils.utils import evaluate

INPUT_DIR = "mickey_original"
OUTPUT_DIR = "output_horn"
SIZE = (960, 720)
SCALE = 1.0

def local_average(Z):
    return 0.25 * (
        np.roll(Z,  1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) +
        np.roll(Z, -1, axis=1)
    )

def horn_schunck_grad(
    Ix, Iy, It,
    alpha=1.0,
    max_iter=1000,
    eps=1e-4
):
    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)

    for _ in range(max_iter):
        u_bar = local_average(u)
        v_bar = local_average(v)

        A = Ix * u_bar + Iy * v_bar + It
        B = alpha**2 + Ix**2 + Iy**2

        u_new = u_bar - Ix * A / B
        v_new = v_bar - Iy * A / B

        # convergence criteria
        if max(
            np.max(np.abs(u_new - u)),
            np.max(np.abs(v_new - v))
        ) < eps:
            break

        u, v = u_new, v_new

    return u, v

def main():
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

        # flow i -> i+1
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

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
            img1,
            (x - alpha * SCALE * u10).astype(np.float32),
            (y - alpha * SCALE * v10).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        img_mid = ((mid0.astype(np.float32) + mid1.astype(np.float32)) * 0.5).astype(np.uint8)

        # save
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"img_{out_idx:04d}.png"), img_mid)
        out_idx += 1

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"img_{out_idx:04d}.png"), img1)
        out_idx += 1

        # eval
        mse, psnr, ssim_score = evaluate(img_mid, img1)

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim_score)

        print(f"  MSE  : {mse:.2f}")
        print(f"  PSNR : {psnr:.2f} dB")
        print(f"  SSIM : {ssim_score:.4f}")


        img_prev = img1 # for the next frame

    print("\n===== RESULTS =====")
    print(f"MSE  mean : {np.mean(mse_list):.2f}")
    print(f"PSNR mean : {np.mean(psnr_list):.2f} dB")
    print(f"SSIM mean : {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    main()