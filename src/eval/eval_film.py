import os
import glob
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from frame_interpolation.eval import util
from frame_interpolation.eval.interpolator import Interpolator

from utils.utils import evaluate_film
from src.FILM import load_image

# config
INPUT_DIR = "mickey_original"
MODEL_PATH = "pretrained_models/film_net/Style/saved_model"

def main():
    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    interpolator = Interpolator(MODEL_PATH)

    mse_list = []
    psnr_list = []
    ssim_list = []

    out_idx = 0

    for i in tqdm(range(0, len(images) - 2, 2)):
        print(f"Processing frames {i}, {i+1}, {i+2}")

        img_t   = load_image(images[i])
        img_gt  = load_image(images[i + 1])  # GT intermédiaire
        img_t2  = load_image(images[i + 2])

        # save observed frame
        out_idx += 1

        # interpolate at t+0.5
        time = tf.constant([0.5], dtype=tf.float32)
        mid = interpolator.interpolate(img_t, img_t2, time)

        # --- evaluation ---
        mse, psnr, ssim_score = evaluate_film(mid, img_gt)

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
