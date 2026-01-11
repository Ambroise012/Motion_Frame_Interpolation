import os
import glob
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from frame_interpolation.eval import util
from frame_interpolation.eval.interpolator import Interpolator

from utils.utils import evaluate_film

# config
INPUT_DIR = "mickey_original"
OUTPUT_DIR = "output/output_film"
MODEL_PATH = "pretrained_models/film_net/Style/saved_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path):
    img = util.read_image(path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def save_image(img, path):
    img = tf.squeeze(img, axis=0)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    tf.io.write_file(path, tf.image.encode_png(img))

def main():
    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))

    interpolator = Interpolator(MODEL_PATH)

    out_idx = 0

    mse_list = []
    psnr_list = []
    ssim_list = []

    for i in tqdm(range(len(images) - 1)):
        img1 = load_image(images[i])
        img2 = load_image(images[i + 1])

        # save original img
        save_image(img1, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1

        time = tf.constant([0.5], dtype=tf.float32)

        mid = interpolator.interpolate(
            img1,
            img2,
            time
        )

        save_image(mid, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1
    
    # save last frame
    last = load_image(images[-1])
    save_image(last, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
    print("Done !")
if __name__ == "__main__":
    main()
