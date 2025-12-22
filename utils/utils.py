import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim

def evaluate(pred, gt):
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    mse = np.mean((pred_f - gt_f) ** 2)
    psnr = cv2.PSNR(pred_f, gt_f)

    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(pred_g, gt_g, data_range=255)

    return mse, psnr, ssim_score