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

def evaluate_film(pred, gt):
    """
    pred, gt : tf.Tensor ou np.ndarray
               shape [1, H, W, 3] ou [H, W, 3]
               valeurs dans [0,1]
    """

    # --- TensorFlow -> NumPy ---
    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    if hasattr(gt, "numpy"):
        gt = gt.numpy()

    # --- Remove batch dimension ---
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]

    # --- Convert to uint8 [0,255] ---
    pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    gt_u8   = np.clip(gt * 255.0, 0, 255).astype(np.uint8)

    # --- Float version for MSE ---
    pred_f = pred_u8.astype(np.float32)
    gt_f   = gt_u8.astype(np.float32)

    # --- Metrics ---
    mse = np.mean((pred_f - gt_f) ** 2)
    psnr = cv2.PSNR(pred_f, gt_f)

    pred_g = cv2.cvtColor(pred_u8, cv2.COLOR_BGR2GRAY)
    gt_g   = cv2.cvtColor(gt_u8, cv2.COLOR_BGR2GRAY)

    ssim_score = ssim(pred_g, gt_g, data_range=255)

    return mse, psnr, ssim_score
