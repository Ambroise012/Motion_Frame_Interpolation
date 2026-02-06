import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import cv2

from src.train_horn_unet import UNet
from utils.utils import generate_horn_interpolation_tensor
from skimage.metrics import structural_similarity as ssim


def evaluate(pred, gt):
    """
    pred : torch.Tensor (1, H, W) dans [0,1]
    gt   : torch.Tensor (1, 1, H, W) dans [0,1]
    """

    pred_np = (pred[0].detach().cpu().numpy() * 255.0)
    gt_np = (gt[0, 0].detach().cpu().numpy() * 255.0)

    mse = np.mean((pred_np - gt_np) ** 2)

    psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 1e-10 else 100.0

    ssim_score = ssim(
        gt_np,
        pred_np,
        data_range=255.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return mse, psnr, ssim_score


def load_image(path):
    """Charge une image en niveaux de gris et la convertit en tenseur PyTorch"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Erreur de chargement de l'image {path}")
    img = cv2.resize(img, (320, 240))  # Même taille que pendant l'entraînement
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img_tensor

def save_image(tensor, path):
    """Sauvegarde un tenseur PyTorch en tant qu'image"""
    img = tensor.squeeze(0).squeeze(0).cpu().numpy()  # Supprime les dimensions batch et channel
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)

def generate_intermediate_frames(model, img1, img2, device="cpu", RES_SCALE=15.0):
    model.eval()

    img1 = img1.to(device)   # (1, 1, H, W)
    img2 = img2.to(device)   # (1, 1, H, W)

    with torch.no_grad():
        # Horn initial : (1, 1, H, W)
        mid_initial = generate_horn_interpolation_tensor(
            img1,
            img2,
            device
        )

        # concat EXACTEMENT comme à l'entraînement
        x = torch.cat([
            mid_initial,  # (1, 1, H, W)
            img1,         # (1, 1, H, W)
            img2          # (1, 1, H, W)
        ], dim=1)           # -> (1, 3, H, W)

        residual_pred = model(x)   # (1, 1, H, W)

        mid_corrected = torch.clamp(
            mid_initial + residual_pred / RES_SCALE,
            0, 1
        )

    return mid_corrected.squeeze(0)   # (1, H, W)


def main():
    INPUT_DIR = "mickey_original"
    OUTPUT_DIR = "output/outputs_mickey_origin_unet_FT"
    MODEL_PATH = "models/unet_50_epoch_improv-unet.pth"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if len(images) < 3:
        raise ValueError("Pas assez d'images pour l'interpolation")

    RES_SCALE = 10.0
    out_idx = 0

    mse_list, psnr_list, ssim_list = [], [], []

    # --- interpolation 1 image sur 2 ---
    for i in tqdm(range(1, len(images) - 4, 3)):
        img1 = load_image(images[i])       # t-1 observée
        img_gt = load_image(images[i + 1]) # t (GT non observée)
        img2 = load_image(images[i + 2])   # t+1 observée

        # sauvegarde image observée
        save_image(img1, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1

        # interpolation
        mid_pred = generate_intermediate_frames(
            model, img1, img2, device, RES_SCALE
        )

        save_image(mid_pred, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1

        # --- évaluation ---
        mse, psnr, ssim_score = evaluate(mid_pred, img_gt)

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim_score)

        print(f"\nFrame {i+1} (GT)")
        print(f"  MSE  : {mse:.6f}")
        print(f"  PSNR : {psnr:.2f} dB")
        print(f"  SSIM : {ssim_score:.4f}")

    print("\n===== RESULTS =====")
    print(f"MSE  mean : {np.mean(mse_list):.6f}")
    print(f"PSNR mean : {np.mean(psnr_list):.2f} dB")
    print(f"SSIM mean : {np.mean(ssim_list):.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
