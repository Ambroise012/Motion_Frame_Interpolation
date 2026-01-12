import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import cv2

from src.train_horn_unet import UNet
from utils.utils import generate_horn_interpolation_tensor

# Configuration
INPUT_DIR = "mickey_original"
OUTPUT_DIR = "output/output_unet_horn"
MODEL_PATH = "models/unet_50_epoch_res_15.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    with torch.no_grad():

        # img1, img2: (1, 1, H, W)
        mid_initial = generate_horn_interpolation_tensor(
            img1.to(device),
            img2.to(device),
            device
        ).squeeze(0)  # -> (1, H, W)

        residual_pred = model(
            mid_initial.unsqueeze(0),  # (1, 1, H, W)
            img1.to(device),
            img2.to(device)
        )

        mid_corrected = torch.clamp(
            mid_initial + residual_pred[0] / RES_SCALE,
            0, 1
        )

    return mid_corrected

def main():
    # Configuration
    INPUT_DIR = "mickey_original"
    OUTPUT_DIR = "output/output_unet_horn"
    MODEL_PATH = "models/unet_50_epoch_res_15.pth"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Charger le modèle
    device = torch.device("cpu")
    model = UNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Lister les images
    images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not images:
        raise ValueError(f"Aucune image trouvée dans {INPUT_DIR}")

    out_idx = 0
    RES_SCALE = 15.0  # Même valeur que pendant l'entraînement

    for i in tqdm(range(len(images) - 1)):
        img1 = load_image(images[i])
        img2 = load_image(images[i + 1])

        # Sauvegarder l'image originale
        save_image(img1, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1

        # Générer l'image intermédiaire
        mid_corrected = generate_intermediate_frames(model, img1, img2, device, RES_SCALE)

        # Sauvegarder l'image intermédiaire
        save_image(mid_corrected, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))
        out_idx += 1

    # Sauvegarder la dernière image
    last = load_image(images[-1])
    save_image(last, os.path.join(OUTPUT_DIR, f"frame_{out_idx:05d}.png"))

    print("Done!")

if __name__ == "__main__":
    main()
