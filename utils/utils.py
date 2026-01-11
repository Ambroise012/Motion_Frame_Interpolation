import numpy as np
import cv2
import os
import torch 

from skimage.metrics import structural_similarity as ssim
from src.horn_schunck import horn_schunck_grad

# ----- DL ------
def gradient_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)


# ---- horn - DL ----
def extract_frames(video_path, output_dir, num_frames=2):
    """Extraire des paires d'images consécutives depuis une vidéo."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (960, 720))
            frames.append(frame)

    cap.release()
    return frames


# def generate_horn_interpolation(img0, img1, SCALE=1.0):
#     gray0 = img0.astype(np.float32) / 255.0
#     gray1 = img1.astype(np.float32) / 255.0

#     Ix = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
#     Iy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
#     It = gray1 - gray0

#     u01, v01 = horn_schunck_grad(Ix, Iy, It)

#     Ix = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
#     Iy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
#     It = gray0 - gray1

#     u10, v10 = horn_schunck_grad(Ix, Iy, It)

#     h, w = u01.shape
#     y, x = np.mgrid[0:h, 0:w]

#     alpha = 0.5
#     mid0 = cv2.remap(
#         img0,
#         (x - alpha * SCALE * u01).astype(np.float32),
#         (y - alpha * SCALE * v01).astype(np.float32),
#         cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )

#     mid1 = cv2.remap(
#         img1,
#         (x - alpha * SCALE * u10).astype(np.float32),
#         (y - alpha * SCALE * v10).astype(np.float32),
#         cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )

#     img_mid = ((mid0.astype(np.float32) + mid1.astype(np.float32)) * 0.5).astype(np.uint8)

#     return img_mid

def generate_horn_interpolation(img0, img1, SCALE=1.0):
    # Assure-toi que img0 et img1 sont bien en niveaux de gris
    if len(img0.shape) == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    gray0 = img0.astype(np.float32) / 255.0
    gray1 = img1.astype(np.float32) / 255.0

    Ix = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
    It = gray1 - gray0

    u01, v01 = horn_schunck_grad(Ix, Iy, It)

    Ix = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    It = gray0 - gray1

    u10, v10 = horn_schunck_grad(Ix, Iy, It)

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

    return img_mid


# def generate_horn_interpolation_tensor(img0, img1, device="cpu"):
#     """
#     Génère une interpolation entre img0 et img1 en utilisant ton algorithme personnalisé.
#     img0, img1 : Tenseurs PyTorch de forme (B, C, H, W)
#     """
#     batch_size = img0.shape[0]
#     mid_batch = []

#     for i in range(batch_size):
#         # Convertir les tenseurs en tableaux NumPy
#         img0_np = img0[i].squeeze(0).cpu().numpy() * 255.0
#         img1_np = img1[i].squeeze(0).cpu().numpy() * 255.0

#         # Générer l'interpolation
#         img_mid = generate_horn_interpolation(img0_np.astype(np.uint8), img1_np.astype(np.uint8))

#         # Convertir en tenseur PyTorch
#         img_mid_tensor = torch.from_numpy(img_mid).unsqueeze(0).float() / 255.0
#         mid_batch.append(img_mid_tensor)

#     # Empiler les images interpolées en un seul tenseur
#     mid_batch = torch.stack(mid_batch, dim=0)
#     return mid_batch.to(device)

def generate_horn_interpolation_tensor(img0, img1, device="cpu"):
    batch_size = img0.shape[0]
    mid_batch = []

    for i in range(batch_size):
        img0_np = img0[i].squeeze(0).cpu().numpy() * 255.0
        img1_np = img1[i].squeeze(0).cpu().numpy() * 255.0

        if img0_np is None or img1_np is None:
            raise ValueError(f"Erreur de chargement des images aux indices {i}")

        img_mid = generate_horn_interpolation(img0_np.astype(np.uint8), img1_np.astype(np.uint8))

        img_mid_tensor = torch.from_numpy(img_mid).unsqueeze(0).float() / 255.0
        mid_batch.append(img_mid_tensor)

    mid_batch = torch.stack(mid_batch, dim=0)
    assert mid_batch.min() >= 0 and mid_batch.max() <= 1, "Les valeurs des pixels ne sont pas dans la plage [0, 1]"
    return mid_batch.to(device)


def generate_farneback_interpolation(img0, img1):
    """
    img0, img1 : numpy arrays uint8 (H, W, 3) BGR
    Retourne une image interpolée RGB (H, W, 3)
    """

    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = gray0.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + flow[..., 0] * 0.5).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * 0.5).astype(np.float32)

    interp = cv2.remap(img0, map_x, map_y, cv2.INTER_LINEAR)
    interp = cv2.cvtColor(interp, cv2.COLOR_BGR2RGB)

    return interp

