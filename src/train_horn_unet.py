import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

from src.dataset import FrameFolderTripletDataset
from utils.utils import generate_horn_interpolation_tensor

def gradient_loss(pred, gt):
    dx = torch.abs(pred[:, :, :, 1:] - gt[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - gt[:, :, :-1, :])
    return dx.mean() + dy.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

        lap = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("laplacian", lap)

    def laplacian_loss(self, x, y):
        x_lap = nn.functional.conv2d(x, self.laplacian, padding=1)
        y_lap = nn.functional.conv2d(y, self.laplacian, padding=1)
        return self.l1(x_lap, y_lap)


    def forward(self, pred, gt):
        l1 = self.l1(pred, gt)
        lap = self.laplacian_loss(pred, gt)
        grad = gradient_loss(pred, gt)

        return (
            1.0 * l1 +
            0.5 * lap +
            0.3 * grad
        )

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return self.conv(x) + x

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.InstanceNorm2d(out_c, affine=True),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        self.dec4 = conv_block(512, 256)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1)
        )
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, mid, img0, img1):
        x = torch.cat([mid, img0, img1], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


# def train(model, dataloader, epochs=20, device="cpu"):
#     model = model.to(device)
#     criterion = CombinedLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for img0, img_mid_gt, img1 in dataloader:
#             img0, img_mid_gt, img1 = img0.to(device), img_mid_gt.to(device), img1.to(device)

#             # Générer une interpolation initiale
#             mid_initial = generate_horn_interpolation_tensor(img0, img1, device)

#             # Prédiction
#             residual_pred = model(mid_initial, img0, img1)
#             target_residual = img_mid_gt - mid_initial

#             # Calcul de la perte
#             loss = criterion(residual_pred, target_residual)

#             mid_corrected = torch.clamp(mid_initial + residual_pred, 0, 1)

#             # Mise à jour
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         if epoch % 2 == 0:
#             visualize_interpolation(
#                 img0[0],
#                 img_mid_gt[0],
#                 img1[0],
#                 mid_initial,
#                 mid_corrected,
#                 epoch=epoch,
#             )

#         scheduler.step(running_loss / len(dataloader))
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.6f}")

#     return model

def train(model, dataloader, epochs=20, device="cpu"):
    model = model.to(device)
    criterion = CombinedLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    RES_SCALE = 15.0  # pour sortir de Horn -> augmenter si besoin : 15

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for img0, img_mid_gt, img1 in dataloader:
            img0 = img0.to(device)
            img_mid_gt = img_mid_gt.to(device)
            img1 = img1.to(device)

            # Horn initial (hors graphe)
            with torch.no_grad():
                mid_initial = generate_horn_interpolation_tensor(img0, img1, device)

            # Résidu prédit
            residual_pred = model(mid_initial, img0, img1)

            # Résidu cible amplifié
            target_residual = RES_SCALE * (img_mid_gt - mid_initial)

            # Reconstruction finale
            mid_corrected = mid_initial + residual_pred / RES_SCALE

            # Loss combinée
            loss = (
                criterion(residual_pred, target_residual)
                + 0.5 * nn.functional.l1_loss(mid_corrected, img_mid_gt)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 2 == 0:
            visualize_interpolation(
                img0[0],
                img_mid_gt[0],
                img1[0],
                mid_initial[0],
                torch.clamp(mid_corrected[0], 0, 1),
                epoch=epoch,
            )

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    return model

def save_image(tensor, path):
    if isinstance(tensor, torch.Tensor):
        img = tensor.clamp(0, 1).cpu().numpy()  # Recadre les valeurs entre 0 et 1
    else:
        img = tensor
    img = (img * 255).astype(np.uint8)
    if img.shape[0] == 1:  # Si (C, H, W)
        img = img.squeeze(0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def visualize_interpolation(
    img0,
    img_mid_gt,
    img1,
    mid_initial,
    mid_corrected,
    epoch,
    save_dir="outputs_DL/debug"
):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Convertir les tenseurs en tableaux numpy et s'assurer qu'ils sont en niveaux de gris
    img0_np = img0.squeeze(0).cpu().numpy() if img0.dim() == 3 else img0.cpu().numpy()
    img_mid_gt_np = img_mid_gt.squeeze(0).cpu().numpy() if img_mid_gt.dim() == 3 else img_mid_gt.cpu().numpy()
    img1_np = img1.squeeze(0).cpu().numpy() if img1.dim() == 3 else img1.cpu().numpy()
    mid_initial_np = mid_initial[0].squeeze(0).cpu().numpy() if mid_initial[0].dim() == 3 else mid_initial[0].cpu().numpy()
    mid_corrected_np = (
        mid_corrected[0]
        .squeeze(0)
        .detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    images = [
        (img0_np, "img0"),
        (mid_initial_np, "mid_initial"),
        (mid_corrected_np, "mid_corrected"),
        (img_mid_gt_np, "img_mid_gt"),
        (img1_np, "img1"),
    ]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')  # Utiliser cmap='gray' pour les images en niveaux de gris
        ax.set_title(title)
        ax.axis("off")

    filename = f"epoch_{epoch:04d}.png"
    plt.savefig(os.path.join(save_dir, filename),
                dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cpu"

    FRAME_DIR = "mickey_original"
    OUTPUT_DIR = "outputs_DL"
    MODEL_DIR = "models"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = FrameFolderTripletDataset(FRAME_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Créer le modèle
    model = UNet()

    # Entraîner le modèle
    model = train(model, dataloader, device=device, epochs=50)

    # Sauvegarder APRÈS l'entraînement
    torch.save(model.state_dict(), f"{MODEL_DIR}/unet_50_epoch_res_15.pth")

    # Inférence
    model.eval()
    with torch.no_grad():
        img0, img_mid_gt, img1 = dataset[10]

        img0 = img0.to(device)
        img1 = img1.to(device)

        mid_initial = generate_horn_interpolation_tensor(
            img0.unsqueeze(0),
            img1.unsqueeze(0),
            device
        ).squeeze(0)

        residual = model(
            mid_initial.unsqueeze(0),
            img0.unsqueeze(0),
            img1.unsqueeze(0),
        )[0]

        mid_corrected = torch.clamp(
            mid_initial + residual / 10.0,
            0, 1
        )

        save_image(img0, f"{OUTPUT_DIR}/img0.png")
        save_image(mid_corrected, f"{OUTPUT_DIR}/pred.png")
        save_image(img_mid_gt, f"{OUTPUT_DIR}/gt.png")
        save_image(mid_initial, f"{OUTPUT_DIR}/mid_initial.png")


if __name__ == "__main__":
    main()

