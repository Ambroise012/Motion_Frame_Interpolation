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

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Charger VGG16 pour la perte perceptuelle
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, y_pred, y_true):
        y_pred_3 = y_pred.repeat(1, 3, 1, 1)
        y_true_3 = y_true.repeat(1, 3, 1, 1)
        return self.mse_loss(self.vgg(y_pred_3), self.vgg(y_true_3))

    def gradient_loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        return torch.mean(dx) + torch.mean(dy)

    def ssim_loss(self, y_pred, y_true):
        # Implémentation simplifiée de la perte SSIM
        mu_x = torch.mean(y_pred, dim=[1, 2, 3], keepdim=True)
        mu_y = torch.mean(y_true, dim=[1, 2, 3], keepdim=True)

        sigma_x = torch.std(y_pred, dim=[1, 2, 3], keepdim=True)
        sigma_y = torch.std(y_true, dim=[1, 2, 3], keepdim=True)
        sigma_xy = torch.mean((y_pred - mu_x) * (y_true - mu_y), dim=[1, 2, 3], keepdim=True)

        ssim = (2 * mu_x * mu_y + 1e-6) * (2 * sigma_xy + 1e-6)
        ssim /= (mu_x**2 + mu_y**2 + 1e-6) * (sigma_x**2 + sigma_y**2 + 1e-6)
        return torch.mean(1 - ssim)

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        l1 = self.l1_loss(y_pred, y_true)
        perceptual = self.perceptual_loss(y_pred, y_true)
        gradient = self.gradient_loss(y_pred)
        ssim = self.ssim_loss(y_pred, y_true)

        # Poids des différentes pertes
        total_loss = (
            1.0 * mse + 
            0.5 * l1 +
            0.2 * perceptual +
            0.2 * gradient +
            0.2 * ssim
        )
        return total_loss

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

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = self._make_layer(1, 32)
        self.encoder2 = self._make_layer(32, 64)
        self.encoder3 = self._make_layer(64, 128)
        self.encoder4 = self._make_layer(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self._make_layer(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder4 = self._make_layer(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder3 = self._make_layer(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder2 = self._make_layer(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.decoder1 = self._make_layer(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            ResidualBlock(out_c),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        output = self.final(d1)
        return output


def train(model, dataloader, epochs=20, device="cuda"):
    model = model.to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img0, img_mid_gt, img1 in dataloader:
            img0, img_mid_gt, img1 = img0.to(device), img_mid_gt.to(device), img1.to(device)

            # Générer une interpolation initiale
            mid_initial = generate_horn_interpolation_tensor(img0, img1, device)

            # Prédiction
            mid_corrected = model(mid_initial)

            # Calcul de la perte
            loss = criterion(mid_corrected, img_mid_gt)

            # Mise à jour
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 4 == 0:
            visualize_interpolation(
                img0[0],
                img_mid_gt[0],
                img1[0],
                mid_initial,
                mid_corrected,
                epoch=epoch,
            )

        scheduler.step(running_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.6f}")

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
    model = train(model, dataloader, device=device, epochs=40)

    # Sauvegarder APRÈS l'entraînement
    torch.save(model.state_dict(), f"{MODEL_DIR}/sunet_40_epoch.pth")

    # Inférence
    model.eval()
    with torch.no_grad():
        img0, img_mid_gt, img1 = dataset[10]

        mid_initial = generate_horn_interpolation_tensor(
            img0.unsqueeze(0),
            img1.unsqueeze(0),
            device
        ).squeeze(0)

        pred = model(mid_initial.unsqueeze(0).to(device))[0]

        save_image(img0, f"{OUTPUT_DIR}/img0.png")
        save_image(pred.cpu(), f"{OUTPUT_DIR}/pred.png")
        save_image(img_mid_gt, f"{OUTPUT_DIR}/gt.png")
        save_image(mid_initial, f"{OUTPUT_DIR}/mid_initial.png")

if __name__ == "__main__":
    main()

