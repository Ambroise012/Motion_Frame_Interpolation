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

from utils.dataset import FrameFolderTripletDataset
from utils.utils import generate_horn_interpolation_tensor
from utils.loss import CombinedLoss
from utils.model import UNet

def train(model, dataloader, epochs=20, device="cpu", save_dir="outputs_DL/debug"):
    model = model.to(device)
    criterion = CombinedLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    RES_SCALE = 15.0  # a tester # pour sortir de Horn -> augmenter si besoin : 15

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for img0, img_mid_gt, img1 in dataloader:
            img0 = img0.to(device)
            img_mid_gt = img_mid_gt.to(device)
            img1 = img1.to(device)

            # Horn initial
            with torch.no_grad():
                mid_initial = generate_horn_interpolation_tensor(img0, img1, device)

            # pred residuals
            x = torch.cat([mid_initial, img0, img1], dim=1)
            residual_pred = model(x)

            target_residual = RES_SCALE * (img_mid_gt - mid_initial)

            # reconstruction
            mid_corrected = mid_initial + residual_pred / RES_SCALE

            # Loss 
            loss = (
                criterion(residual_pred, target_residual)
                + 0.5 * nn.functional.l1_loss(mid_corrected, img_mid_gt)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        visualize_interpolation(
            img0[0],
            img_mid_gt[0],
            img1[0],
            mid_initial[0],
            torch.clamp(mid_corrected[0], 0, 1),
            epoch=epoch,
            save_dir=save_dir
        )

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    return model


def save_image(tensor, path):
    if isinstance(tensor, torch.Tensor):
        img = tensor.clamp(0, 1).cpu().numpy()
    else:
        img = tensor
    img = (img * 255).astype(np.uint8)
    if img.shape[0] == 1:
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

    # convert tensor and gray level
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
        ax.imshow(img, cmap='gray')  # limited to gray
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
    OUTPUT_DIR = "outputs_DL_improv_unet"
    MODEL_DIR = "models"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = FrameFolderTripletDataset(FRAME_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Créer le modèle
    model = UNet(n_channels=3, n_classes=1)
    
    # train
    model = train(model, dataloader, device=device, epochs=50, save_dir=f"{OUTPUT_DIR}/debug")

    # save model
    torch.save(model.state_dict(), f"{MODEL_DIR}/unet_50_epoch_improv-unet.pth")

    # inferrence (test)
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

        # residual = model(
        #     mid_initial.unsqueeze(0),
        #     img0.unsqueeze(0),
        #     img1.unsqueeze(0),
        # )[0]
        x = torch.cat([
            mid_initial.unsqueeze(0),
            img0.unsqueeze(0),
            img1.unsqueeze(0)
        ], dim=1)

        residual = model(x)[0]

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

