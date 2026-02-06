import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_loss(pred, gt, alpha=1.0):
    """Calcule la perte de gradient pondérée."""
    # Gradients horizontaux et verticaux
    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    dx_gt = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    dy_gt = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

    # Moyenne des différences de gradients
    dx_loss = F.l1_loss(dx_pred, dx_gt)
    dy_loss = F.l1_loss(dy_pred, dy_gt)

    return alpha * (dx_loss + dy_loss)

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")  # important !

    def forward(self, pred, gt):
        dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # Moyenne simple des gradients
        edge_weight = (dx + dy) / 2.0          # (B, C, H, W)
        edge_weight = edge_weight.mean(1, True)  # (B, 1, H, W)

        edge_weight = F.interpolate(
            edge_weight,
            size=pred.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.l1(pred, gt)  # (B, C, H, W)
        return torch.mean(loss * edge_weight)
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")  # important !

    def forward(self, pred, gt):
        dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # Moyenne simple des gradients
        edge_weight = (dx + dy) / 2.0          # (B, C, H, W)
        edge_weight = edge_weight.mean(1, True)  # (B, 1, H, W)

        edge_weight = F.interpolate(
            edge_weight,
            size=pred.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.l1(pred, gt)  # (B, C, H, W)
        return torch.mean(loss * edge_weight)
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")  # important !

    def forward(self, pred, gt):
        dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # Moyenne simple des gradients
        edge_weight = (dx + dy) / 2.0          # (B, C, H, W)
        edge_weight = edge_weight.mean(1, True)  # (B, 1, H, W)

        edge_weight = F.interpolate(
            edge_weight,
            size=pred.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.l1(pred, gt)  # (B, C, H, W)
        return torch.mean(loss * edge_weight)

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")  # important !

    def forward(self, pred, gt):
        dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        # Moyenne simple des gradients
        edge_weight = (dx + dy) / 2.0          # (B, C, H, W)
        edge_weight = edge_weight.mean(1, True)  # (B, 1, H, W)

        edge_weight = F.interpolate(
            edge_weight,
            size=pred.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.l1(pred, gt)  # (B, C, H, W)
        return torch.mean(loss * edge_weight)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

        # Noyau laplacien
        lap = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian", lap)

        # Perte pondérée par les contours
        self.edge_loss = EdgeAwareLoss()

    def laplacian_loss(self, x, y):
        B, C, H, W = x.shape
        lap = self.laplacian.repeat(C, 1, 1, 1)

        x_lap = F.conv2d(x, lap, padding=1, groups=C)
        y_lap = F.conv2d(y, lap, padding=1, groups=C)

        return self.l1(x_lap, y_lap)

    def forward(self, pred, gt):
        # Pertes de base
        l1 = self.l1(pred, gt)
        lap = self.laplacian_loss(pred, gt)
        grad = gradient_loss(pred, gt)

        # Perte pondérée par les contours
        edge = self.edge_loss(pred, gt)

        # Combinaison pondérée
        print(
            f"L1: {l1.item():.3f} | "
            f"Lap: {lap.item():.3f} | "
            f"Grad: {grad.item():.3f} | "
            f"Edge: {edge.item():.3f}"
        )

        return (
            1.0 * l1 +
            0.5 * lap +
            0.3 * grad +
            0.2 * edge  # Poids ajustable
        )
