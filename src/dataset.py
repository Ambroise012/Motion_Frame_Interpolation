from torch.utils.data import Dataset
import torch
import cv2
from glob import glob
import os

class FrameFolderTripletDataset(Dataset):
    def __init__(self, frame_dir, resize=(320, 240)):
        self.frame_paths = sorted(
            glob(os.path.join(frame_dir, "*.png")) +
            glob(os.path.join(frame_dir, "*.jpg"))
        )

        if len(self.frame_paths) < 3:
            raise ValueError("Pas assez d'images dans le dossier. Il faut au moins 3 images.")

        self.resize = resize

    def __len__(self):
        return len(self.frame_paths) - 2

    def __getitem__(self, idx):
        img0 = cv2.imread(self.frame_paths[idx], cv2.IMREAD_GRAYSCALE)
        img_mid = cv2.imread(self.frame_paths[idx + 1], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(self.frame_paths[idx + 2], cv2.IMREAD_GRAYSCALE)

        if img0 is None or img_mid is None or img1 is None:
            raise ValueError(f"Erreur de chargement des images aux indices {idx}, {idx+1}, {idx+2}")

        img0 = cv2.resize(img0, self.resize)
        img_mid = cv2.resize(img_mid, self.resize)
        img1 = cv2.resize(img1, self.resize)

        def to_tensor(img):
            img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            assert img_tensor.shape[0] == 1, f"Le tenseur doit avoir 1 canal, mais a {img_tensor.shape[0]} canaux"
            return img_tensor

        return (
            to_tensor(img0),
            to_tensor(img_mid),
            to_tensor(img1)
        )

