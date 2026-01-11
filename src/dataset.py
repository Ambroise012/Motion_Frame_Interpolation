from torch.utils.data import Dataset
import torch
import cv2
from glob import glob
import os

# class UCFDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])


# class UCFTripletDataset(Dataset):
#     def __init__(self, video_paths, frame_interval=1, resize=(320, 240)):
#         self.samples = []
#         self.resize = resize

#         for video_path in video_paths:
#             cap = cv2.VideoCapture(video_path)
#             n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#             for t in range(0, n - 2 * frame_interval, frame_interval):
#                 self.samples.append((video_path, t, frame_interval))

#             cap.release()

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         video_path, t, fi = self.samples[idx]
#         cap = cv2.VideoCapture(video_path)

#         frames = []
#         for i in range(3):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, t + i * fi)
#             ret, frame = cap.read()
#             if not ret:
#                 cap.release()
#                 raise RuntimeError("Frame read error")

#             frame = cv2.resize(frame, self.resize)
#             frames.append(frame)

#         cap.release()

#         img0, img_mid, img1 = frames

#         mid_init = generate_farneback_interpolation(img0, img1)

#         # to tensor (C,H,W)
#         def to_tensor(img):
#             return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

#         return (
#             to_tensor(mid_init),
#             to_tensor(cv2.cvtColor(img_mid, cv2.COLOR_BGR2RGB))
#         )

# class FrameFolderTripletDataset(Dataset):
#     def __init__(self, frame_dir, resize=(320, 240)):
#         self.frame_paths = sorted(
#             glob(os.path.join(frame_dir, "*.png")) +
#             glob(os.path.join(frame_dir, "*.jpg"))
#         )

#         self.resize = resize

#     def __len__(self):
#         return len(self.frame_paths) - 2

#     def __getitem__(self, idx):
#         # Charger les images
#         img0 = cv2.imread(self.frame_paths[idx])
#         img_mid = cv2.imread(self.frame_paths[idx + 1])
#         img1 = cv2.imread(self.frame_paths[idx + 2])

#         # Redimensionner les images
#         img0 = cv2.resize(img0, self.resize)
#         img_mid = cv2.resize(img_mid, self.resize)
#         img1 = cv2.resize(img1, self.resize)

#         def to_tensor(img):
#             # Convertir de BGR à RGB
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             # Convertir en tenseur PyTorch et normaliser entre 0 et 1
#             img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
#             # Vérifier que les valeurs sont dans la plage [0, 1]
#             assert img_tensor.min() >= 0 and img_tensor.max() <= 1, "Les valeurs des pixels ne sont pas dans la plage [0, 1]"
#             return img_tensor

#         return (
#             to_tensor(img0),
#             to_tensor(img_mid),
#             to_tensor(img1)
#         )

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

