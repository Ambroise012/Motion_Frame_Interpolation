import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from src.horn_schunck import horn_schunck_grad

IMG0_PATH = "mickey_original/frame_00104.png"
IMG1_PATH = "mickey_original/frame_00105.png"
OUTPUT_PATH = "flow_arrows_horn2.png"

SIZE = (960, 720)
STEP = 20
ALPHA = 1.0
MAX_ITER = 1000
EPS = 1e-4

img0 = cv2.imread(IMG0_PATH)
img1 = cv2.imread(IMG1_PATH)

img0 = cv2.resize(img0, SIZE)
img1 = cv2.resize(img1, SIZE)

gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

Ix = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
It = gray1 - gray0

u, v = horn_schunck_grad(Ix, Iy, It, alpha=ALPHA, max_iter=MAX_ITER, eps=EPS)

h, w = u.shape
y, x = np.mgrid[0:h:STEP, 0:w:STEP]

plt.figure(figsize=(12, 8))
plt.imshow(gray0, cmap="gray")
plt.quiver(
    x, y,
    u[::STEP, ::STEP],
    v[::STEP, ::STEP],
    color='red'
)

plt.title("Flot optique - Horn & Schunck")
plt.axis("off")

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

