import cv2
import os

# Dossier contenant les images
image_folder = "output_farneback"
output_video = "anim_farneback.mp4"

# Images triées
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Lire la première image pour la taille
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Création de la vidéo
video = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    24,  # fps
    (width, height)
)

# Ajouter chaque image
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
