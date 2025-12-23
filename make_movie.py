import cv2
import os
import argparse

def img_to_video(image_folder, output_video, fps):
    images = sorted(
        [img for img in os.listdir(image_folder) 
        if img.endswith(".png")])

    if not images:
        raise ValueError(f"Aucune image .png trouvée dans {image_folder}")

    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape

    video = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for img in images:
        img_path = os.path.join(image_folder, img)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Movie from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="folder path containing images (png)"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="output folder path (.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frame per sec (default: 24)"
    )

    args = parser.parse_args()

    img_to_video(
        args.image_folder,
        args.output_video,
        args.fps
    )
    