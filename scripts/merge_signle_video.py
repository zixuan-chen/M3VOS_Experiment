#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm


def merge_mask_images(original_image_folder: str, mask_image_folder: str, output_video: str) -> None:
    """
    Merges original images with mask images and creates a video.

    Args:
        original_image_folder (str): Path to the folder containing original images.
        mask_image_folder (str): Path to the folder containing mask images.
        output_video (str): Path to the output video file.

    Raises:
        ValueError: If the number of original images and mask images is not the same.

    Returns:
        None
    """
    # Get the list of image files and sort them by filename
    original_images = [os.path.join(original_image_folder, img) for img in os.listdir(original_image_folder) if img.endswith('.jpg')]
    images = [False] * len(original_images) * 50

    for img in original_images:
        img_idx = int(''.join(filter(str.isdigit, os.path.splitext(img)[0]))[-5:])
        images[img_idx] = True

    # Filter mask images based on their index and ensure they correspond to the original images
    mask_images = [os.path.join(mask_image_folder, img) for img in os.listdir(mask_image_folder) if img.endswith('.png') and int(''.join(filter(str.isdigit, os.path.splitext(img)[0]))) < len(original_images) * 6 + 6 and images[int(''.join(filter(str.isdigit, os.path.splitext(img)[0])))]]

    # Match original images with mask images
    original_images = [os.path.join(original_image_folder, os.path.splitext(os.path.basename(img))[0] + '.jpg') for img in mask_images]

    original_images.sort()
    mask_images.sort()

    print(original_image_folder, " : ", len(original_images))
    print(mask_image_folder, " : ", len(mask_images))

    # Check if the number of original images and mask images is the same
    if len(original_images) != len(mask_images):
        raise ValueError("The number of original images and mask images does not match")

    # Set the video frame rate (number of images displayed per second)
    fps = 24  # You can adjust this value as needed

    # Set transparency
    alpha = 0.5  # Transparency value (between 0 and 1)

    # Create a list to store the composite images
    composite_images = []
    bar = tqdm(zip(original_images, mask_images))

    # Iterate through each pair of original and mask images
    for orig_path, mask_path in bar:
        # Read the original and mask images
        orig_image = cv2.imread(orig_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Ensure the mask image has four channels (including the alpha channel)
        if mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)

        # Apply transparency to the alpha channel of the mask image
        mask_image[:, :, 3] = (mask_image[:, :, 3] * alpha).astype(mask_image.dtype)

        # Overlay the mask image onto the original image
        composite_image = orig_image.copy()
        for c in range(0, 3):
            composite_image[:, :, c] = mask_image[:, :, 3] / 255.0 * mask_image[:, :, c] + \
                                    (1.0 - mask_image[:, :, 3] / 255.0) * composite_image[:, :, c]

        # Add the composite image to the list
        composite_images.append(composite_image)

    # Use moviepy to generate a video from the composite images
    clip = ImageSequenceClip([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in composite_images], fps=fps)
    print("fps:", fps)
    clip.write_videofile(output_video, codec='libx264', fps=fps)


def merge_all_mask_video(masks_folder, images_folders, exp_folder):
    """
    Merges mask images with original images from multiple folders and creates videos.

    Args:
        masks_folder (str): Path to the folder containing mask image folders.
        images_folders (str): Path to the folder containing original image folders.
        exp_folder (str): Path to the folder where the output videos will be saved.

    Returns:
        None
    """
    # Get the list of subfolders in the masks and images folders
    mask_folders = [folder for folder in os.listdir(masks_folder) if os.path.isdir(os.path.join(masks_folder, folder))]
    image_folders = mask_folders
    print("ALL video:", len(mask_folders))

    # Check if the number of mask folders and image folders is the same
    if len(mask_folders) != len(image_folders):
        raise ValueError("The number of mask folders and image folders does not match")

    # Iterate through each pair of mask and image folders
    for mask_folder, image_folder in tqdm(zip(mask_folders, image_folders), desc="Merge mask and image"):
        mask_folder = os.path.join(masks_folder, mask_folder)
        image_folder = os.path.join(images_folder, image_folder)

        # Get the folder name
        folder_name = os.path.basename(mask_folder)
        # Set the output video file name
        output_video = os.path.join(exp_folder, "{}_merge.mp4".format(folder_name))

        if os.path.exists(output_video):
            print(output_video, "exists")
            continue

        # Call the merge_mask_images function to merge the mask and images into a video
        merge_mask_images(original_image_folder=image_folder, mask_image_folder=mask_folder, output_video=output_video)


if __name__ == "__main__":
    # Set folder paths
    masks_folder = '/path/to/Annotations'
    output_exp = '/path/to/merge_video'  # Set the output video file name

    import argparse

    parser = argparse.ArgumentParser(description='Merge mask images with original images and create videos.')
    parser.add_argument('--images_folder', type=str, default="/path/to/VOST/JPEGImages", help='Path to the folder containing original image folders')
    parser.add_argument('--masks_folder', type=str, default="/path/to/Annotations", help='Path to the folder containing mask image folders')
    parser.add_argument('--output_exp', type=str, default="/path/to_merge_video", help='Path to the folder where the output videos will be saved')

    args = parser.parse_args()
    images_folder = args.images_folder
    masks_folder = args.masks_folder
    output_exp = args.output_exp
    images_folder = "/path/to/ROVES/JPEGImages/0375_make_pottery_6"

    masks_folder = "/path/to/Cutie/cutie_output/roves_week10/roves-val/Annotations/0375_make_pottery_6"
    output_video = "/path/to/tmp/Cutie_375.mp4"

    merge_mask_images(images_folder, masks_folder, output_video)