import os
import cv2
import numpy as np
from PIL import Image
from methods.RMem.aot_plus.tools.demo import color_palette, _palette, overlay

def merge_video_and_mask(input_video_path, mask_root, output_root):
    reader = cv2.VideoCapture(input_video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    seq_name = input_video_path.split("/")[-1].split(".")[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = os.path.join(
        output_root, '{}_segmented.mp4'.format(seq_name))
    
    frame_idx = 0
    while True:
        ret, frame = reader.read()
        pred_label = Image.open()
        overlayed_image = overlay(
                    np.array(frame, dtype=np.uint8),
                    np.array(pred_label, dtype=np.uint8), color_palette)
                videoWriter.write(overlayed_image[..., [2, 1, 0]])  