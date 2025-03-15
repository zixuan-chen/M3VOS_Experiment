import os
import numpy as np
from PIL import Image

import sys
from time import time
import argparse
from tqdm import tqdm 
import asyncio
import time
import glob, os

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def main(res_path, save_path):

    video_name_list = os.listdir(res_path)
    video_name_list = [video_name for video_name in video_name_list if os.path.isdir(os.path.join(res_path, video_name))]
    video_name_list.sort()

    for video_name in video_name_list:
        merge_mask_per_video(res_path, save_path, video_name)
    
    return 0

@background
def merge_mask_per_video(res_path, save_path, video_name, threshold=0.3):

    obj_id_list = os.listdir(os.path.join(res_path, video_name))
    obj_id_list.sort()
    frame_list = os.listdir(os.path.join(res_path, video_name, obj_id_list[0], 'frames'))
    frame_list.sort()
    for frame in tqdm(frame_list):
        mask_pred_list = []
        for idx, obj_id in enumerate(obj_id_list):
            mask_path = os.path.join(res_path, video_name, obj_id, 'frames', frame)
            img_array = np.load(mask_path)

            # confidence score
            # confidence_per_obj = (img_array[img_array > threshold]).mean()
            # img_array[img_array > threshold] = confidence_per_obj
            # img_array[img_array > threshold] = img_array[img_array > threshold] - mean_per_obj + (1 - threshold) / 2.0


            if len(mask_pred_list) == 0:
                bg_array = np.ones_like(img_array) * threshold
                mask_pred_list.append(bg_array)

            mask_pred_list.append(img_array)

        # for i in range(1, len(mask_pred_list)-1):
        #     for j in range(i+1, len(mask_pred_list)):
        #         if len(mask_pred_list) == j:
        #             continue
        #         A = mask_pred_list[i]
        #         B = mask_pred_list[j]
        #         overlap = np.logical_and(A > threshold, B > threshold)
        #         # get avergae of score witin overlap
        #         if overlap.sum() > 0:
        #             if (A[overlap] > 0.8).sum() > (B[overlap] > 0.8).sum():
        #                 B[overlap] = 0
        #             else:   
        #                 A[overlap] = 0
        #         else:
        #             continue

        mask_pred_list_stacked = np.stack(mask_pred_list, axis=0)
        mask_pred = mask_pred_list_stacked.argmax(0)

        mask_img = Image.fromarray(mask_pred.astype(np.uint8)).convert('L')
        mask_img.putpalette(color_map().flatten().tolist())
        save_mask_path = os.path.join(save_path, video_name, frame[:-3]+'png')
        os.makedirs(os.path.join(save_path, video_name), exist_ok=True)
        mask_img.save(save_mask_path)
    print("Complete: ", video_name)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--results_path', type=str, help='Path to the folder containing the sequences folders',
                        required=True)
    parser.add_argument('-o','--new_folder_name', type=str, help='Name of the new folder containing the merged masks',
                        required=False, default='merged')
    args, _ = parser.parse_known_args()

    # Create new folder
    results_path = args.results_path
    new_folder_path = os.path.join(os.path.abspath(os.path.join(results_path, os.pardir)), args.new_folder_name)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    else:
        print('Folder already exists, exiting...')
        print('Overwrite the previous results...')

    # Run main function
    main(results_path, new_folder_path)