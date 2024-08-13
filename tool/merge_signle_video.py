#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from tqdm  import tqdm


   
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
    # 获取图片文件列表，并按文件名排序

    original_images = [os.path.join(original_image_folder, img) for img in os.listdir(original_image_folder) if img.endswith('.jpg')]
    images = [False] * len(original_images) * 50
    # print(len(images))

    for  img in original_images :
        # print(os.path.splitext(img)[0][-5:])
        img_idx = int(''.join(filter(str.isdigit, os.path.splitext(img)[0]))[-5:])
        # print( ''.join(filter(str.isdigit, os.path.splitext(img)[0]))[-5:])
        images[img_idx] = True
        

    # img = (os.listdir(mask_image_folder) [0])
    # print(int(os.path.splitext(img)[0]) % 6 == 0)
    # 这里按照编号
    mask_images = [os.path.join(mask_image_folder, img) for img in os.listdir(mask_image_folder) if img.endswith('.png') and  int(''.join(filter(str.isdigit, os.path.splitext(img)[0]))) < len(original_images) * 6 + 6 and   images[ int(''.join(filter(str.isdigit, os.path.splitext(img)[0])))]  ]

    
    original_images = [ os.path.join(original_image_folder, os.path.splitext(os.path.basename(img))[0] +'.jpg'  ) for img in   mask_images  ]
    # print(original_images)

    original_images.sort()
    mask_images.sort()

    print( original_image_folder , " : ",len(original_images))
    print(mask_image_folder , " : " ,len(mask_images))
    # 检查原图和mask图片数量是否一致
    if len(original_images) != len(mask_images):
        raise ValueError("原图和mask图片数量不一致")

    # 设置视频帧率（每秒显示的图片数量）
    fps = 24  # 你可以根据需要调整这个值

    # 设置透明度
    alpha = 0.5  # 透明度值（0到1之间）

    # 创建一个列表来存储合成后的图片
    composite_images = []
    bar = tqdm( zip(original_images, mask_images))

    # 遍历每对原图和mask
    for orig_path, mask_path in bar:
        # 读取原图和mask图像
        orig_image = cv2.imread(orig_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # 确保mask图像是四通道（包括透明度通道）
        if mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)

        #
        # 将mask图像的alpha通道应用透明度
        mask_image[:, :, 3] = (mask_image[:, :, 3] * alpha).astype(mask_image.dtype)

        # 将mask图像叠加到原图上
        composite_image = orig_image.copy()
        for c in range(0, 3):
            composite_image[:, :, c] = mask_image[:, :, 3] / 255.0 * mask_image[:, :, c] + \
                                    (1.0 - mask_image[:, :, 3] / 255.0) * composite_image[:, :, c]

        # 添加到合成图片列表
        composite_images.append(composite_image)


    # 使用moviepy将合成后的图片生成视频
    clip = ImageSequenceClip([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in composite_images], fps=fps)
    clip.write_videofile(output_video, codec='libx264')

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
    # 获取mask文件夹和images文件夹的子文件夹列表
    mask_folders = [folder for folder in os.listdir(masks_folder) if os.path.isdir(os.path.join(masks_folder, folder))]
    image_folders = mask_folders
    print("ALL video:" , len(mask_folders))
    # print(mask_folders)


    # 检查mask文件夹和images文件夹的数量是否一致
    if len(mask_folders) != len(image_folders):
        raise ValueError("mask文件夹和images文件夹数量不一致")
    # cnt = 0

    # 遍历每对mask文件夹和images文件夹
    for mask_folder, image_folder in tqdm(zip(mask_folders, image_folders), desc="Merge mask and image"):
        mask_folder = os.path.join( masks_folder, mask_folder)
        image_folder = os.path.join(images_folder, image_folder)
        # 获取文件夹名称
        folder_name = os.path.basename(mask_folder)
        # 设置输出视频文件名
        output_video = os.path.join(exp_folder, "{}_merge.mp4".format(folder_name))

        if os.path.exists(output_video):
            print(output_video , "exists")
            continue
        # cnt += 1
        # print(cnt)
        # 调用merge_mask_images方法，将对应文件夹里的mask和images合并成视频
        merge_mask_images(original_image_folder=image_folder, mask_image_folder=mask_folder, output_video=output_video)

if __name__ == "__main__":
    # 设置文件夹路径
    # original_image_folder =  '/home/lijiaxin/Deform_VOS/DeformVOS/data/VOST/JPEGImages/3511_unscrew_jar'  # 替换为你的图片文件夹路径
    # mask_image_folder = '/home/lijiaxin/Deform_VOS/DeformVOS/exp/Cutie_base1080p/Annotations/3511_unscrew_jar'
    # output_video = '/home/lijiaxin/Deform_VOS/DeformVOS/tmp/test_merge.mp4'  # 设置输出视频文件名

    # merge_mask_images(original_image_folder= original_image_folder, mask_image_folder=mask_image_folder , output_video=output_video)    images_folder =  '/home/lijiaxin/Deform_VOS/DeformVOS/data/VOST/JPEGImages'  # 替换为你的图片文件夹路径
    masks_folder = '/home/lijiaxin/Deform_VOS/DeformVOS/exp/Cutie_base1080p/Annotations'
    output_exp = '/home/lijiaxin/Deform_VOS/DeformVOS/exp/Cutie_base1080p/merge_video'  # 设置输出视频文件名

    import argparse

    parser = argparse.ArgumentParser(description='Merge mask images with original images and create videos.')
    parser.add_argument('--images_folder', type=str, default= "/home/lijiaxin/Deform_VOS/DeformVOS/data/VOST/JPEGImages",help='Path to the folder containing original image folders')
    parser.add_argument('--masks_folder', type=str,default="/home/lijiaxin/Deform_VOS/DeformVOS/exp/Cutie_base1080p/Annotations" , help='Path to the folder containing mask image folders')
    parser.add_argument('--output_exp', type=str, default="/home/lijiaxin/Deform_VOS/DeformVOS/exp/Cutie_base1080p_merge_video", help='Path to the folder where the output videos will be saved')

    args = parser.parse_args()
    images_folder = args.images_folder
    masks_folder = args.masks_folder
    output_exp = args.output_exp
    images_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary/ROVES_week_1/JPEGImages/clipoff_wood_2" 


    # masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_deaot_deaot/mv_lime_2" 
    # masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/debug/break_egg_1" 
    # masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_1/cut_cucumber_1"
    masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/testAOT_bug_again/clipoff_wood_2"
    # masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_1_ckpt_unknown/Annotations/cut_celery_2"

    
    # masks_folder = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/datasets/ROVES/Annotations/cut_green_pepper_4"
    output_video = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/clipoff_wood_2_debug.mp4"

    merge_mask_images(images_folder, masks_folder, output_video)
    print(output_video)