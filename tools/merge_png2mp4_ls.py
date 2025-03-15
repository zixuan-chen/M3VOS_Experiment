#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from tqdm  import tqdm
# import torch
from scipy.ndimage import zoom

   
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

    if os.path.exists(output_video):
        print(output_video , " has existed!")
        return 

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

    # 检查原图和mask图片数量是否一致
    if len(original_images) != len(mask_images):
        raise ValueError("原图和mask图片数量不一致")

    # 设置视频帧率（每秒显示的图片数量）
    fps = 24  # 你可以根据需要调整这个值

    # 设置透明度
    alpha = 0.8 # 透明度值（0到1之间）

    # 创建一个列表来存储合成后的图片
    composite_images = []
    bar = tqdm( zip(original_images, mask_images))

    # 遍历每对原图和mask
    for orig_path, mask_path in bar:
        # 读取原图和mask图像
        orig_image = cv2.imread(orig_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # print("O shape:" , orig_image.shape)
        # print("M shape:" , mask_image.shape)
        # # assert False


        if mask_image.shape != orig_image.shape:
            mask_image = mask_image[None, :]  

            # 计算缩放因子
            scale_factors = (
                1,  # 不改变批次维度
                orig_image.shape[0] / mask_image.shape[1],  # 高度缩放因子
                orig_image.shape[1] / mask_image.shape[2],  # 宽度缩放因子
                1,  # 不改变通道数
            )
        
            
            # 使用 scipy.ndimage.zoom 进行插值调整尺寸
            mask_image = zoom(mask_image, scale_factors, order=1)  # 使用线性插值（order=1）
            # print("after shape:" , mask_image.shape)
            
            # 删除添加的批次维度
            mask_image = mask_image[0]



        # 确保mask图像是四通道（包括透明度通道）
        if mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)

        # print("one mask image:",mask_image.shape )

        #
        # 将mask图像的alpha通道应用透明度
        mask_image[:, :, 3] = (mask_image[:, :, 3] * alpha).astype(mask_image.dtype)

        # 将mask图像叠加到原图上
        composite_image = orig_image.copy()
        # print("mask image:", mask_image.shape)

        mask_position = (mask_image[:, :, 2] != 0) + (mask_image[:, :, 1] != 0) 
        # print("mask_position:" , mask_position.shape)
        # print("mask_position:" , mask_position)

        # for c in range(0, 3):
        #     composite_image[:, :, c] = mask_image[:, :, 3] / 255.0 * mask_image[:, :, c] * alpha + \
        #                             (1.0 - alpha * mask_image[:, :, 3] / 255.0) * composite_image[:, :, c]


        for c in range(0, 3):
            composite_image[mask_position, c] = mask_image[mask_position, 3] / 255.0 * mask_image[mask_position, c] * alpha + \
                                    (1.0 - alpha * mask_image[mask_position, 3] / 255.0) * composite_image[mask_position, c]

        

        

        # 添加到合成图片列表
        composite_images.append(composite_image)

        # cv2.imwrite("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/test_light.png", mask_position * 255, params=None)
        # cv2.imwrite("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/test_light.png", composite_image, params=None)

        # print("Save /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/test_light.png")

        # assert False


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
    # print("ALL video:" , len(mask_folders))
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

def merge_ls_SAM2(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary", SAM2_res_path ="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs"  , id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp"):
    """
    eg:
    Image_path: /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary/ROVES_week_12/JPEGImages/0484_cut_ice_cream_1
    Anno_path: /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs/sam2_in_roves_week_12/0484_cut_ice_cream_1
    根据输入的id_ls ,找到对应编号的视频图像文件夹和Anno文件夹，利用merge_mask_images函数把视频存在/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/目录下
    """
    
    os.makedirs(tmp_dir, exist_ok=True)

    for week_folder in os.listdir(ROVES_ROOT):
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        week_id = int(re.findall(r'\d+', week_folder)[0])
        
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]

            if week_id >= 8:
                if not vid.isdigit() :
                    continue



                if not vid in id_ls:
                    continue
    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue
            
            print("[BEGIN] Merge video:" , video_name)



            for mask_week in os.listdir(SAM2_res_path):
                mask_week_folder = os.path.join(SAM2_res_path, mask_week)

                if not os.path.isdir(mask_week_folder):
                    continue


                for mask_video_name in os.listdir(mask_week_folder):
        

                    if week_id > 8:
                    
                        if mask_video_name.split("_")[0] == vid:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)
                    else:

                        if mask_video_name  == video_name:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)

def merge_ls_Cutie(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary", Cutie_res_path ="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output"  , id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/Cutie_bad_case"):
    """
    eg:
    Image_path: /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary/ROVES_week_12/JPEGImages/0484_cut_ice_cream_1
    Anno_path: /home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week12/roves-val/Annotations/0484_cut_ice_cream_1
    根据输入的id_ls ,找到对应编号的视频图像文件夹和Anno文件夹，利用merge_mask_images函数把视频存在/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/目录下
    """
    
    os.makedirs(tmp_dir, exist_ok=True)

    for week_folder in os.listdir(ROVES_ROOT):

        week_id = int(re.findall(r'\d+', week_folder)[0])
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        print("image_folder:", image_folder)
        
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]
            

            if week_id >= 8:
                if not vid.isdigit() :
                    continue



                if not vid in id_ls:
                    continue
    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue
            print("[BEGIN] Merge video:" , video_name)


            for mask_week in os.listdir(Cutie_res_path):
                if "roves" not in mask_week:
                    continue
                mask_week_folder = os.path.join(Cutie_res_path, mask_week, "roves-val", "Annotations")

                for mask_video_name in os.listdir(mask_week_folder):
                    

                    if week_id > 8:
                    
                        if mask_video_name.split("_")[0] == vid:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)
                    else:

                        if mask_video_name  == video_name:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)

def merge_ls_GT(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary",  id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/GT"):
    """
    eg:
    Image_path: /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary/ROVES_week_12/JPEGImages/0484_cut_ice_cream_1
    Anno_path: /home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week12/roves-val/Annotations/0484_cut_ice_cream_1
    根据输入的id_ls ,找到对应编号的视频图像文件夹和Anno文件夹，利用merge_mask_images函数把视频存在/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/目录下
    """
    
    os.makedirs(tmp_dir, exist_ok=True)

    for week_folder in os.listdir(ROVES_ROOT):
        week_id = int(re.findall(r'\d+', week_folder)[0])
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]

            if week_id >= 8:
                if not vid.isdigit() :
                    continue

                if not vid in id_ls:
                    continue


    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue
     


            print("[BEGIN] Merge video:" , video_name)
            img = os.path.join(image_folder, video_name)
            mask_folder =  os.path.join(ROVES_ROOT, week_folder, "Annotations")
            mask = os.path.join(mask_folder, video_name)

            output =  os.path.join(tmp_dir, video_name + ".mp4")
            merge_mask_images(img, mask,output)
            print("[DONE] Merge video:" , output)
import re

def merge_ls_ours_Cutie_reverse(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary", ours_cutie_plus_res_path ="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output"  , id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/Cutie_bad_case"):
    """
    Anno_path:/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output/roves_week8_ckpt176300/roves-val/Annotations/0160_erupting_liquid_1
    
    """
    
    os.makedirs(tmp_dir, exist_ok=True)


    for week_folder in os.listdir(ROVES_ROOT):

        week_id = int(re.findall(r'\d+', week_folder)[0])
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        print("image:", image_folder)
        
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]
   


            if week_id >= 8:
                if not vid.isdigit() :
                    continue



                if not vid in id_ls:
                    continue
    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue

        

            for mask_week in os.listdir(ours_cutie_plus_res_path):

# roves_week0_mega_v4_72800
                pattern = r'^roves_week\d+_mega_v4_72800$'

                if not bool(re.match(pattern, mask_week)):
                    continue
                

                if "roves" not in mask_week:
                    continue
                # if "ckpt176300" not in mask_week:
                #     continue

                if "72800" not in mask_week:
                    continue


                mask_week_folder = os.path.join(ours_cutie_plus_res_path, mask_week, "roves-val", "Annotations")

                for mask_video_name in os.listdir(mask_week_folder):

                    if week_id > 8:
                    
                        if mask_video_name.split("_")[0] == vid:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)
                    else:

                        if mask_video_name  == video_name:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)


def merge_ls_Rmem(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary", rmem_res_path ="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves"  , id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/Cutie_bad_case"):
    """
    Anno_path:/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves/test_rmem_wo_pte_roves_in_aot_week_0/break_brocolli_3
    
    """
    
    os.makedirs(tmp_dir, exist_ok=True)


    for week_folder in os.listdir(ROVES_ROOT):

        week_id = int(re.findall(r'\d+', week_folder)[0])
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        print("image:", image_folder)
        
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]



            if week_id >= 8:
                if not vid.isdigit() :
                    continue



                if not vid in id_ls:
                    continue
    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue
            print("[BEGIN] Merge video:" , video_name)
            # print(os.listdir(rmem_res_path))

        

            for mask_week in os.listdir(rmem_res_path):

                # test_rmem_wo_pte_roves_in_aot_week_\d+$
                pattern = r'test_rmem_wo_pte_roves_in_aot_week_\d+$'

                if not bool(re.match(pattern, mask_week)):
                    continue
                

                if "roves" not in mask_week:
                    continue
                # if "ckpt176300" not in mask_week:
                #     continue

          

                mask_week_folder = os.path.join(rmem_res_path,  mask_week)

                for mask_video_name in os.listdir(mask_week_folder):
                    if week_id > 8:
                    
                        if mask_video_name.split("_")[0] == vid:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)
                    else:

                        if mask_video_name  == video_name:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)

def merge_ls_Deaot(ROVES_ROOT= "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary", deaot_res_path ="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves"  , id_ls = ["0484"],tmp_dir = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp/Cutie_bad_case"):
    """
    Anno_path: /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves/roves_val_low_res_infer_roves_in_deaot_swin_week_13_SwinB_DeAOTL_PRE_ckpt_unknown
    """

    os.makedirs(tmp_dir, exist_ok=True)


    for week_folder in os.listdir(ROVES_ROOT):

        week_id = int(re.findall(r'\d+', week_folder)[0])
        image_folder = os.path.join(ROVES_ROOT, week_folder, "JPEGImages")
        # print("image:", image_folder)
        
        for video_name in os.listdir(image_folder):
            vid = video_name.split("_")[0]
            # print(video_name)


            if week_id >= 8:
                if not vid.isdigit() :
                    continue



                if not vid in id_ls:
                    continue
    
            if week_id < 8:
                if video_name not in id_ls: # week_6以内的直接用video_name:
                    continue
            print("[BEGIN] Merge video:" , video_name)
            # print(os.listdir(rmem_res_path))

        

            for mask_week in os.listdir(deaot_res_path):

                # roves_val_low_res_infer_roves_in_deaot_swin_week_\d+_SwinB_DeAOTL_PRE_ckpt_unknown$
                pattern = r'roves_val_low_res_infer_roves_in_deaot_swin_week_\d+_SwinB_DeAOTL_PRE_ckpt_unknown$'


                if not bool(re.match(pattern, mask_week)):
                    continue
                

                if "roves" not in mask_week:
                    continue
                # if "ckpt176300" not in mask_week:
                #     continue

          

                mask_week_folder = os.path.join(deaot_res_path,  mask_week)


                for mask_video_name in os.listdir(mask_week_folder):
                    # print("mask_week_foler：", os.path.join(mask_week_folder,mask_video_name))

                    if week_id > 8:
                    
                        if mask_video_name.split("_")[0] == vid:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)
                    else:

                        if mask_video_name  == video_name:
                            img = os.path.join(image_folder, video_name)
                            mask = os.path.join(mask_week_folder, video_name)
                            output =  os.path.join(tmp_dir, video_name + ".mp4")
                            merge_mask_images(img, mask,output)
                            print("[DONE] Merge video:" , output)

           


if __name__ == "__main__":
    TMP_DIR = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/tmp"
    # merge_ls_SAM2(id_ls = ['0487', '0503', '0498', '0486', '0489', '0494', '0312', '0282', '0283', '0285', '0257', '0309', '0404', '0465', '0442', '0442', '0453', '0392', '0452', '0454', '0463', '0359', '0348', '0347', '0382', '0381', '0385', '0356', '0358'])
    # merge_ls_GT(id_ls=['0487', '0503', '0498', '0486', '0489', '0494', '0312', '0282', '0283', '0285', '0257', '0309', '0404', '0465', '0442', '0442', '0453', '0392', '0452', '0454', '0463', '0359', '0348', '0347', '0382', '0381', '0385', '0356', '0358'],tmp_dir=os.path.join(TMP_DIR, "GT", "SAM2_bad_case"))
    
    # merge_ls_Cutie(id_ls=['0348', '0347', '0382', '0381', '0385', '0356', '0358', '0404', '0444', '0444', '0442', '0442', '0453', '0458', '0438', '0440', '0443', '0454', '0447', '0463', '0498', '0489', '0494', '0173', '0175', '0230', '0234', '0283', '0302', '0257', '0309', '0325'])
    # merge_ls_GT(id_ls=['0348', '0347', '0382', '0381', '0385', '0356', '0358', '0404', '0444', '0444', '0442', '0442', '0453', '0458', '0438', '0440', '0443', '0454', '0447', '0463', '0498', '0489', '0494', '0173', '0175', '0230', '0234', '0283', '0302', '0257', '0309', '0325'],tmp_dir=os.path.join(TMP_DIR, "GT", "Cutie_bad_case"))
    # merge_ls_Cutie(id_ls=['0281', '0282','0283','0284', '0285', '0286','0287','0288','0289','0290','0291'],tmp_dir=os.path.join(TMP_DIR,  "Cutie_make_glass_case"))
    # merge_ls_Cutie(id_ls=['0538', '0282', '0161','0519'],tmp_dir=os.path.join(TMP_DIR,  "draw_faliure_case", "Cutie"))
    # merge_ls_GT(id_ls=['0538', '0282', '0161', '0519'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case", "GT"))
    # merge_ls_SAM2(id_ls=['0538', '0519'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case", "SAM2"))
    # merge_ls_ours_Cutie_reverse(id_ls=['0538','0282', '0161', '0519'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case", "Ours_light"))
    # merge_ls_ours_Cutie_reverse(id_ls=['0538', '0161', '0519'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case", "Ours_light"))
    # -----------------------

    # merge_ls_GT(id_ls=['tear_pepper_4'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple", "GT"))
    # merge_ls_SAM2(id_ls=['tear_pepper_4'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple", "SAM2"))
    # merge_ls_ours_Cutie_reverse(id_ls=['tear_pepper_4'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple", "Ours"))
    

    # merge_ls_Cutie(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR,  "draw_faliure_case_in_supple_2", "Cutie"))
    # merge_ls_Deaot(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "DeAOT"))
    # merge_ls_Rmem(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "Rmem"))
    # merge_ls_ours_Cutie_reverse(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "Ours"))
    # merge_ls_GT(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "GT"))
    # merge_ls_SAM2(id_ls=["0536","0266","0548",'0282', '0420','knitting_sweater_1','cut_spinach_1'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "SAM2"))

    # merge_ls_GT(id_ls=["0300",'0299', '0228','0210','0190'],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "GT"))

    # merge_ls_Cutie(id_ls=["assemble_puzzle_3","0404","0419","0359"],tmp_dir=os.path.join(TMP_DIR,  "draw_faliure_case_in_supple_2", "Cutie"))
    # merge_ls_Deaot(id_ls=["0419"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "DeAOT"))
    # merge_ls_Rmem(id_ls=["assemble_puzzle_3","0404","0419","0359"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "Rmem"))
    # merge_ls_ours_Cutie_reverse(id_ls=["assemble_puzzle_3","0404","0419","0359"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "Ours"))
    # merge_ls_GT(id_ls=["assemble_puzzle_3","0404","0419","0359"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "GT"))
    # merge_ls_SAM2(id_ls=["assemble_puzzle_3","0404","0419","0359"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "SAM2"))

    merge_ls_GT(id_ls=["crystallize_liquid_1"],tmp_dir=os.path.join(TMP_DIR, "draw_faliure_case_in_supple_2", "GT"))