
import os
import csv
import re
import matplotlib.pyplot as plt

import json
import numpy as np
from scipy.optimize import curve_fit

valid_weeks = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]

def get_distribution(dataset="cutie", res_dir = "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=r"^roves_week(\d+)$"):
# /home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week3/roves-val/Annotations
    video_scores = {}
    for week_folder in  sorted(os.listdir(res_dir)):
        match = re.match(pattern, week_folder)
        if not(match and int(match.group(1)) in valid_weeks):
            continue
        if dataset == "cutie":
            csv_path = os.path.join(res_dir, week_folder , "roves-val" , "Annotations" ,"per-sequence_results-val.csv")
        else:
            csv_path = os.path.join(res_dir, week_folder, "per-sequence_results-val.csv")

        if not os.path.exists(csv_path):
            print(csv_path , " not exists")
            continue

        print("data from:", csv_path)

        with open(csv_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                video_name = row['Sequence']
                scores = {
                    'J-Mean': float(row['J-Mean']),
                    'J_last-Mean': float(row['J_last-Mean']),
                    'J_cc-Mean': float(row['J_cc-Mean'])
                }
                video_scores[video_name] = scores

    return video_scores

def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

def plot_v_size_mtrics_split_map(video_scores, video_scores_v2, metric = "J-Mean", output_file_path = "", x_label = "velocity" ,   meta_json_path = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/size_each_seq.json",thresold=None):
    meta_json = json.load(open(meta_json_path, "rb"))

    scores = []
    metas = []

    for video_name in video_scores:
        video_id = video_name[:-2]
        obj_id = video_name[-1]
        if video_id not in meta_json:
            continue
        # print( video_name , meta_json[video_id])

        meta_dict = meta_json[video_id]
        if obj_id not in meta_dict:
            continue



        meta = meta_dict[obj_id]
        if x_label == "velocity":
            # 之前计算以为是24fps， 其实是30fps
            meta *= 30 / 24 
        score = video_scores[video_name][metric]

        if thresold != None and (meta < thresold[0] or meta >  thresold[1]):
            print("skip:" , video_name)
            continue
        
        import math
        scores.append(score)
        metas.append(math.log(meta))

    scores_v2 = []
    metas_v2 = []


    for video_name_v2 in video_scores_v2:
        video_id_v2 = video_name_v2[:-2]
        obj_id_v2 = video_name_v2[-1]
        if video_id_v2 not in meta_json:
            continue
        # print( video_name_v2 , meta_json[video_id])

        meta_dict_v2 = meta_json[video_id_v2]
        if obj_id_v2 not in meta_dict:
            continue



        meta_v2 = meta_dict_v2[obj_id_v2]
        if x_label == "velocity":
            # 之前计算以为是24fps， 其实是30fps
            meta *= 30 / 24 
        score_v2 = video_scores_v2[video_name_v2][metric]

        if thresold != None and (meta_v2 < thresold[0] or meta_v2 >  thresold[1]):
            print("skip:" , video_name_v2)
            continue
        
        import math
        scores_v2.append(score_v2)
        metas_v2.append(math.log(meta_v2))


    # 设置画布大小
    plt.figure(figsize=(10, 8))
    # 创建散点图
    plt.scatter(metas, scores, c='blue', alpha=0.5)

    plt.scatter(metas_v2, scores_v2, c='orange', alpha=0.8)
    
    # 添加拟合线
    if metas and scores:  # 确保数据不为空
        popt, _ = curve_fit(exponential_fit, metas, scores, maxfev=10000)
        x_fit = np.linspace(min(metas), max(metas), 100)
        y_fit = exponential_fit(x_fit, *popt)
        plt.plot(x_fit, y_fit, "purple",linestyle="--")

    # 添加拟合线
    if metas_v2 and scores_v2:  # 确保数据不为空
        popt, _ = curve_fit(exponential_fit, metas_v2, scores_v2, maxfev=10000)
        x_fit = np.linspace(min(metas_v2), max(metas_v2), 100)
        y_fit = exponential_fit(x_fit, *popt)
        plt.plot(x_fit, y_fit, "orange",linestyle="--")


    # 添加标题和标签
    # plt.title('Scatter Plot Example')
    # plt.ylabel('scores')
    plt.xlabel(x_label, fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend()

    # 显示图形
    plt.savefig(output_file_path)
    plt.close()
    print(f"分布图已保存到 {output_file_path}")

def plot_long_mtrics_split_map(video_scores, metric = "J-Mean", output_file_path = "", x_label = "velocity" ,   meta_json_path = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/Long_each_seq.json",thresold=None):
    meta_json = json.load(open(meta_json_path, "rb"))

    scores = []
    metas = []

    for video_name in video_scores:
        video_id = video_name[:-2]
        obj_id = video_name[-1]
        if video_id not in meta_json:
            continue
        # print( video_name , meta_json[video_id])

        meta = meta_json[video_id]
        

        score = video_scores[video_name][metric]

        if thresold != None and (meta < thresold[0] or meta >  thresold[1]):
            print("skip:" , video_name)
            continue
 


        
        import math
        scores.append(score)
        metas.append((meta))
    print("metas:", metas)
    # print(scores)
    # 创建散点图
    plt.scatter(metas, scores, c='blue', alpha=0.5, label='Random Points')

    # 添加标题和标签
    plt.title('Scatter Plot Example')
    plt.ylabel('scores')
    plt.xlabel(x_label)
    plt.legend()

    # 显示图形
    plt.savefig(output_file_path)
    plt.close()
    print(f"分布图已保存到 {output_file_path}")

def get_percent_high_meta_case( meta_json_path = None , percent=0.9):
    meta = json.load(open(meta_json_path, "r"))

    if isinstance(list(meta.values())[0] , dict):

        update_meta = {}

        for video_id , video_info in meta.items():
            for obj_id , meta_value in video_info.items():
                update_meta[f"{video_id}_{obj_id}"] = meta_value

    else:
        update_meta = meta

    print(update_meta)
    n = int(len(update_meta) * percent)
    sorted_items =  sorted(update_meta.items(), key=lambda item: item[1])

    return dict(sorted_items[n:])

def get_percent_low_meta_case( meta_json_path = None , percent=0.9):
    meta = json.load(open(meta_json_path, "r"))

    if isinstance(list(meta.values())[0] , dict):

        update_meta = {}

        for video_id , video_info in meta.items():
            for obj_id , meta_value in video_info.items():
                update_meta[f"{video_id}_{obj_id}"] = meta_value

    else:
        update_meta = meta

    print(update_meta)
    n = int(len(update_meta) * percent)
    sorted_items =  sorted(update_meta.items(), key=lambda item: item[1])

    return dict(sorted_items[:n])

def get_value_high_meta_case( meta_json_path = None , threshold= 1e-4):
    meta = json.load(open(meta_json_path, "r"))

    if isinstance(list(meta.values())[0] , dict):

        update_meta = {}

        for video_id , video_info in meta.items():
            for obj_id , meta_value in video_info.items():
                update_meta[f"{video_id}_{obj_id}"] = meta_value


    else:
        update_meta = meta

    # print( update_meta )

    video_obj_ls = [video_obj for (video_obj, value) in update_meta.items() if value > threshold ]


    return  video_obj_ls


def get_value_low_meta_case( meta_json_path = None , threshold=1e-15):
    meta = json.load(open(meta_json_path, "r"))

    if isinstance(list(meta.values())[0] , dict):

        update_meta = {}

        for video_id , video_info in meta.items():
            for obj_id , meta_value in video_info.items():
                update_meta[f"{video_id}_{obj_id}"] = meta_value

    else:
        update_meta = meta



    video_obj_ls = [video_obj for (video_obj, value) in update_meta.items() if value < threshold ]



    

    return  video_obj_ls


def plot_metrics_distribution(video_scores,output_file_path):
    j_mean_scores = [scores['J-Mean'] for scores in video_scores.values()]
    j_last_mean_scores = [scores['J_last-Mean'] for scores in video_scores.values()]
    j_cc_mean_scores = [scores['J_cc-Mean'] for scores in video_scores.values()]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.hist(j_mean_scores, bins=10, color='blue', alpha=0.7)
    plt.title('J-Mean Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(j_last_mean_scores, bins=10, color='green', alpha=0.7)
    plt.title('J_last-Mean Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(j_cc_mean_scores, bins=10, color='red', alpha=0.7)
    plt.title('J_cc-Mean Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()
    print(f"分布图已保存到 {output_file_path}")

def find_low_j_mean_samples(video_scores, threshold=0.5):

    low_j_mean_samples = {video: scores for video, scores in video_scores.items() if scores['J-Mean'] < threshold}
    return low_j_mean_samples



def find_low_percent_samples(video_scores, percent=0.9):
    n = int(len(video_scores) * percent)

    sorted_items =  sorted(video_scores.items(), key=lambda item: item[1]["J-Mean"])

    return dict(sorted_items[:n])

def find_high_percent_samples(video_scores, percent=0.5):
    n = int(len(video_scores) * percent)

    sorted_items =  sorted(video_scores.items(), key=lambda item: item[1]["J-Mean"])

    return dict(sorted_items[n:])

def calculate_average_scores(video_scores):
    average_scores = {}
    J_mean = 0
    J_ls = 0
    J_cc = 0
    cnt = 0
    for video_name, scores in video_scores.items():
        J_mean  += scores["J-Mean"]
        J_ls += scores['J_last-Mean']
        J_cc += scores["J_cc-Mean"]
        cnt += 1
        
    average_scores["J-Mean"] = J_mean / float( cnt)
    average_scores['J_last-Mean'] = J_ls / float(cnt)
    average_scores["J_cc-Mean"] = J_cc / float(cnt)
    return average_scores

def filter_scores(scores, common_keys, excluded_videos):
    new_scores = {}
    for videos_id in scores.keys():
        if (not videos_id.startswith(excluded_videos)) and (videos_id in common_keys):
            new_scores[videos_id] = scores[videos_id]
    return new_scores

def is_substr_in_strls(substr, strls):
    flag = False
    final_target_str = None
    for target_str in strls:
        if substr in target_str:
            flag = True
            final_target_str = target_str
            break

    return flag , final_target_str

def add_challenge_in_meta(seq_obj_ls, challenge= "FastMotion" , ori_meta_path = None, out_meta_path = None):
    update_meta = json.load(open(ori_meta_path, "r") )
    for seq_obj in seq_obj_ls:
        obj_id = "id_" +  seq_obj[-1:]
        seq_name = seq_obj[:-2]


        flag , video_key = is_substr_in_strls(seq_name, list(update_meta.keys()))
        if flag:
            if obj_id in update_meta[video_key] :
                # print(video_info)
                update_meta[video_key] [obj_id].append(challenge)
   
            else:

                update_meta[video_key] [obj_id] = [challenge]
            # print("old seq:", video_key)

            # print( update_meta[video_key] [obj_id])

 
            # print("new seq:" , seq_name)
        


    with open(out_meta_path, "w") as f:
        json.dump(update_meta , f)
    print("Save into: " , out_meta_path)

        


if __name__ == "__main__":
    # video_scores =  get_distribution_SAM2("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs")
    pattern = r'^roves_week(\d+)_mega_v4_72800$'
    meta_root = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta"
    print("loading scores...")
    our_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=pattern)
    cutie_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output", pattern=r'^roves_week(\d+)$')
    
    sam2_video_scores = get_distribution(dataset="sam2",
                                         res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs", 
                                         pattern=r"sam2_in_roves_week_(\d+)$")
    aot_video_scores = get_distribution(dataset="aot",
                                        res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves",
                                        pattern=r"test_roves_in_aot_week_(\d+)$")
    deaot_video_scores = get_distribution(dataset="deaot",
                                          res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves",
                                          pattern=r"test_roves_week_in_deaotRmem_(\d+)$")
    xmem_video_scores = get_distribution(dataset="xmem",
                                         res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output",
                                         pattern=r'^roves_week(\d+)$')
    # plot_metrics_distribution(video_scores,"./myVOS_metrics.png")
    excluded_videos = ("mv_cola_1", "mv_cola_2", "mv_cola_3", "mv_energy_1", "mv_energy_2", "mv_lime_1", 
                       "mv_lime_2", "mv_mark_1", "mv_cup_4", "mv_cup_5", "mv_redpen_2", "mv_redpen_3", "mv_brocolli_1",
                       "mv_brocolli_8", "mv_egg_2", "mv_egg_4", "mv_pepper_2", "mv_pepper_3", "mv_green_pepper_5", "0209_break_glass_5")
    
    print("filtering...")

    fig_root = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/evaluation/fig_result"

    common_keys = set(our_video_scores.keys()) & \
                    set(cutie_video_scores.keys()) & \
                    set(sam2_video_scores.keys()) 
                    # set(aot_video_scores.keys()) &\
                    # set(deaot_video_scores.keys()) &\
                    # set(xmem_video_scores.keys())


    # ！这里加一个hard subset往这里一放置就可以过滤一下了
    
    new_our_video_scores = filter_scores(our_video_scores, common_keys, excluded_videos)

    meta_dict = {
        "Size": r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/size_each_seq.json",
        "Velocity": r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/v_each_seq.json",
        "Video Frame": r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/Long_each_seq.json"
    }

    ls_scores = [new_our_video_scores,sam2_video_scores ]

    # 画散点图
    plot_v_size_mtrics_split_map(new_our_video_scores, cutie_video_scores , metric = "J-Mean", x_label = "Region Ratio/log"  ,output_file_path =os.path.join(fig_root, "Our_size_J_mean.png"), meta_json_path =  meta_dict["Size"])
    plot_v_size_mtrics_split_map(new_our_video_scores, cutie_video_scores ,  metric = "J-Mean", x_label = "Velocity " ,output_file_path =os.path.join(fig_root, "Our_V_J_mean.png"), meta_json_path =  meta_dict["Velocity"])
    # plot_long_mtrics_split_map(new_our_video_scores, metric = "J-Mean", x_label = "Video Frame" ,output_file_path =os.path.join(fig_root, "Our_L_J_mean.png"), meta_json_path = meta_dict["Video Frame"], thresold= (0,800))

    #  ---------------- 统计数据

    # # 统计速度前三分之一的case
    # get_percent_high_meta_case(meta_dict["Velocity"],percent = 0.7 )


    # # 统计size小于三分之一的case
    # get_percent_low_meta_case(meta_dict["Size"],percent = 0.3 )

    # # 统计高速 (大于1e-15)的物体
    # FM_ls = get_value_high_meta_case(meta_dict["Velocity"], threshold = 0.00000030)

    # # 统计小物体 （小于1e-4）的物体
    # SM_ls = get_value_low_meta_case(meta_dict["Size"], threshold =0.0183)

    # # 统计的视频
    # FM_ls = get_value_high_meta_case(meta_dict["Velocity"], threshold = 1e-15 )

    # print(SM_ls)
    # print(FM_ls)
    # print("num of FS:" , len(FM_ls))

    # ori_meta_path = os.path.join(meta_root, "hard_case_challenge_78_wo_VLS.json")

    # add_challenge_in_meta(FM_ls , challenge= "FastMotion" ,ori_meta_path = ori_meta_path, out_meta_path = os.path.join(meta_root, "challenge_with_FM_new_78.json") )
    # add_challenge_in_meta(SM_ls , challenge= "Small Object" ,ori_meta_path =os.path.join(meta_root, "challenge_with_FM_new_78.json") , out_meta_path = os.path.join(meta_root, "challenge_with_FM_SM_new_78.json") )
    # add_challenge_in_meta(FM_ls , challenge= "FastMotion" ,ori_meta_path = ori_meta_path, out_meta_path = os.path.join(meta_root, "challenge_with_FM.json") )

    
    
    # ------------------


    

    # print("calculating...")

    # print("---------------------------------Our score-----------------------------")
    # low_samples = find_low_percent_samples(new_our_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_our_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_our_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_our_video_scores))


    # new_cutie_video_scores = filter_scores(cutie_video_scores, common_keys, excluded_videos)
    
    # print("----------------------------------Cutie score-----------------------------")
    # low_samples = find_low_percent_samples(new_cutie_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_cutie_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_cutie_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_cutie_video_scores))


    # new_sam2_video_scores = filter_scores(sam2_video_scores, common_keys, excluded_videos)
    
    # print("----------------------------------SAM2 score-----------------------------")
    # low_samples = find_low_percent_samples(new_sam2_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_sam2_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_sam2_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_sam2_video_scores))

    # new_aot_videos_scores = filter_scores(aot_video_scores, common_keys, excluded_videos)
    
    # print("-----------------------------------AOT score-----------------------------")
    # low_samples = find_low_percent_samples(new_aot_videos_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_aot_videos_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_aot_videos_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_aot_videos_scores))

    # new_deaot_videos_scores = filter_scores(deaot_video_scores, common_keys, excluded_videos)
    
    # print("-----------------------------------DEAOT score-----------------------------")
    # low_samples = find_low_percent_samples(new_deaot_videos_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_deaot_videos_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_deaot_videos_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_deaot_videos_scores))

    # new_xmem_video_scores = filter_scores(xmem_video_scores, common_keys, excluded_videos)
    
    # print("-----------------------------------XMem score-----------------------------")
    # low_samples = find_low_percent_samples(new_xmem_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_xmem_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_xmem_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_xmem_video_scores))




     



