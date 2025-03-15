
import os
import csv
import re
import matplotlib.pyplot as plt
import json
from plot_phase_transition import excluded_videos
# from plot_challenge_distribution import get_test_split
from  get_hard_mid_easy_split import get_avg_score_for_objs

from plot_phase_transition import *

valid_weeks = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]

def get_test_split(meta_root, mode= "hard", has_keys = None):
    subset = set(json.load(open(os.path.join(meta_root ,   mode + "_subset.json"))))
    print(os.path.join(meta_root ,    mode + "_subset.json"))
    if has_keys is not None:
        subset = subset & has_keys

    return subset

def substr_is_keyls(substr, key_ls):
    for key in key_ls:
        if substr in key:
            return key
    return None



def get_test_txt_split(meta_root, mode= "hard", has_keys = None, prefix = ""):
    split_txt =  (os.path.join(meta_root ,  prefix  +  "all_" + mode + "_objs.txt"))


    with open(split_txt, "r") as file:
        subset = set ([line.strip() for line in file.readlines() if len(line.strip()) >= 0])


    # print(mode, ":" , subset)
    res = []

    for substr in subset:
        key = substr_is_keyls(substr , has_keys)
        if key is not None:
            res.append(key)

    res = set(res)

    return res

def get_all_core_objs_from(common_keys):
    split_txt = "/home/bingxing2/home/scx8ah2/dataset/ROVES_meta/all_core_objs.txt"
    with open(split_txt, "r") as file:
        all_videos = set ([line.strip() for line in file.readlines() if len(line.strip()) >= 0])
    matched_keys = []
    for video_name in all_videos:
        num_str = match_prefix_numbers(video_name)
        if is_to_remove(num_str) and starts_with_digit_underscore(video_name):
            video_name = remove_prefix(video_name)
        
        for key in common_keys:
            if key.startswith(video_name):
                matched_keys.append(key)
        
    return matched_keys



def get_distribution(dataset="cutie", res_dir = "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=r"^roves_week(\d+)$"):
# /home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week3/roves-val/Annotations
    video_scores = {}
    for week_folder in  sorted(os.listdir(res_dir)):
        match = re.match(pattern, week_folder)
        if not(match and int(match.group(1)) in valid_weeks):
            continue
        if dataset == "cutie":
            csv_path = os.path.join(res_dir, week_folder , "roves-val" , "Annotations" ,"per-sequence_results-val.csv")
        # /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/checkpoints/pretrain_vost/roves_eval_week_11/to_eval_pred/per-sequence_results-val.csv
        elif dataset == "tam_vt":
            csv_path = os.path.join(res_dir, week_folder , "to_eval_pred", "per-sequence_results-val.csv")
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

def mean_video_scores(scores_ls, common_keys):
    mean_video_scores = {}

    for common_key in common_keys:
        model_J_mean = 0
        model_J_cc = 0
        model_J_last = 0
        flag = False
        cnt = 0


        for idx,  video_scores in enumerate(scores_ls):
            # print("video_scores :" , len(video_scores) )
            # print("idx:",idx)
            if common_key not in video_scores and idx != len(scores_ls) - 1:
                flag = True
                break

            if common_key not in video_scores and  idx == len(scores_ls) - 1:
                continue

            


            model_J_mean += video_scores[common_key]["J-Mean"]
            model_J_cc += video_scores[common_key]["J_cc-Mean"]
            model_J_last  +=video_scores[common_key]["J_last-Mean"]
            cnt += 1

        if flag:
            # print("!!!:",common_key)
            continue
        
        mean_video_scores[common_key] = {}

        mean_video_scores[common_key]["J-Mean"] = model_J_mean / cnt
        mean_video_scores[common_key]["J_cc-Mean"] = model_J_cc  /cnt
        mean_video_scores[common_key]["J_last-Mean"] = model_J_last / cnt
       
        # print("mean len:", len(mean_video_scores))
    
    return mean_video_scores

def normalize_dict_values(input_dict,neg=False):
    # 获取字典中的所有值
    values = list(input_dict.values())
    
    # 计算最小值和最大值
    min_value = min(values)
    max_value = max(values)
    
    # 计算归一化参数
    normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in input_dict.items()}

    if neg:
        normalized_dict = {k:  1 - v for k, v in input_dict.items()}

    
    return normalized_dict

def get_the_metric_dict(input_dict, mode):
    average_dict = {}
    for k, v in input_dict.items():
        if mode == "avg":
            average_value = 0
            for metric_value in v.values():
                average_value += metric_value
            average_value /= len(v.values())
            average_dict[k] = average_value 
        elif mode == "J_cc":
            average_dict[k] = v["J_cc-Mean"]
        elif mode == "J_st":
            average_dict[k] = v["J_last-Mean"]
        elif mode == "J":
            average_dict[k] = v["J-Mean"]
        

    return average_dict

            # normalized_dict[k] = {
            #     'J-Mean': 1 - (v['J-Mean']),
            #     'J_cc-Mean': 1 - (v['J_cc-Mean'] ),
            #     'J_last-Mean': 1 - (v['J_last-Mean'] ) 
            # }

def combine_scores(real_mean_vdieo_scores  ,usr_study_video_scores, delta = 0.5, mode = "avg"):
    print("real_mean:" , len(real_mean_vdieo_scores))
    print("usr study:" , len(usr_study_video_scores))
    assert len(real_mean_vdieo_scores.keys()) == len(usr_study_video_scores)
    real_mean_vdieo_scores = get_the_metric_dict(real_mean_vdieo_scores, mode =mode)
    neg_normalized_real_scores = normalize_dict_values(real_mean_vdieo_scores, neg = True)
    normalized_usr_study_scores = normalize_dict_values(usr_study_video_scores, neg = False)

    combined_scores = {}

    # print(normalized_usr_study_scores)
    # print(neg_normalized_real_scores)


    for key in real_mean_vdieo_scores.keys():
        combined_scores[key] = delta * neg_normalized_real_scores[key] + (1- delta) * normalized_usr_study_scores[key]
    return combined_scores

def get_hard_mid_easy(input_dict, meta_root="", prefix="test_"):
    # 获取字典中的所有项，并按值排序
    sorted_items = sorted(input_dict.items(), key=lambda item: item[1], reverse=True)
    
    # 计算前30%的数量
    top_33_percent_count = int(len(sorted_items) * 0.33)
    # 获取前33%的键
    top_hard_objs = [item[0] for item in sorted_items[:top_33_percent_count]]
    with open(os.path.join(meta_root,prefix + "all_hard_objs.txt" ), 'w') as f:
        for obj in top_hard_objs:
            f.write(obj + "\n")

    
    # 计算前66%的数量
    top_66_percent_count = int(len(sorted_items) * 0.66)
    # 获取前66%的键
    top_mid_objs = [item[0] for item in sorted_items[:top_66_percent_count]]
    # 获取字典中的所有项，并按值排序
    with open(os.path.join(meta_root,prefix +"all_mid_objs.txt" ), 'w') as f:
        for obj in top_mid_objs:
            f.write(obj + "\n")


    with open(os.path.join(meta_root,prefix + "all_easy_objs.txt" ), 'w') as f:
        for obj in sorted_items:
            f.write(obj[0] + "\n")

    print("save:" , os.path.join(meta_root,prefix + "all_easy_objs.txt" ))

def A_fail_B_good(A_scores, B_scores, fail_threshold=0.5 ,success_threshold=0.8,  all_threshold = 1,  metric = "J-Mean"):
    key_ls = []
    for key in (A_scores.keys()):
        if key not in (B_scores.keys()):
            continue
        
        if A_scores[key] [metric]< fail_threshold and B_scores[key] [metric]> success_threshold and \
          B_scores[key] [metric] <  all_threshold  and  A_scores[key] [metric]<  all_threshold :
            key_ls.append(key)

    return key_ls
        


if __name__ == "__main__":
    # video_scores =  get_distribution_SAM2("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs")
    test_prefix = "test_"
    produce_prefix = "test_"

    pattern = r'^roves_week(\d+)_mega_v4_72800$'
    meta_root = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta"
    print("loading scores...")
    our_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=pattern)
    our_video_scores_boost_1_01 = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", 
                                                   pattern=r"roves_week(\d+)_booster_1.01")
    our_video_scores_boost_1_2 = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", 
                                                   pattern=r"roves_week(\d+)_booster_1.2")
    our_video_scores_boost_1_5 = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", 
                                                   pattern=r"roves_week(\d+)_booster_1.5")
    our_video_scores_boost_2_0 = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", 
                                                   pattern=r"roves_week(\d+)_booster_2.0")
    cutie_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output", pattern=r'^roves_week(\d+)$')
    
    sam2_video_scores = get_distribution(dataset="sam2",
                                         res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs", 
                                         pattern=r"sam2_in_roves_week_(\d+)$")
    # aot_video_scores = get_distribution(dataset="aot",
    #                                     res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves",
    #                                     pattern=r"test_roves_in_aot_week_(\d+)$")

    # rmem_video_scores = get_distribution(dataset="rmem",
    #                                     res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves",
    #                                     pattern=r"test_roves_week_in_deaotRmem_(\d+)$")
    rmem_video_scores = get_distribution(dataset="rmem",
                                        res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves",
                                        pattern=r"test_rmem_wo_pte_roves_in_aot_week_(\d+)$")

    deaot_video_scores = get_distribution(dataset="deaot",
                                          res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves",
                                          pattern=r"roves_val_low_res_infer_roves_in_deaot_swin_week_(\d+)_SwinB_DeAOTL_PRE_ckpt_unknown$")
                                        #    /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves/roves_val_low_res_infer_roves_in_deaot_swin_week_0_SwinB_DeAOTL_PRE_ckpt_unknown
    xmem_video_scores = get_distribution(dataset="xmem",
                                         res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output",
                                         pattern=r'^roves_week(\d+)$')

    ours_rmem_scores = get_distribution(dataset="rmem",
                                        res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/RMem_ReVOS/results/aotplus_R50_DeAOTL/pre_vost/eval/roves", 
                                         pattern=r'^test_rmem_wo_pte_roves_950_ema_week_(\d+)$')


    tam_vt_scores = get_distribution(dataset="tam_vt", 
                                    res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/checkpoints/pretrain_vost",
                                    pattern=r'^roves_eval_week_(\d+)$')

    
    


                                        
    # plot_metrics_distribution(video_scores,"./myVOS_metrics.png")
    # excluded_videos = ("mv_cola_1", "mv_cola_2", "mv_cola_3", "mv_energy_1", "mv_energy_2", "mv_lime_1", 
                    #    "mv_lime_2", "mv_mark_1", "mv_cup_4", "mv_cup_5", "mv_redpen_2", "mv_redpen_3", "mv_brocolli_1",
                    #    "mv_brocolli_8", "mv_egg_2", "mv_egg_4", "mv_pepper_2", "mv_pepper_3", "mv_green_pepper_5", "0209_break_glass_5",
                    #    "tear_green_pepper_1", "break_egg_3","cut_green_pepper_4","cut&break_brocolli_5", "tear_green_pepper_6") # 这部分是mask调整过的
    
    print("filtering...")



    before_common_keys = set(our_video_scores.keys()) & \
                    set(cutie_video_scores.keys()) & \
                    set(sam2_video_scores.keys())  & \
                    set(xmem_video_scores.keys())  & \
                    set(rmem_video_scores.keys())

    common_keys = set(our_video_scores.keys()) & \
                    set(cutie_video_scores.keys()) & \
                    set(sam2_video_scores.keys())  & \
                    set(xmem_video_scores.keys())  & \
                    set(rmem_video_scores.keys()) & \
                    set(tam_vt_scores.keys()) 
                    # set(deaot_video_scores.keys()) 
                    # set(aot_video_scores.keys()) 

    print("common_keys:", len(common_keys))
    print("before_common_keys:", len(before_common_keys))

    

    hard_keys =  get_test_txt_split(meta_root, mode= "hard", has_keys = common_keys, prefix = test_prefix)
    mid_keys = get_test_txt_split(meta_root, mode= "mid", has_keys = common_keys, prefix = test_prefix)
    easy_keys = get_test_txt_split(meta_root, mode= "easy", has_keys = common_keys, prefix = test_prefix)


    # hard_keys = get_test_split(meta_root, mode= "hard", has_keys = common_keys)
    # mid_keys =   set.union( get_test_split(meta_root, mode= "mid", has_keys = common_keys) ,  hard_keys)
    # easy_keys =  set.union(get_test_split(meta_root, mode= "easy", has_keys = common_keys) ,  mid_keys)
                    
    
    new_our_video_scores = filter_scores(our_video_scores, common_keys, excluded_videos)
    print("calculating...")



    print("---------------------------------Our score-----------------------------")

    hard_samples = filter_scores(our_video_scores, hard_keys , excluded_videos)
    mid_samples = filter_scores(our_video_scores, mid_keys , excluded_videos)
    easy_samples = filter_scores(our_video_scores, easy_keys , excluded_videos)
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))




    # print("---------------------------------Our score-----------------------

    # print("---------------------------------Our score-----------------------------")
    # low_samples = find_low_percent_samples(new_our_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # hard_subset =  set(H_low_samples)
    # print("hard subset总数:", len(hard_subset))
    # with open(os.path.join(meta_root, "hard_subset.json"),"w") as f:
    #     json.dump(sorted(list(hard_subset)), f)
    
    
    # low_samples = find_low_percent_samples(new_our_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # mid_subset = set(low_samples) - set(H_low_samples)
    # print("mid subset总数:", len(mid_subset))
    # with open(os.path.join(meta_root, "mid_subset.json"),"w") as f:
    #     json.dump(sorted(list(mid_subset)), f)
    


    # print("Simple(All)总数:", len(new_our_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_our_video_scores))

    # easy_subset = set(new_our_video_scores) - set(low_samples)
    # print("easy subset总数:", len(easy_subset))
    # with open(os.path.join(meta_root, "easy_subset.json"),"w") as f:
    #     json.dump(sorted(list(easy_subset)), f)
    


    new_cutie_video_scores = filter_scores(cutie_video_scores, common_keys, excluded_videos)
    
    print("----------------------------------Cutie score-----------------------------")


    hard_samples = filter_scores(cutie_video_scores ,hard_keys , excluded_videos)
    mid_samples = filter_scores(cutie_video_scores , mid_keys ,excluded_videos)
    easy_samples = filter_scores(cutie_video_scores ,easy_keys , excluded_videos)

    with open("from_all_video_scores.txt", "w") as f:
        for obj in easy_samples.keys():
            f.write(obj + '\n')
    
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))



    # low_samples = find_low_percent_samples(new_cutie_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_cutie_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_cutie_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_cutie_video_scores))


    new_sam2_video_scores = filter_scores(sam2_video_scores, common_keys, excluded_videos)
    
    print("----------------------------------SAM2 score-----------------------------")

    hard_samples = filter_scores(sam2_video_scores , hard_keys ,excluded_videos)
    mid_samples = filter_scores(sam2_video_scores , mid_keys ,excluded_videos)
    easy_samples = filter_scores(sam2_video_scores , easy_keys ,excluded_videos)
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    # low_samples = find_low_percent_samples(new_sam2_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_sam2_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_sam2_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_sam2_video_scores))




    new_rmem_video_scores = filter_scores(rmem_video_scores, common_keys, excluded_videos)
    
    print("-----------------------------------RMEM score-----------------------------")

    hard_samples = filter_scores(new_rmem_video_scores ,hard_keys , excluded_videos)
    mid_samples = filter_scores(new_rmem_video_scores ,mid_keys , excluded_videos)
    easy_samples = filter_scores(new_rmem_video_scores ,easy_keys , excluded_videos)
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))


    # low_samples = find_low_percent_samples(new_rmem_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_rmem_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_rmem_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(easy_samples))

    # print("undone case:",set(new_cutie_video_scores.keys()) - set(new_rmem_video_scores.keys()))

    # new_aot_videos_scores = filter_scores(aot_video_scores, common_keys, excluded_videos)
    
    # print("-----------------------------------AOT score-----------------------------")

    # hard_samples = filter_scores(aot_videos_scores ,hard_keys , excluded_videos)
    # mid_samples = filter_scores(aot_videos_scores ,mid_keys , excluded_videos)
    # easy_samples = filter_scores(aot_videos_scores ,easy_keys , excluded_videos)
    # print("Hard subset score:" , calculate_average_scores(hard_samples))
    # print("Hard subset num:" , len(hard_samples))
    # print("Mid subset score:" , calculate_average_scores(mid_samples))
    # print("Mid subset num:" , len(mid_samples))
    # print("Easy subset score:" , calculate_average_scores(easy_samples))
    # print("Easy subset num:" , len(easy_samples))


    # low_samples = find_low_percent_samples(new_aot_videos_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_aot_videos_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_aot_videos_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_aot_videos_scores))

    new_deaot_video_scores = filter_scores(deaot_video_scores, common_keys, excluded_videos)
    
    print("-----------------------------------DEAOT score-----------------------------")

    hard_samples = filter_scores(deaot_video_scores ,hard_keys , excluded_videos)
    mid_samples = filter_scores(deaot_video_scores  ,mid_keys , excluded_videos)
    easy_samples = filter_scores(deaot_video_scores  , easy_keys ,excluded_videos)
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    # low_samples = find_low_percent_samples(new_deaot_videos_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_deaot_videos_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_deaot_videos_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_deaot_videos_scores))

    new_xmem_video_scores = filter_scores(xmem_video_scores, common_keys, excluded_videos)
    
    print("-----------------------------------XMem score-----------------------------")
    hard_samples = filter_scores(xmem_video_scores , hard_keys ,excluded_videos)
    mid_samples = filter_scores(xmem_video_scores  , mid_keys ,excluded_videos)
    easy_samples = filter_scores(xmem_video_scores    , easy_keys ,excluded_videos)
    print("Hard subset score:" , calculate_average_scores(hard_samples))
    print("Hard subset num:" , len(hard_samples))
    print("Mid subset score:" , calculate_average_scores(mid_samples))
    print("Mid subset num:" , len(mid_samples))
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))


    # low_samples = find_low_percent_samples(new_xmem_video_scores, 0.33)
    # print("Hard(33%)总数:", len(low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(low_samples))

    # low_samples = find_low_percent_samples(new_xmem_video_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # print("Simple(All)总数:", len(new_xmem_video_scores))
    # print("Simple(All)均分:", calculate_average_scores(new_xmem_video_scores))




    

    # H_low_samples = find_low_percent_samples(mean_vdieo_scores, 0.33)
    # print("Hard(33%)总数:", len(H_low_samples))
    # print("Hard(33%)均分:", calculate_average_scores(H_low_samples))

    # hard_subset =  set(H_low_samples)
    # print("hard subset总数:", len(hard_subset))
    # with open(os.path.join(meta_root, "xmem_hard_subset.json"),"w") as f:
    #     json.dump(sorted(list(hard_subset)), f)
    
    
    # low_samples = find_low_percent_samples(mean_vdieo_scores, 0.66)
    # print("Mid(66%)总数:", len(low_samples))
    # print("Mid(66%)均分:", calculate_average_scores(low_samples))

    # mid_subset = set(low_samples) - set(H_low_samples)
    # print("mid subset总数:", len(mid_subset))
    # with open(os.path.join(meta_root, "xmem_mid_subset.json"),"w") as f:
    #     json.dump(sorted(list(mid_subset)), f)
    



    # easy_subset = set(mean_vdieo_scores) - set(low_samples)
    # print("easy subset总数:", len(easy_subset))
    # with open(os.path.join(meta_root, "xmem_easy_subset.json"),"w") as f:
    #     json.dump(sorted(list(easy_subset)), f)
    core_keys =  get_all_core_objs_from(common_keys)



    # ---------统计分数平均数------------

    print("-----------------------------------Ours(booster 1.1) score-----------------------------")
    core_samples = filter_scores(our_video_scores, core_keys , excluded_videos)

    print("Core subset score:" , calculate_average_scores(core_samples))
    print("Core subset num:" , len(core_samples))

    easy_samples = filter_scores(our_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------Ours(booster 1.01) score-----------------------------")
    core_samples = filter_scores(our_video_scores_boost_1_01, core_keys , excluded_videos)

    print("Core subset score:" , calculate_average_scores(core_samples))
    print("Core subset num:" , len(core_samples))

    easy_samples = filter_scores(our_video_scores_boost_1_01, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------Ours(booster 1.2) score-----------------------------")
    core_samples = filter_scores(our_video_scores_boost_1_2, core_keys , excluded_videos)

    print("Core subset score:" , calculate_average_scores(core_samples))
    print("Core subset num:" , len(core_samples))

    easy_samples = filter_scores(our_video_scores_boost_1_2, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------Ours(booster 1.5) score-----------------------------")
    core_samples = filter_scores(our_video_scores_boost_1_5, core_keys , excluded_videos)

    print("Core subset score:" , calculate_average_scores(core_samples))
    print("Core subset num:" , len(core_samples))

    easy_samples = filter_scores(our_video_scores_boost_1_5, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------Ours(booster 2.0) score-----------------------------")
    core_samples = filter_scores(our_video_scores_boost_2_0, core_keys , excluded_videos)

    print("Core subset score:" , calculate_average_scores(core_samples))
    print("Core subset num:" , len(core_samples))

    easy_samples = filter_scores(our_video_scores_boost_2_0, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))


    print("-----------------------------------TAM_VT score-----------------------------")

    core_samples = filter_scores(tam_vt_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(tam_vt_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------Cutie score-----------------------------")
    core_samples = filter_scores(cutie_video_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(cutie_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------SAM2 score-----------------------------")
    core_samples = filter_scores(sam2_video_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(sam2_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------RMEM score-----------------------------")
    core_samples = filter_scores(rmem_video_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(rmem_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))


    print("-----------------------------------XMEM score-----------------------------")
    core_samples = filter_scores(xmem_video_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(xmem_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------DeAOT score-----------------------------")
    core_samples = filter_scores(deaot_video_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(deaot_video_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    print("-----------------------------------RMem+ReVOS(ours) score-----------------------------")
    core_samples = filter_scores(ours_rmem_scores ,core_keys , excluded_videos)
    print("Core subset score:" , calculate_average_scores( core_samples))
    print("Core subset num:" , len( core_samples))
    easy_samples = filter_scores(ours_rmem_scores, easy_keys , excluded_videos)
    print("Easy subset score:" , calculate_average_scores(easy_samples))
    print("Easy subset num:" , len(easy_samples))

    ours_easy_samples = filter_scores(our_video_scores  , easy_keys ,excluded_videos)

    cutie_easy_samples = filter_scores(cutie_video_scores , easy_keys ,excluded_videos)

    sam2_easy_samples = filter_scores(sam2_video_scores , easy_keys ,excluded_videos)

    rmem_easy_sample = filter_scores(new_rmem_video_scores , easy_keys , excluded_videos)

    xmem_easy_samples = filter_scores(xmem_video_scores    ,  easy_keys  ,excluded_videos)

    deaot_easy_samples = filter_scores(deaot_video_scores  , easy_keys ,excluded_videos)

    scores_ls = [cutie_easy_samples , sam2_easy_samples,rmem_easy_sample,deaot_easy_samples] # deaot必須放在最后

    real_mean_vdieo_scores = mean_video_scores(scores_ls, common_keys)


    # print("mean_video_scores:", mean_vdieo_scores )

    # usr_study_video_scores = get_avg_score_for_objs(weight= {"颜色":1, "形状":1, "位置":1, "状态":1, "视频质量":1, "相似干扰":1, "遮挡":1, "长视频":1, "透明物体":1})

    # combine_video_scores = combine_scores(real_mean_vdieo_scores ,usr_study_video_scores , delta = 0, mode = "avg")

    meam_adhoc_Jcc = A_fail_B_good(cutie_easy_samples ,ours_easy_samples , metric= "J_cc-Mean")  
    meam_adhoc_Jst = A_fail_B_good(real_mean_vdieo_scores, ours_easy_samples , metric= "J_last-Mean")  
    meam_adhoc_J= A_fail_B_good(rmem_easy_sample ,ours_easy_samples , metric= "J-Mean")  
    print(meam_adhoc_J)
    print(len(meam_adhoc_J))




    # print(combine_video_scores)

    # get_hard_mid_easy(combine_video_scores, meta_root = meta_root)

    # J_cc_some_case = A_fail_B_good(rmem_easy_sample,ours_easy_samples )
    # # print(len(J_cc_some_case))

    # J_st_some_case = A_fail_B_good(rmem_easy_sample,ours_easy_samples , metric= "J_last-Mean")
    # # print(len(J_st_some_case))

    # J_cc_some_case = A_fail_B_good(rmem_easy_sample,ours_easy_samples , metric= "J_cc-Mean")
    # # print(len(J_cc_some_case))

    # Rmem_adhoc = (set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case))

    # print((set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case)))
    # print(len(set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case)))




    # J_cc_some_case = A_fail_B_good(cutie_easy_samples,ours_easy_samples )
    # print(len(J_cc_some_case))

    # J_st_some_case = A_fail_B_good(cutie_easy_samples,ours_easy_samples , metric= "J_last-Mean")
    # print(len(J_st_some_case))

    # J_cc_some_case = A_fail_B_good(cutie_easy_samples,ours_easy_samples , metric= "J_cc-Mean")
    # print(len(J_cc_some_case))



    # print((set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case)))
    # print(len(set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case)))

    # Cutie_adhoc = (set(J_cc_some_case) | set(J_st_some_case)| set(J_cc_some_case))


    # print(len(Cutie_adhoc | Rmem_adhoc))
    # candidate_keys = (Cutie_adhoc | Rmem_adhoc)

    # print("candidate:" ,candidate_keys )




    # ours_core_samples = filter_scores(our_video_scores,Cutie_adhoc  , excluded_videos)


    # print("Hard(33%)总数:", len(ours_core_samples))
    # print("Hard(33%)均分:", calculate_average_scores(ours_core_samples))

    # cutie_core_samples = filter_scores(cutie_video_scores ,Cutie_adhoc  , excluded_videos)


    # print("Hard(33%)总数:", len(cutie_core_samples))
    # print("Hard(33%)均分:", calculate_average_scores(cutie_core_samples))


    # ours_core_samples = filter_scores(our_video_scores,meam_adhoc_J  , excluded_videos)


    # print("Hard(33%)总数:", len(ours_core_samples))
    # print("Hard(33%)均分:", calculate_average_scores(ours_core_samples))

    # cutie_core_samples = filter_scores(cutie_video_scores ,ours_core_samples  , excluded_videos)


    # print("Hard(33%)总数:", len(cutie_core_samples))
    # print("Hard(33%)均分:", calculate_average_scores(cutie_core_samples))
     



