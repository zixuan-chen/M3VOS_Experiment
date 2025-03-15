import json
import re
from get_metric_distribution import get_distribution , get_test_split
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib
from plot_phase_transition import excluded_videos
import os
import numpy as np


def remove_prefix(input_string):
    # 使用正则表达式去除前面的数字和下划线
    result = re.sub(r'^\d+_', '', input_string)
    return result

def match_prefix_numbers(input_string):
    # 使用正则表达式匹配开头的数字
    match = re.match(r'^\d+', input_string)
    return match.group() if match else None

def is_to_remove(num):
    return int(num) < 150 or num in ['0009', '0010', '0011', '0014', '0015', '0016', '0019', '0020', 
                                 '0021', '0022', '0023', '0024', '0026', '0027', '0028', '0029', 
                                 '0030', '0031', '0033', '0035', '0037', '0040', '0041', '0042', 
                                 '0043', '0044', '0045', '0046', '0048', '0049', '0050', '0051', 
                                 '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', 
                                 '0060', '0061', '0062', '0063', '0064', '0065', '0066', '0067', 
                                 '0068', '0069', '0070', '0071', '0072', '0073', '0074', '0075', 
                                 '0076', '0077', '0078', '0079', '0080', '0081', '0082', '0083', 
                                 '0084', '0085', '0086', '0087', '0088', '0089', '0090', '0091', 
                                 '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', 
                                 '0100', '0101', '0102', '0103', '0104', '0105', '0106', '0107', 
                                 '0108', '0109', '0110', '0111', '0112', '0113', '0114', '0115', 
                                 '0116', '0117', '0118', '0119', '0120', '0121', '0122', '0123', 
                                 '0124', '0125', '0126', '0127', '0128', '0129', '0130', '0131', 
                                 '0132', '0133', '0134', '0135', '0136', '0137', '0138', '0139', 
                                 '0140', '0141', '0142', '0143', '0144', '0145', '0146', '0147', 
                                 '0148', '0149', '0241', '0242', '0243', '0244', '0245', '0246', 
                                 '0247', '0248', '0249', '0250', '0251', '0252', '0253', '0254', 
                                 '0255', '0256']

def get_phase2obj():
    with open("/home/bingxing2/home/scx8ah2/dataset/ROVES_meta/all_phase_label.json") as f:
        video_to_phase = json.load(f)

    phase2obj = defaultdict(list)
    obj_cnt = 0
    print("analyze phase transitions")
    for video_id, obj2phase in video_to_phase.items():
        for obj_id, phase in obj2phase.items():
            obj_cnt += 1
            
            raw_obj_full_id = f"{video_id}_{obj_id}"
            match_ = match_prefix_numbers(raw_obj_full_id)

            
            
            if match_ is not None:
                if is_to_remove(match_):
                    obj_full_id = remove_prefix(raw_obj_full_id)
                else:
                    obj_full_id = raw_obj_full_id
            else:
                print("Error: Discover unnumbered obj_id: ", raw_obj_full_id)
            
            phase2obj[phase["phase transition"]].append(obj_full_id)
    return phase2obj

def get_challenge2obj():
    with open("/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/challenge_with_FM_SM_new_78.json") as f:
        video_to_challenge = json.load(f)

    challenge2obj = defaultdict(list)
    obj_cnt = 0
    print("analyze challenge")
    for video_id, obj2challenge in video_to_challenge.items():
        if video_id[:4].isdigit() and is_to_remove(video_id[:4]):
            print("skip wrong label:" , video_id)
            continue

        for obj_id, challenges in obj2challenge.items():


            obj_cnt += 1
            obj_id = obj_id[-1]
            
            raw_obj_full_id = f"{video_id}_{obj_id}"
            match_ = match_prefix_numbers(raw_obj_full_id)



            if match_ is not None:
                if is_to_remove(match_):
                    obj_full_id = remove_prefix(raw_obj_full_id)
                else:
                    obj_full_id = raw_obj_full_id
            else:
                pass
                # print("Discover unnumbered obj_id: ", raw_obj_full_id)

            for challenge in challenges:

            
                challenge2obj[challenge].append(obj_full_id)
    return challenge2obj

def filter_scores(scores, common_keys, excluded_videos):
    new_scores = {}
    for videos_id in scores.keys():
        if (not videos_id.startswith(excluded_videos)) and (videos_id in common_keys):
            new_scores[videos_id] = scores[videos_id]
    return new_scores

def merge_different_challenge_score(J_mean_dict, J_st_dict, J_cc_dict, cnt_dict):
    merge = {
        "OC":["Occlusion: little occlusion", "Occlusion: half occlusion", "Occlusion: total occlusion", "LittleOcclusion","TotalOcclusion","HalfOcclusion","Occlusion"],
        "TO": ["Transparent Object"],

        "CC":["Color Change"],
        "OF":["Out of frame","OutOfFrame","Out Of Frame"],
        "SD":["Similar Distribute in background", "SimilarDisturb","Similar Distribution"],
        "FM":["FastMotion"],
        "SM":["Small Object"]
        
    }

    J_mean_merge = {}
    J_st_merge = {}
    J_cc_merge = {}
    cnt_merge = {}

    for challenge_label , challenge_ls in merge.items():
        J_mean = 0
        J_st = 0
        J_cc = 0
        cnt = 0

        for sub_challenge in challenge_ls:
            if sub_challenge in J_mean_dict:
                J_mean += J_mean_dict[sub_challenge] * cnt_dict[sub_challenge]
                J_st += J_st_dict[sub_challenge] * cnt_dict[sub_challenge]
                J_cc += J_cc_dict[sub_challenge] * cnt_dict[sub_challenge]
                cnt += cnt_dict[sub_challenge]

        if cnt != 0:

            J_mean_merge[challenge_label] = J_mean / cnt
            J_st_merge[challenge_label] = J_st / cnt
            J_cc_merge[challenge_label] = J_cc / cnt
            cnt_merge[challenge_label] = cnt

    return J_mean_merge, J_st_merge, J_cc_merge, cnt_merge




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

    
    print("filtering...")

    common_keys = set(our_video_scores.keys()) & \
                    set(cutie_video_scores.keys()) & \
                    set(sam2_video_scores.keys()) 
                    # set(xmem_video_scores.keys())
                    # set(aot_video_scores.keys()) &\
                    # set(deaot_video_scores.keys()) &\

    hard_keys = get_test_split(meta_root = meta_root, mode="hard" , has_keys = common_keys)
                    
    
    new_our_video_scores = filter_scores(our_video_scores, hard_keys, excluded_videos)
    new_cutie_video_scores = filter_scores(cutie_video_scores, hard_keys, excluded_videos)
    new_sam2_video_scores = filter_scores(sam2_video_scores, hard_keys, excluded_videos)
    new_rmem_video_scores = filter_scores(rmem_video_scores, hard_keys, excluded_videos)
    new_deaot_video_scores = filter_scores(deaot_video_scores, hard_keys, excluded_videos)
    new_xmem_video_scores = filter_scores(xmem_video_scores, hard_keys, excluded_videos)
    print("calculating...")

    video_scores =new_deaot_video_scores

    challenge2obj = get_challenge2obj()


    J_mean = {}
    J_last = {}
    J_cc = {}
    challenge_count = {}

    not_in_videos_scores = []
    print("calculate phase scores")
    for challenge, obj_ids in challenge2obj.items():
        total_J_mean, total_J_last, total_J_cc, count = 0, 0, 0, 0
        for obj_id in obj_ids:
            if obj_id in video_scores:
                total_J_mean += video_scores[obj_id]['J-Mean']
                total_J_last += video_scores[obj_id]['J_last-Mean']
                total_J_cc += video_scores[obj_id]['J_cc-Mean']
                count += 1
            else:
                not_in_videos_scores.append(obj_id)
        if count > 0:
            J_mean[challenge] = total_J_mean / count
            J_last[challenge] = total_J_last / count
            J_cc[challenge] = total_J_cc / count
            challenge_count[challenge] = count

    


    J_mean_merge , J_st_merge, J_cc_merge , cnt_merge = merge_different_challenge_score(J_mean_dict = J_mean, J_st_dict = J_last, J_cc_dict = J_cc, cnt_dict = challenge_count )
    print(" -------------------")
    
    print("Result:")
    # print("J_mean:" , J_mean_merge)

    print("J_last:" , J_st_merge)
    # print("J_cc:" , J_cc_merge)
    print("cnt:",  cnt_merge)
    # sorted_keys = sorted(J_mean, key=J_mean.get, reverse=True)



    # subsets = {
    #     "Intra-Phase (solid)": ["flow", "paint","splash", "mix",  "drip"],
    #     "Intra-Phase (Liquid)": ["separate",  "twist",  "break", "stretch", "split", "merge", "crush"],
    #     "Intra-Phase (Aerosol//Gas)": ["diffusion"],
    #     "Cross-Phase": ['solidify', "melt" , "deposition", "vaporize", "crystallize" , "sublimate", "dissolve", "compress" , "flow out" , "soften"],
    # }
    # colors = {
    #     "Intra-Phase (solid)": ["navy", "blue", "lightblue"],
    #     "Intra-Phase (Liquid)": ["darkgreen", "green", "lightgreen"],
    #     "Intra-Phase (Aerosol//Gas)": ["#FF8C00", "orange", "#FED8B1"],
    #     "Cross-Phase": ["darkred", "red", "lightcoral"],
    # }

    # abbr = {
    #     "Intra-Phase (solid)": "IS",
    #     "Intra-Phase (Liquid)": "IL",
    #     "Intra-Phase (Aerosol//Gas)": "IG",
    #     "Cross-Phase": "CP"
    # }

    # print("these objs not in video_scores", not_in_videos_scores)

    # print("plotting")


    # # 准备数据
    
    # width = 0.2  # 柱的宽度
    # fig, ax = plt.subplots(figsize=(12, 6))
    # offset = 0
    # xticks = []
    # xticks_labels = []
    # for i, (subset_name, keys) in enumerate(subsets.items()):
    #     sorted_subset_keys  = sorted(keys, key=lambda k: J_mean[k], reverse=True)
    #     print(sorted_subset_keys)
    #     x = np.arange(len(sorted_subset_keys))  # 横轴的位置
    #     bars1 = ax.bar(x+offset, [J_mean[k] for k in sorted_subset_keys], width, 
    #                    label=f'J_mean({abbr[subset_name]})', color=colors[subset_name][0])
    #     bars2 = ax.bar([p + width + offset for p in x], [J_last[k] for k in sorted_subset_keys], width, 
    #                    label=f'J_last({abbr[subset_name]})', color=colors[subset_name][1])
    #     bars3 = ax.bar([p + 2 * width + offset for p in x], [J_cc[k] for k in sorted_subset_keys], width, 
    #                    label=f'J_cc({abbr[subset_name]})', color=colors[subset_name][2])
    #     # 添加标签
        
        
    #     xticks += [p + 0.5 * width + offset for p in x]

    #     xticks_labels += [f"{key}" for key in sorted_subset_keys]
    #     offset += len(sorted_subset_keys)

    #     # if i < len(subsets) - 1:
    #     #     ax.axvline(x=offset - 0.5, color='black', linestyle='--', linewidth=2, )
        
    #     text_x = (offset + offset - len(sorted_subset_keys)) / 2
    #     ax.text(text_x, 0.9, abbr[subset_name], ha='center', va='bottom')
    
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels,  rotation=45, fontsize=10)  # 在标签后加上括号注明数量
    # # 添加图例
    # ax.legend()
    
    # # 添加标题和标签
    # ax.set_title("Performance of Different Phase Transitions", y=1.05)
    # ax.set_xlabel('Phase transitions')
    # ax.set_ylabel('Scores')

    # plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    # plt.savefig("phase_transition_score.pdf")

    # print("finished!")