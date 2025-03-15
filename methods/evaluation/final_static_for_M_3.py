import json
import re
import os 
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import csv

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
                                 
excluded_videos = ("mv_cola_1", "mv_cola_2", "mv_cola_3", "mv_energy_1", "mv_energy_2", "mv_lime_1", 
                        "mv_lime_2", "mv_mark_1", "mv_cup_4", "mv_cup_5", "mv_redpen_2", "mv_redpen_3", "mv_brocolli_1",
                        "mv_brocolli_8", "mv_egg_2", "mv_egg_4", "mv_pepper_2", "mv_pepper_3", "mv_green_pepper_5", "0209_break_glass_5",
                        "tear_green_pepper_1", "break_egg_3","cut_green_pepper_4","cut&break_brocolli_5", "tear_green_pepper_6", "pinch_plush_1_1")

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
                print("Error: Discover unnumbered obj_id in json: ", raw_obj_full_id)
            
            phase2obj[phase["phase transition"]].append(obj_full_id)
    return phase2obj


if __name__ == "__main__":

    
    phase2obj = get_phase2obj()

    pattern = r'^roves_week(\d+)_mega_v4_72800$'

    video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=pattern)
    # video_scores = {remove_prefix(key): value for key, value in video_scores.items()}

    J_mean = {}
    J_last = {}
    J_cc = {}
    phase_count = {}

    not_in_videos_scores = []
    total_cnt = 0
    print("calculate phase scores")

    # video_cnt = 0
    video_ls = []

    

    with open("from_phase&videos_score.txt", 'w') as f:
        for phase_transition, obj_ids in phase2obj.items():
            total_J_mean, total_J_last, total_J_cc, count = 0, 0, 0, 0
            for obj_id in obj_ids:
                if obj_id in video_scores and not obj_id.startswith(excluded_videos):
                    total_J_mean += video_scores[obj_id]['J-Mean']
                    total_J_last += video_scores[obj_id]['J_last-Mean']
                    total_J_cc += video_scores[obj_id]['J_cc-Mean']
                    count += 1
                    total_cnt += 1
                    f.write(obj_id + '\n')
                    video_ls.append(obj_id[:-2])
                else:
                    not_in_videos_scores.append(obj_id)
                
            if count > 0:
                J_mean[phase_transition] = total_J_mean / count
                J_last[phase_transition] = total_J_last / count
                J_cc[phase_transition] = total_J_cc / count
                phase_count[phase_transition] = count
            
        print(video_ls)

        print("video cnt:" , len(set(video_ls)))
        print("total count = ", total_cnt)

    json.dump(list(set(video_ls)) , open("all_video_name.json", "w")  )
    print("all_video_name.json")




    # sorted_keys = sorted(J_mean, key=J_mean.get, reverse=True)
    # subsets = {
    #     "Intra-Phase (solid)": ["separate", "twist", "break", "stretch", "split", "merge", "crush"]  ,
    #     "Intra-Phase (Liquid)": ["flow", "paint", "splash", "mix", "drip"],
    #     "Intra-Phase (Aerosol//Gas)": ["diffusion"],
    #     "Cross-Phase": ['solidify', "melt", "deposition", "vaporize", "crystallize", "sublimate", "dissolve", "compress", "flow out", "soften"],
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
    # print("J_mean = ", J_mean)
    # for i, (subset_name, keys) in enumerate(subsets.items()):
    #     sorted_subset_keys = sorted(keys, key=lambda k: J_mean[k], reverse=True)
    #     print(sorted_subset_keys)
    #     x = np.arange(len(sorted_subset_keys))  # 横轴的位置
    #     bars1 = ax.bar(x + offset, [J_mean[k] for k in sorted_subset_keys], width,
    #                 label=f'J_mean({abbr[subset_name]})', color=colors[subset_name][0])
    #     bars2 = ax.bar([p + width + offset for p in x], [J_last[k] for k in sorted_subset_keys], width,
    #                 label=f'J_last({abbr[subset_name]})', color=colors[subset_name][1])
    #     bars3 = ax.bar([p + 2 * width + offset for p in x], [J_cc[k] for k in sorted_subset_keys], width,
    #                 label=f'J_cc({abbr[subset_name]})', color=colors[subset_name][2])
    #     # 添加标签

    #     xticks += [p + 0.5 * width + offset for p in x]
    #     print("x ticks:", subset_name)

    #     xticks_labels += [f"{key}" for key in sorted_subset_keys]
    #     offset += len(sorted_subset_keys)

    #     text_x = (offset + offset - len(sorted_subset_keys)) / 2
    #     ax.text(text_x, 0.9, abbr[subset_name], ha='center', va='bottom', fontsize=15)

    # print("all:" , xticks_labels)

    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels, rotation=45, fontsize=15)  # 在标签后加上括号注明数量

    # # 添加图例
    # ax.legend(fontsize=12)

    # # 设置纵轴刻度标签的字体大小
    # ax.tick_params(axis='y', labelsize=15)

    # # 添加标题和标签
    # # ax.set_title("Performance of Different Phase Transitions", y=1.05, fontsize=15)
    # ax.set_xlabel('Phase transitions', fontsize=15)
    # ax.set_ylabel('Scores', fontsize=15)

    # plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    # plt.savefig("phase_transition_score.pdf")
    # print("phase_transition_score.png")

    # print("finished!")