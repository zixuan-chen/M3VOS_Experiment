from get_metric_distribution import *
from plot_phase_transition import *
from collections import defaultdict, Counter

subsets = {
        "Intra-Phase (solid)": ["flow", "paint","splash", "mix",  "drip"],
        "Intra-Phase (Liquid)": ["separate",  "twist",  "break", "stretch", "split", "merge", "crush"],
        "Intra-Phase (Aerosol//Gas)": ["diffusion"],
        "Cross-Phase": ['solidify', "melt" , "deposition", "vaporize", "crystallize" , "sublimate", "dissolve", "compress" , "flow out" , "soften"],
    }



all_phase_label = "/home/bingxing2/home/scx8ah2/dataset/ROVES_meta/all_phase_label.json"

with open(all_phase_label, "r", encoding="utf-8") as file:
    phase_label = json.load(file)

phase2obj = defaultdict(list)

obj_num = 0

phase2obj = get_phase2obj()

print("object number from json", obj_num)

subset2objs = defaultdict(list)

for subset_name, phase_trans_list in subsets.items():
    for phase in phase_trans_list:
        subset2objs[subset_name].extend(phase2obj[phase])

def calculate_averages(sub_keys, scores):
    J_mean_sum = 0
    J_last_sum = 0
    J_cc_sum = 0
    count = 0
    
    for key in sub_keys:
        if key in scores:
            J_mean_sum += scores[key]["J-Mean"]
            J_last_sum += scores[key]['J_last-Mean']
            J_cc_sum += scores[key]["J_cc-Mean"]
            count += 1
    
    J_mean_avg = J_mean_sum / count if count else 0
    J_last_avg = J_last_sum / count if count else 0
    J_cc_avg = J_cc_sum / count if count else 0

    return {"J-Mean":J_mean_avg, 
            "J_last-Mean": J_last_avg,
            "J_cc-Mean": J_cc_avg
            }

def find_duplicates(lst):
    counter = Counter(lst)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    # 打印重复的元素及其出现次数
    print(duplicates)

def get_all_phase_metric(video_scores:dict):
    
    all_valid_keys = []
    subset2keys = defaultdict(list)
    excluded_keys = list(video_scores.keys())
    excluded_objs = []
    for subset_name, objs in subset2objs.items():
        for obj in objs:
            unmatched = True
            for key in excluded_keys:
                if obj in key:
                    subset2keys[subset_name].append(key)
                    excluded_keys.remove(key)
                    unmatched = False
                    break
            if unmatched:
                excluded_objs.append(obj)
        all_valid_keys.extend(subset2keys[subset_name])

        print(f"{subset_name} 数量: {len(subset2keys[subset_name])}")
        print(f"{subset_name}: {calculate_averages(subset2keys[subset_name], video_scores)}")
    
    # with open('all_valid_objs.txt', 'w', encoding='utf-8') as file:
    #     # 遍历列表中的每个元素
    #     for item in all_valid_keys:
    #         # 将元素写入文件，每个元素后跟一个换行符
    #         file.write(item + '\n')

    # print("exluded objects: ", excluded_objs)
    # print("exluded keys: ", excluded_keys)


if __name__ == "__main__":
    pattern = r'^roves_week(\d+)_mega_v4_72800$'
    print("loading scores...")
    our_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output", pattern=pattern)
    cutie_video_scores = get_distribution(res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output", pattern=r'^roves_week(\d+)$')
    
    sam2_video_scores = get_distribution(dataset="sam2",
                                         res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/segment-anything-2/outputs", 
                                         pattern=r"sam2_in_roves_week_(\d+)$")
    aot_video_scores = get_distribution(dataset="aot",
                                        res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves",
                                        pattern=r"test_roves_in_aot_week_(\d+)$")

    rmem_video_scores = get_distribution(dataset="rmem",
                                        res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves",
                                        pattern=r"test_rmem_wo_pte_roves_in_aot_week_(\d+)$")

    deaot_video_scores = get_distribution(dataset="deaot",
                                          res_dir="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves",
                                          pattern=r"roves_val_low_res_infer_roves_in_deaot_swin_week_(\d+)_SwinB_DeAOTL_PRE_ckpt_unknown$")
    xmem_video_scores = get_distribution(dataset="xmem",
                                         res_dir="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output",
                                         pattern=r'^roves_week(\d+)$')
    # plot_metrics_distribution(video_scores,"./myVOS_metrics.png")
    excluded_videos = ("mv_cola_1", "mv_cola_2", "mv_cola_3", "mv_energy_1", "mv_energy_2", "mv_lime_1", 
                        "mv_lime_2", "mv_mark_1", "mv_cup_4", "mv_cup_5", "mv_redpen_2", "mv_redpen_3", "mv_brocolli_1",
                        "mv_brocolli_8", "mv_egg_2", "mv_egg_4", "mv_pepper_2", "mv_pepper_3", "mv_green_pepper_5", "0209_break_glass_5",
                        "tear_green_pepper_1", "break_egg_3","cut_green_pepper_4","cut&break_brocolli_5", "tear_green_pepper_6", "pinch_plush_1_1")
    
    print("filtering...")
    
   
    all_valid_objs = []
    with open("all_easy_objs.txt") as f:
        for line in f:
            all_valid_objs.append(line.strip())
    
    matched_objs = []
    common_keys = set(our_video_scores.keys()) & \
                    set(cutie_video_scores.keys()) & \
                    set(sam2_video_scores.keys()) & \
                    set(xmem_video_scores.keys()) & \
                    set(all_valid_objs)
                    # set(aot_video_scores.keys()) &\
                    # set(deaot_video_scores.keys()) &\
                    
    
    
    print("---------------------- Our score -----------------------------")
    new_our_video_scores = filter_scores(our_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_our_video_scores))
    get_all_phase_metric(new_our_video_scores)
    
    print("---------------------- Cutie score -----------------------------")
    new_cutie_video_scores = filter_scores(cutie_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_cutie_video_scores))
    get_all_phase_metric(new_cutie_video_scores)
    
    print("---------------------- SAM2 score -----------------------------")
    new_sam2_video_scores = filter_scores(sam2_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_sam2_video_scores))
    get_all_phase_metric(new_sam2_video_scores)

    print("---------------------- XMem score -----------------------------")
    new_xmem_video_scores = filter_scores(xmem_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_xmem_video_scores))
    get_all_phase_metric(new_xmem_video_scores)


    print("---------------------- RMem score -----------------------------")
    new_rmem_video_scores = filter_scores(rmem_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_rmem_video_scores))
    get_all_phase_metric(new_rmem_video_scores)

    print("---------------------- Deaot score -----------------------------")
    new_deaot_video_scores = filter_scores(deaot_video_scores, common_keys, excluded_videos)
    print("总数: ", len(new_deaot_video_scores))
    get_all_phase_metric(new_deaot_video_scores)