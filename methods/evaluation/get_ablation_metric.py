from get_all_phase_metric import *
from get_metric_distribution import get_test_txt_split
if __name__ == "__main__":
    pattern = r'^roves_week(\d+)_mega_v4_72800$'
    res_dir = "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output/"
    print("loading scores...")
    SL_30_RI_30_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_SL_30_RI_30$')
    
    SL_10_RI_30_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_SL_10_RI_30$')

    SL_30_RI_10_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_SL_30_RI_10$')

    SL_30_RI_60_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_SL_30_RI_60$')

    SL_60_RI_30_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_SL_60_RI_30$')

    no_booster_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_without_booster$')

    no_fuser_scores = get_distribution(res_dir=res_dir, pattern=r'^roves_week(\d+)_without_fuser$')

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

    common_keys = set(SL_30_RI_30_scores.keys()) & \
                    set(SL_10_RI_30_scores.keys()) & \
                    set(SL_30_RI_10_scores.keys()) & \
                    set(SL_30_RI_60_scores.keys()) & \
                    set(SL_60_RI_30_scores.keys()) & \
                    set(no_booster_scores.keys()) & \
                    set(no_fuser_scores.keys()) &\
                    set(all_valid_objs)
    meta_root = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta"
    easy_keys = get_test_txt_split(meta_root, mode= "easy", has_keys = common_keys)
                    # set(aot_video_scores.keys()) &\
                    # set(deaot_video_scores.keys()) &\

    print("---------------------- SL_30_RI_30 score -----------------------------")
    new_SL_30_RI_30_scores = filter_scores(SL_30_RI_30_scores, common_keys, excluded_videos)
    print("总数: ", len(new_SL_30_RI_30_scores))
    print("score:" , calculate_average_scores(new_SL_30_RI_30_scores))

    print("---------------------- SL_10_RI_30 score -----------------------------")
    new_SL_10_RI_30_scores = filter_scores(SL_10_RI_30_scores, common_keys, excluded_videos)
    print("总数: ", len(new_SL_10_RI_30_scores))
    print("score:" , calculate_average_scores(new_SL_10_RI_30_scores))

    print("---------------------- SL_30_RI_10 score -----------------------------")
    new_SL_30_RI_10_scores = filter_scores(SL_30_RI_10_scores, common_keys, excluded_videos)
    print("总数: ", len(new_SL_30_RI_10_scores))
    print("score:" , calculate_average_scores(new_SL_30_RI_10_scores))

    print("---------------------- SL_30_RI_60 score -----------------------------")
    new_SL_30_RI_60_scores = filter_scores(SL_30_RI_60_scores, common_keys, excluded_videos)
    print("总数: ", len(new_SL_30_RI_60_scores))
    print("score:" , calculate_average_scores(new_SL_30_RI_60_scores))

    print("---------------------- SL_60_RI_30 score -----------------------------")
    new_SL_60_RI_30_scores = filter_scores(SL_60_RI_30_scores, common_keys, excluded_videos)
    print("总数: ", len(new_SL_60_RI_30_scores))
    print("score:" , calculate_average_scores(new_SL_60_RI_30_scores))

    print("---------------------- no_booster score -----------------------------")
    new_no_booster_scores = filter_scores(no_booster_scores, common_keys, excluded_videos)
    print("总数: ", len(new_no_booster_scores))
    print("score:" , calculate_average_scores(new_no_booster_scores))

    print("---------------------- no_fuser score -----------------------------")
    new_no_fuser_scores = filter_scores(no_fuser_scores, common_keys, excluded_videos)
    print("总数: ", len(new_no_fuser_scores))
    print("score:" , calculate_average_scores(new_no_fuser_scores))