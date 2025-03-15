import pandas as pd
import numpy as np
import os
from plot_phase_transition  import *

def get_avg_score_for_videos(weight= {"颜色":1, "形状":1, "位置":1, "状态":1, "视频质量":1, "相似干扰":1, "遮挡":1, "长视频":1, "透明物体":1}):
    folder_path = "M_cube_VOS_human_score"
    df_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)

            df_list.append(df)


    zixuan_df = pd.read_csv("M_cube_VOS_human_score/M cube VOS 场景统计-子轩难度评价.csv")
    del zixuan_df["小物体"]
    cleaned_zixuan_df = zixuan_df.dropna(axis=0)

    common_rows = cleaned_zixuan_df.index

    common_cols = list(set.intersection(*[set(df.columns) for df in df_list]))

    video_names = df_list[0]["Video Name "][common_rows]
    
    common_cols.remove("Video Name ")

    common_df_list = [df.loc[common_rows, common_cols] for df in df_list]

    # df = common_df_list[0]
    # duplicated_rows = df[df.duplicated('Video ID', keep=False)]
    # print(duplicated_rows)
    average_df = pd.DataFrame(None, index=common_rows, columns=common_cols)
    for index in common_rows:
        for col_name in common_cols:
            average_df[col_name][index] = np.mean([df[col_name][index] for df in common_df_list])



    extra_cols_zixuan = list(set(zixuan_df.columns) - set(common_cols))

    extra_df_zixuan = cleaned_zixuan_df.loc[:, extra_cols_zixuan]

    if not isinstance(average_df, pd.DataFrame):
        average_df = average_df.to_frame().T

    result_df = pd.concat([average_df, extra_df_zixuan], axis = 1)

    valid_rows = result_df.index
    valid_columns = ["颜色", "形状", "位置", "状态", "视频质量", "相似干扰", "遮挡", "长视频", "透明物体"]
    for metric in (valid_columns):
        result_df['row_sum'] =  weight[metric] *  result_df[metric]

    res =  pd.concat([result_df['row_sum'], video_names], axis = 1)
    
    return res

def starts_with_digit_underscore(s):
    return bool(re.match(r'^\d+_', s))

def get_avg_score_for_objs(weight= {"颜色":1, "形状":1, "位置":1, "状态":1, "视频质量":1, "相似干扰":1, "遮挡":1, "长视频":1, "透明物体":1}):

    all_valid_objs = []
    
    all_valid_objs_scores = {}
    with open("all_valid_obj.txt") as f:
        for line in f:
            all_valid_objs.append(line.strip())
    matched_objs = []
    avg_score_for_videos = get_avg_score_for_videos(weight = weight)

    for index, row in avg_score_for_videos.iterrows():
        video_name = row["Video Name "]
        video_idx = int(index)
        num_str = "{:04d}".format(video_idx)
        video_name = video_name.split()[0]
        if is_to_remove(num_str) and starts_with_digit_underscore(video_name):
            video_name = remove_prefix(video_name)
        elif (not is_to_remove(num_str)) and (not starts_with_digit_underscore(video_name)):
            video_name = num_str + "_" + video_name
        
        for obj in all_valid_objs:
            if obj.startswith(video_name):
                all_valid_objs_scores[obj] = row['row_sum']
                matched_objs.append(obj)

    print ("unmatched:", set(all_valid_objs) - set(matched_objs))
    return all_valid_objs_scores


    
def get_hard_mid_easy():

    all_valid_objs_scores = get_avg_score_for_objs()

    df = pd.DataFrame(list(all_valid_objs_scores.items()), columns=['Key', 'Value'])

    # 根据Value对DataFrame进行降序排序
    df_sorted = df.sort_values(by='Value', ascending=False)

    
    num_keys = len(df_sorted)
    # 计算前33%的元素数量
    num_top_hard = int(num_keys * 0.33)
    # 提取前33%最大的value对应的key
    top_hard_objs = set(df_sorted.head(num_top_hard)['Key'])

    with open("all_hard_objs.txt", 'w') as f:
        for obj in top_hard_objs:
            f.write(obj + "\n")

    num_top_mid = int(num_keys * 0.66)
    # 提取前66%最大的value对应的key
    top_mid_objs = set(df_sorted.head(num_top_mid)['Key'])
    with open("all_mid_objs.txt", 'w') as f:
        for obj in top_mid_objs:
            f.write(obj + "\n")
    
    all_objs_easy = set(df_sorted['Key'].tolist())
    with open("all_easy_objs.txt", 'w') as f:
        for obj in all_objs_easy:
            f.write(obj + "\n")
    
if __name__ == "__main__":
    all_valid_objs_scores = get_avg_score_for_objs()
    print((all_valid_objs_scores))
    # get_hard_mid_easy()

    