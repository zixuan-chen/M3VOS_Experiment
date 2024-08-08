import pandas as pd
import numpy as np
import os
import sys


def pause_or_exit():
    print("程序已暂停，按回车键继续，按'q'键退出程序。")
    try:
        # 等待用户输入，如果输入的是'q'，则退出循环和程序
        if input().lower() == 'q':
            print("程序已退出。")
            sys.exit()
    except KeyboardInterrupt:
        # 如果用户使用Ctrl+C中断，也允许退出
        print("程序已退出。")
        sys.exit()

def merge_dataframes(*dataframes):
    # 首先，将所有DataFrame转换为列表，如果它们不是列表的话
    if not isinstance(dataframes, list):
        dataframes = list(dataframes)
    
    # 检查是否至少有一个DataFrame
    if not dataframes:
        raise ValueError("No dataframes provided for merging.")
    
    # 将所有DataFrame按'Sequence'列排序，确保每个'Sequence'的最后一行是来自最后一个DataFrame
    sorted_dataframes = [df.sort_values('Sequence') for df in dataframes]
    
    # 合并所有DataFrame，ignore_index=True会重新索引
    combined_df = pd.concat(sorted_dataframes, ignore_index=True)
    
    # 根据'Sequence'列去重，保留最后一个DataFrame中的行
    unique_df = combined_df.drop_duplicates(subset='Sequence', keep='last')
    
    return unique_df


def calculate_subset_average(df, subset):

    # 过滤DataFrame，只保留subset中的行
    filtered_df = df[df['Sequence'].isin(subset)]
    
    # 计算J-Mean和J_last-Mean的平均值
    j_mean_average = filtered_df['J-Mean'].mean()
    j_last_mean_average = filtered_df['J_last-Mean'].mean()
    
    return j_mean_average, j_last_mean_average



def calculate_averages_from_dir(dir_path, csv_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 检查输入的DataFrame是否包含必要的列
    if 'Sequence' not in df.columns or 'J-Mean' not in df.columns or 'J_last-Mean' not in df.columns:
        raise ValueError("CSV file must contain 'Sequence', 'J-Mean', and 'J_last-Mean' columns.")
    
    # 初始化一个空列表来存储结果
    results = []
    
    # 遍历文件夹中的每个文件
    for filename in os.listdir(dir_path):
        # 构建文件的完整路径
        file_path = os.path.join(dir_path, filename)
        
        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            # 读取文件中的subset，这里假设每行一个Sequence名
            with open(file_path, 'r') as f:
                subset = [line.strip().replace('_id', '') for line in f.readlines()]
            # 计算该subset的J-Mean和J_last-Mean的平均值
            j_mean_average, j_last_mean_average = calculate_subset_average(df, subset)
            
            # 将结果添加到列表中
            results.append({
                'subset': filename,
                'j_mean_average': j_mean_average,
                'j_last_mean_average': j_last_mean_average
            })
            
            
    # 将结果列表转换为DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

if __name__ == "__main__":
    week1_5_filename = "/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/aot_week_1_5/per-sequence_results-val.csv"
    subset_dir = "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/tmp"
    print(calculate_averages_from_dir(subset_dir, week1_5_filename))
