
CFBI_PATH=r"./CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations/per-sequence_results-val.csv"
AOT_PATH=r"./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}/per-sequence_results-val.csv"
XMEM_PATH=r"./XMem/output/roves_week${week_num}/per-sequence_results-val.csv"
CUTIE_PATH=r"./Cutie/cutie_output/roves_week${week_num}/roves-val/Annotations/per-sequence_results-val.csv"
DEAOT_PATH=r"./RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}/per-sequence_results-val.csv"
SAM2_PATH=r"./segment-anything-2/outputs/sam2_in_roves_week_${week_num}/per-sequence_results-val.csv"

DATASET_ROOT_PATH = r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/CFBI/datasets/ROVES_summary/ROVES_week_${week_num}"
ALL_OUTPUT_PATH = [CFBI_PATH, AOT_PATH, XMEM_PATH, CUTIE_PATH , DEAOT_PATH, SAM2_PATH]



MODEL2Idx= {"CFBI":0 , "AOT": 1, "XMEM": 2, "Cutie": 3, "DeAOT": 4, "SAM2": 5}



import pandas as pd
import os
import json

def replace_week_num(path, week_num):
    return path.replace("${week_num}", str(week_num))

def exclude_sublist(str_case, exclude_list):
    # for s in exclude_list:
    #     print("str_case:",  str_case , " in " , s , ":" ,s in str_case )

    return any(s in str_case for s in exclude_list)

def calculate_subset_sum(df, subset):

    # 过滤DataFrame，只保留subset中的行
    filtered_df = df[df['Sequence'].isin(subset)]
    
    # 计算J-Mean和J_last-Mean的平均值
    j_mean_average = filtered_df['J-Mean'].sum()
    j_last_mean_average = filtered_df['J_last-Mean'].sum()
    j_cc_mean_average = filtered_df['J_cc-Mean'].sum()
    
    return j_mean_average, j_last_mean_average, j_cc_mean_average, len(filtered_df)

def get_all_challenge(model, all_num= 5, exclude_list = []):

    # print("exclu:" ,exclude_list)

    model_res_path = ALL_OUTPUT_PATH[MODEL2Idx[model]]

    challenge_res = {}

    for week_id in range(all_num + 1):


        dir_path =  os.path.join( replace_week_num( DATASET_ROOT_PATH, week_id), "ImageSets" )

        for filename in os.listdir(dir_path):
            if filename == "val.txt":
                continue
            # 构建文件的完整路径
            file_path = os.path.join(dir_path, filename)

            # 确保是文件而不是文件夹
            if os.path.isfile(file_path):

                if filename not in challenge_res.keys():
                    challenge_res[filename] = {"Jcc_sum": 0 , "Jst_sum": 0 , "J_sum": 0, "cnt": 0}

                df = pd.read_csv(replace_week_num(model_res_path , week_id))

                df = df[~df['Sequence'].apply(lambda x: exclude_sublist(x, exclude_list))]


  
                # 读取文件中的subset，这里假设每行一个Sequence名
                with open(file_path, 'r') as f:
                    subset = [line.strip().replace('_id', '') for line in f.readlines()]
                # 计算该subset的J-Mean和J_last-Mean的平均值

                j_mean_sum, j_last_sum, j_cc_sum ,  cnt = calculate_subset_sum(df, subset)

                challenge_res[filename]["Jcc_sum"] += j_cc_sum
                challenge_res[filename]["Jst_sum"] += j_last_sum
                challenge_res[filename]["J_sum"] += j_mean_sum
                challenge_res[filename]["cnt"] += cnt

        # print("week ", week_id , ":"  , challenge_res )


    output_res = []

    for key in challenge_res:
        output_res.append(
           { "subset":  key.replace(".txt", ""),
            'j_mean_average': challenge_res[key]["J_sum"] / challenge_res[key]["cnt"],
            'j_last_mean_average': challenge_res[key]["Jst_sum"] / challenge_res[key]["cnt"],
            "J_cc_mean_average":  challenge_res[key]["Jcc_sum"] / challenge_res[key]["cnt"],
            "case cnt": challenge_res[key]["cnt"],
           }
        )

    result_df = pd.DataFrame(output_res)
    result_df.to_csv(os.path.join("./evaluation/all_result", f"all_challenge_in_{model}_before_week_{all_num}"), index=False)
    print("OUTPUT:" , os.path.join("./evaluation/all_result", f"all_challenge_in_{model}_before_week_{all_num}"))




def get_all_score(model_res_path, all_num= 5, exclude_list = []):
    J_sum = 0
    Jcc_sum = 0
    Jst_sum = 0
    cnt = 0
    # print("exclu:" ,exclude_list)

    for i in range(all_num + 1):
        res_path = replace_week_num(model_res_path , i)
        res_df = pd.read_csv(res_path)
        
        res_df = res_df[~res_df['Sequence'].apply(lambda x: exclude_sublist(x, exclude_list))]
        # print("week " , i, " exclude flag:" , res_df['Sequence'] , " :  " ,  ~res_df['Sequence'].apply(lambda x: exclude_sublist(x, exclude_list)))

        if 'Sequence' not in res_df.columns or 'J-Mean' not in res_df.columns or 'J_last-Mean' not in res_df.columns or "J_cc-Mean" not in res_df.columns:
            print("wrong CSV file path:",  res_path)
            raise ValueError("CSV file must contain 'Sequence', 'J-Mean', and 'J_last-Mean' , and 'J_cc-Mean' columns.")
        
        J_sum += res_df["J-Mean"].sum()
        Jst_sum += res_df["J_last-Mean"].sum()
        Jcc_sum += res_df["J_cc-Mean"].sum()
        # print(res_path)

        cnt +=  len(res_df)
        # print(i , " week : ", cnt)

    return J_sum / cnt , Jcc_sum / cnt , Jst_sum / cnt



def get_all_model_challenge(week_num=5, exclude_json = "./evaluation/meta/exclude_case.json"):
    exclude_list = json.load(open(exclude_json, "r"))

    for model in MODEL2Idx.keys():
        get_all_challenge(model, week_num, exclude_list)

    
def get_all_model_score(week_num=5, exclude_json = "./evaluation/meta/exclude_case.json"):
    result = []

    exclude_list = json.load(open(exclude_json, "r"))

    for model in MODEL2Idx.keys():
        idx = MODEL2Idx[model]

        J_mean, Jcc_mean, Jst_mean = get_all_score(ALL_OUTPUT_PATH[idx], all_num= week_num, exclude_list = exclude_list)

        result.append({
            "Model" : model,
            "J_mean_average" : J_mean,
            "J_last_mean_average": Jst_mean,
            "J_cc_mean_average": Jcc_mean,

        })

    

    result_df = pd.DataFrame(result)
    result_df.to_csv(os.path.join("./evaluation/all_result", f"all_score_before_week_{week_num}"), index=False)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--week_num', type=int)

    args, _ = parser.parse_known_args()

    # print(get_all_score(ALL_OUTPUT_PATH[0], 2))
    # get_all_model_score(5)
    # get_all_challenge("CFBI", 5)
    # get_all_model_challenge(week_num=args.week_num)
    get_all_model_score(week_num=args.week_num)
    















    



