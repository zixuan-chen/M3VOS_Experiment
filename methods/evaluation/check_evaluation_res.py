"""
目的:检查每个方法的evaluation产生的csv文件是否有效涵盖所有的结果
"""
import os
import csv
def get_seq_folder(test_path):
    seq_folders = os.listdir(test_path)
    seq_ls = [seq for seq in seq_folders if seq.split("_")[0].isdigit() and os.path.isdir(os.path.join(test_path,seq)) ] 

    return seq_ls

def read_first_column(csv_file_path):
    first_column = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  
                first_column.append(row[0])
    return first_column


def get_seq_ls_in_csv(test_path):
    csv_path = os.path.join(test_path, "per-sequence_results-val.csv")
    seq_ls = read_first_column(csv_file_path=csv_path)
    seq_ls = [seq[:-2] for seq in seq_ls if seq.split("_")[0].isdigit()  ] 
    return seq_ls
    


def test_one_folder(test_path):
    seq_ls = get_seq_folder(test_path=test_path)
    csv_seq_ls = get_seq_ls_in_csv(test_path=test_path)

    undone_seq_ls = []
    for seq in seq_ls:
        if seq not in csv_seq_ls:
            undone_seq_ls.append(seq)

    return undone_seq_ls

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

# DeAOT + RMEM
# results_path="./RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"

# AOT_different_fps
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_different_fps_roves_in_aot_week_0_fps_${fps}"

# Xmem (Temp)
# results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output/roves_week${week_num}" 

# Cutie (Temp)
# results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week${week_num}/roves-val/Annotations"

# myVOS
# results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output/roves_week${week_num}_ckpt176300/roves-val/Annotations"

# SAM2
# results_path="./segment-anything-2/outputs/sam2_in_roves_week_${week_num}"
def test_model_res(model_name ,model_res_root_dict, week_ls = ['8','9','10','11','12','13']):
    model_res_root = model_res_root_dict[model_name]
    for week_num in week_ls:
        if model_name == "AOT":
            test_folder = os.path.join(model_res_root,f"test_roves_in_aot_week_{week_num}")
        elif model_name == "DeAOT+RMEM":
            test_folder = os.path.join(model_res_root, f"test_roves_week_in_deaotRmem_{week_num}")

        elif model_name == "Xmem":
            test_folder = os.path.join(model_res_root, f"roves_week{week_num}")

        elif model_name == "Cutie":
            test_folder = os.path.join(model_res_root, f"roves_week{week_num}","roves-val","Annotations")

        elif model_name == "SAM2":
            test_folder = os.path.join(model_res_root, f"sam2_in_roves_week_{week_num}")

        else:
            raise NotImplemented

        assert os.path.exists(test_folder)
        
        undone_seqs = test_one_folder(test_path=test_folder)
        if len(undone_seqs) != 0:
            print(f"{model_name} has un evaluate seqs in week {week_num}: " ,  undone_seqs)



if __name__ == "__main__":
    res_root_dict = {
        "AOT": "./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves" ,
        "DeAOT+RMEM":"./RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves" , 
        "Xmem":  "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output",
        "Cutie" : "/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output" ,
        "SAM2": "./segment-anything-2/outputs"  
    }

    undone_seqs = test_one_folder("./segment-anything-2/outputs/sam2_in_roves_week_8")

    test_model_res( "Cutie", res_root_dict, week_ls = ['8','9','10','11','12',"13"])


