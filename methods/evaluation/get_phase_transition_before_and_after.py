import json
import re

def get_phase_dict(json_root="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/all_phase_label.json"):
    phase_dict = json.load(open(json_root,"rb"))
    return phase_dict

def get_all_video(json_root="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_meta/all_video_final.json"):
    all_video = json.load(open(json_root, "r"))
    return all_video

def get_all_valid_obj(txt_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/evaluation/all_valid_obj.txt"):
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip()) >  0]
    return lines


import json
from collections import defaultdict

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

def is_substr_in_ls(substr, str_ls):
    target_key = None
    for str_key in str_ls:
        if substr in str_key:
            target_key = str_key
    return target_key
    # print(cnted_key )- set() )
    
def find_duplicates(input_list):
    # 使用字典来存储每个元素的计数
    element_counts = defaultdict(int)
    
    # 遍历列表，统计每个元素的出现次数
    for element in input_list:
        element_counts[element] += 1
    
    # 找到出现次数大于1的元素
    duplicates = [element for element, count in element_counts.items() if count > 1]
    cnt =  [count for element, count in element_counts.items() if count > 1]
    
    return duplicates,cnt
def before_after_cnt():

    data = get_phase_dict()
    all_video = get_all_video()
    all_obj = get_all_valid_obj()
    cnt = 0
    cnted_key = []

    

    # 初始化一个字典来存储计数
    state_transition_counts = defaultdict(int)

    for video_obj_id in all_obj:
        # print(video_obj_id)

        target_key =  is_substr_in_ls(video_obj_id[:-2] ,list(data.keys()))
        obj_id = video_obj_id[-1]
        if target_key is None:
            print(video_obj_id)
        sub_value = data[target_key][obj_id]




    # # 遍历JSON数据
    # for key, value in data.items():
    #     if is_to_remove(key[:4]):

    #         new_key = re.sub(r'^\d+_', '', key)
    #     else:
    #         new_key = key
    #     target_key = is_substr_in_ls(new_key, all_obj)
    #     if target_key == None:
    #         continue


    #     cnted_key.append(new_key)
    #     for sub_key, sub_value in value.items():
        before_state = sub_value['before_state'].split(":")[-1]
        after_state = sub_value['after_state'].split(":")[-1]
        # print(after_state)
        transition = f"{before_state}2{after_state}"
        state_transition_counts[transition] += 1
        cnt += 1

    # 打印结果
    for transition, count in state_transition_counts.items():
        print(f"{transition}: {count}")
    print("all cnt:", cnt)
    print("len:", len(all_video))
    print(find_duplicates(cnted_key))

    import csv


    with open('state_transitions.csv', 'w', newline='') as csvfile:
        fieldnames = ['before_state', 'after_state']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for transition, count in state_transition_counts.items():
            for id_ in range(count):
                before_state, after_state = transition.split('2')
                writer.writerow({'before_state': before_state, 'after_state': after_state})

    print('state_transitions.csv')

    





if __name__ == "__main__":
    before_after_cnt()



        

    