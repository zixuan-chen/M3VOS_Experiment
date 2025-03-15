import os

import csv

res_done_seqs = os.listdir(os.path.join(r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_8"))

res_done_seqs = [seq for seq in res_done_seqs if not seq.endswith(".csv")]




all_seqs = os.listdir(os.path.join(r"/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/datasets/ROVES_summary/ROVES_week_8/Annotations"))

print(set(all_seqs) - set(res_done_seqs))
print(len(set(all_seqs) - set(res_done_seqs)))
with open('val.txt', "w") as f:
    for seq in (set(all_seqs) - set(res_done_seqs)):


        f.write(seq +'\n')