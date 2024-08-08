#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

dataset_root=" ../datasets/ROVES_summary"
week_num=0

# CFBI
# results_path="./CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations/per-sequence_results-val.csv"

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}/per-sequence_results-val.csv"

# DeAOT + RMEM
results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}/per-sequence_results-val.csv"
dataset="roves"

python ./evaluation/cal_challenge_score.py  --datasets_root ${dataset_root} --week_num ${week_num} --result_csv_path ${results_path}
