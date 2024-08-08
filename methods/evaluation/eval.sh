#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

week_num=1

# CFBI
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations"

# AOT
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

# DeAOT + RMEM
results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"
dataset="roves"

python  ./evaluation/evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --week_num ${week_num}