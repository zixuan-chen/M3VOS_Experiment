#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

week_num=1
# fps=1
# fps=4
# fps=6
# fps=8
# fps=12
fps=24

# CFBI
results_path="./CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations"

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

# DeAOT + RMEM
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"

# AOT_different_fps
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_different_fps_roves_in_aot_week_0_fps_${fps}"

dataset="roves"

python  ./evaluation/evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --week_num ${week_num} --fps ${fps}