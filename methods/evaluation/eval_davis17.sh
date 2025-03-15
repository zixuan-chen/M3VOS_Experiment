#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6


# CFBI
# results_path="./CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations"

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

# DeAOT + RMEM
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"

# AOT_different_fps
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_different_fps_roves_in_aot_week_0_fps_${fps}"

# Xmem (Temp)
results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/XMem/output/d17-val" 

# Cutie (Temp)
# results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie/cutie_output/roves_week${week_num}/roves-val/Annotations"

# SAM2
# results_path="./segment-anything-2/outputs/sam2_in_roves_week_${week_num}"

# myVOS 
# results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/myVOS/cutie_output/roves_week8_with_voter/roves-val/Annotations"

dataset="d17"

set="2017/val"

python  ./evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --set ${set}