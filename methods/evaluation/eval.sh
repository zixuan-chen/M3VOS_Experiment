#!/bin/bash

week_num=13
gpu_id=0
# fps=1
# fps=4
# fps=6
# fps=8
# fps=12
fps=24

# CFBI
# results_path="/path/to/CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations"

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

#DeAOT
# results_path="/path/to/aot-benchmark/results/roves/roves_val_low_res_infer_roves_in_deaot_swin_week_${week_num}_SwinB_DeAOTL_PRE_ckpt_unknown"

# DeAOT + RMEM
# results_path="/path/to/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"
results_path="/path/to/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves/test_rmem_wo_pte_roves_in_aot_week_${week_num}"
# AOT_different_fps
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_different_fps_roves_in_aot_week_0_fps_${fps}"

# Xmem (Temp)
# results_path="/path/to/XMem/output/roves_week${week_num}" 

# Cutie (Temp)
# results_path="/path/to/Cutie/cutie_output/roves_week${week_num}/roves-val/Annotations"

# myVOS
# results_path="/path/to/myVOS/cutie_output/roves_week${week_num}_ckpt176300/roves-val/Annotations"

# SAM2
# results_path="./segment-anything-2/outputs/sam2_in_roves_week_${week_num}"

dataset="roves"

CUDA_VISIBLE_DEVICES=${gpu_id} python  ./evaluation/evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --week_num ${week_num} --fps ${fps}