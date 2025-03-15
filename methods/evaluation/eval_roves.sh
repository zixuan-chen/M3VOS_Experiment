#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

week_num=2
gpu_id=0
# fps=1
# fps=4
# fps=6
# fps=8
# fps=12
fps=24

# CFBI
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/CFBI/result/resnet101_cfbi/eval/roves/roves_test_roves_week_${week_num}_ckpt_unknown/Annotations"

# AOT
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_roves_in_aot_week_${week_num}"

#DeAOT
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/aot-benchmark/results/roves/roves_val_low_res_infer_roves_in_deaot_swin_week_${week_num}_SwinB_DeAOTL_PRE_ckpt_unknown"

# DeAOT + RMEM
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/roves/test_roves_week_in_deaotRmem_${week_num}"
# results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_DeAOTL/pre_vost/eval/roves/test_rmem_wo_pte_roves_in_aot_week_${week_num}"
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

# TAM_VT
results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main/checkpoints/pretrain_vost/roves_eval_week_${week_num}/to_eval_pred"

dataset="roves"

CUDA_VISIBLE_DEVICES=${gpu_id} python  ./evaluation/evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --week_num ${week_num} --fps ${fps}