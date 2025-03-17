#!/bin/bash


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
# results_path="/path/to/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/davis2017/davis2017_val_aotplus_R50_AOTL_pre_vost_ckpt_unknown_ema/Annotations/480p"
# results_path="/path/to/new_Rmem/RMem/aot_plus/results/_SwinB_DeAOTL/pre_vost/eval/davis2017/davis2017_val__SwinB_DeAOTL_pre_vost_ckpt_unknown_ema/Annotations/480p"


# DeAOT + RMEM
# results_path="/path/to/new_Rmem/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/davis2017/davis2017_val_aotplus_R50_DeAOTL_Temp_pe_Slot_4_pre_vost_ckpt_unknown_ema/Annotations/480p"
# results_path="/path/to/new_Rmem/RMem/aot_plus/results/aotplus_infer_rmem_in_davis2017_using_pre_dav_31500_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/davis2017/davis2017_val_aotplus_infer_rmem_in_davis2017_using_pre_dav_31500_R50_DeAOTL_Temp_pe_Slot_4_pre_vost_ckpt_unknown_ema/Annotations/480p/"
# results_path="/path/to/RMem/aot_plus/results/aotplus_R50_DeAOTL_Temp_pe_Slot_4/pre_vost/eval/davis2017/davis2017_val_aotplus_R50_DeAOTL_Temp_pe_Slot_4_pre_vost_ckpt_unknown_ema/Annotations/480p"
# DeAOT
# results_path="/path/to/aot-benchmark/results/davis2017/davis2017_val_test_roves_in_deaot_swin_week_8_SwinB_DeAOTL_PRE_ckpt_unknown/Annotations/480p"

# DeAOT + RMEM wo TPE
# results_path="/path/to/new_Rmem/RMem/aot_plus/results/infer_rmem_in_davis2017_using_office_ytb_davis_pre_no_long_R50_DeAOTL_No_long_mem/pre_vost/eval/davis2017/davis2017_val_infer_rmem_in_davis2017_using_office_ytb_davis_pre_no_long_R50_DeAOTL_No_long_mem_pre_vost_ckpt_unknown_ema/Annotations/480p"
# AOT_different_fps
# results_path="./RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/test_different_fps_roves_in_aot_week_0_fps_${fps}"

# Xmem (Temp)
# results_path="/path/to/XMem/output/d17-val" 

# Cutie (Temp)
# results_path="/path/to/Cutie/cutie_output/roves_week${week_num}/roves-val/Annotations"

# myVOS
# results_path="/path/to/myVOS/cutie_output/roves_week${week_num}_ckpt176300/roves-val/Annotations"

# SAM2
# results_path="/path/to/segment-anything-2/outputs/sam2_in_davis2017"

# TAM_VT
results_path="/path/to/TAM-VT-main/checkpoints/pretrain_vost/davis_eval_davis/to_eval_pred"

dataset="davis2017"

CUDA_VISIBLE_DEVICES=${gpu_id} python  ./evaluation/evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --fps ${fps}