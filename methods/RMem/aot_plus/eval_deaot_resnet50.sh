

#  ------------------ inference in M3VOS ----------------
exp="aotplus"
gpu_num="1"
devices="0"

model="r50_deaotl"
stage="pre_vost"

result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"

dataset="roves"
split="val"
week_num=8
eval_name="test_roves_week_in_deaotRmem_${week_num}"

CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path "${result_path}" \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random \
    --ckpt_path pretrain_models/aotplus_R50_DeOTL_Temp_pe_Slot_4_ema_20000.pth \
	--week_num  ${week_num} \
	--model ${model} \
	--stage ${stage}


# ---------------- inference in davis2017 ----------------
exp="aotplus"
gpu_num="1"
devices="0"

model="r50_deaotl"
stage="pre_vost"

result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"

# dataset="roves"
# dataset="vost"
dataset='davis2017'
split="val"
week_num=8
eval_name="test_davis2017_in_deaot"


CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path "${result_path}" \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random \
    --ckpt_path pretrain_models/aotplus_R50_DeOTL_Temp_pe_Slot_4_ema_20000.pth \
	--week_num ${week_num} \
	--stage ${stage} \
	--model ${model} 





# --------------------- inference in youtubevos2019 ----------------
exp="aotplus"

gpu_num="1"
devices="0"

model="r50_deaotl"

stage="pre_vost"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"


dataset="youtubevos2019"
split="val"
eval_name="debug_"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path ${result_path} \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--ckpt_path pretrain_models/aotplus_R50_DeOTL_Temp_pe_Slot_4_ema_20000.pth \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random 

# ----------------- inference in vost --------------------
exp="aotplus"
# exp="debug"
gpu_num="1"
devices="0"

model="r50_deaotl"
	
stage="pre_vost"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"


dataset="vost"
split="val"
eval_name="debug_"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path ${result_path} \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--ckpt_path pretrain_models/aotplus_R50_DeOTL_Temp_pe_Slot_4_ema_20000.pth \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random 