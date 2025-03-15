#!/bin/bash


source activate LLMSeg_cp310
module load compilers/cuda/11.8

exp="aotplus"
# exp="debug"
gpu_num="1"
devices="0"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
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
	--fix_random \
	# --debug_fix_rando




# result_path="${result_path}/eval/${dataset}/${eval_name}/"
# echo "result_path=$result_path"


# model_name=$(python -c "from configs.models.$model import ModelConfig ;print(ModelConfig().MODEL_NAME)")
# cd ../evaluation
# python ./evaluation_method.py --results_path "../aot_plus/${result_path}" --dataset_path ${dataset} --re