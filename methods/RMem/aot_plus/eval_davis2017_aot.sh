#!/bin/bash


exp="aotplus"
gpu_num="1"
devices="0"

model="r50_aotl"
stage="pre_vost"

result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"

# dataset="roves"
# dataset="vost"
dataset='davis2017'
split="val"
week_num=8
eval_name="test_roves_in_aot_week_${week_num}"


CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path "${result_path}" \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 \
	--eval_name ${eval_name} \
	--latter_mem_len 8 \
	--fix_random \
    --ckpt_path pretrain_models/aotplus_R50_AOTL_ema_20000_492_370.pth \
	--week_num ${week_num} \
	--stage ${stage} \
	--model ${model} 

# result:  /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/davis2017/davis2017_val_aotplus_R50_AOTL_pre_vost_ckpt_unknown_ema/Annotations/480p



# result_path="${result_path}/eval/${dataset}/${eval_name}/"
# echo "result_path=$result_path"

# model_name=$(python -c "from configs.models.$model import ModelConfig ;print(ModelConfig().MODEL_NAME)")
# cd ../../evaluation
# python ./evaluation_method.py --results_path "../aot_plus/${result_path}" --dataset_path ${dataset} --re