#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

results_path="/home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/RMem/aot_plus/results/aotplus_R50_AOTL/pre_vost/eval/roves/aot_week_1_5"
dataset="roves"
python ./evaluation_method.py --results_path ${results_path} --dataset_path ${dataset} --re