#!/bin/bash
source activate zixuan
module purge
module load compilers/cuda/11.8
# python cutie/eval_vos.py dataset=vost-val size=-1 weights=output/cutie-base-vost/cutie-base-vost_main_training_last.pth model=base exp_id=vost1080p gpu=0

cd ../evaluation
results_path="/home/bingxing2/home/scx7kwl/zixuan/DeformVOS/methods/Cutie/cutie_output/vost1080p/vost-val/Annotations"
dataset="vost"
python ./evaluation_method.py --results_path ${results_path} --dataset_path ${dataset} --re
