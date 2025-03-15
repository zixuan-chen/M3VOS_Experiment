#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

cd ../evaluation
results_path="/home/bingxing2/home/scx8ah2/zixuan/DeformVOS/methods/Cutie_ReVOS/cutie_output/vost_without_voter/vost-val/Annotations"
dataset="vost"
week_num=8
fps=24

if [ "$dataset" == "vost" ]; then
    python ./evaluation_method.py --results_path ${results_path} --dataset_path ${dataset} --re
elif [ "$dataset" == "roves" ]; then
    python  ./evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re \
        --week_num ${week_num} --fps ${fps}
else
    echo "data is neither 'roves' nor 'vost'"
fi