#!/bin/bash

dataset="roves" # vost, roves, d17, y19
set="val"
exp_id=${dataset}_${set}
weights=weights/mega_v4_main_training_72800.pth


python cutie/eval_vos.py dataset=${dataset}-${set} size=480 \
        weights=${weights} model=base \
        exp_id=${exp_id} gpu=0 

cd ../evaluation
results_path="cutie_output/${exp_id}/${dataset}-${set}/Annotations"

python  ./evaluation_method.py  \
        --results_path ${results_path} \
        --dataset_path ${dataset}  \
        --set ${set} \
        --re \

