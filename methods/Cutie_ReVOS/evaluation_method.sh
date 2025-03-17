#!/bin/bash

cd ../evaluation
results_path="cutie_output/vost_without_voter/vost-val/Annotations"
dataset="vost"
fps=24

if [ "$dataset" == "vost" ]; then
    python ./evaluation_method.py --results_path ${results_path} --dataset_path ${dataset} --re
elif [ "$dataset" == "roves" ]; then
    python  ./evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re \
        --fps ${fps}
else
    echo "data is neither 'roves' nor 'vost'"
fi