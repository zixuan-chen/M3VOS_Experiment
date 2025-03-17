#!/bin/bash

dataset="roves"

if [ "$dataset" == "roves" ]; then
    python cutie/eval_vos.py dataset=roves-val size=480 \
            weights=output/cutie-base-mega.pth model=base \
            exp_id=roves gpu=0
elif [ "$dataset" == "vost" ]; then
    python cutie/eval_vos.py dataset=vost-val size=-1 \
            weights=output/cutie-base-mega.pth model=base \
            exp_id=vost gpu=0 
else
    echo "data is neither 'roves' nor 'vost'"
fi