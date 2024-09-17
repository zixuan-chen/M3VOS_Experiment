#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

dataset="roves"
week_num=6

if [ "$dataset" == "roves" ]; then
    python eval.py --model saves/Jul31_22.45.04_finetune_vost_s3/Jul31_22.45.04_finetune_vost_s3_25000.pth  \
                --output ./output/${dataset}_week${week_num} \
                --dataset ROVES \
                --roves_path /home/bingxing2/home/scx8ah2/dataset/ROVES_summary/ROVES_week_${week_num} \
                --split val
elif [ "$dataset" == "vost" ]; then
    python eval.py --model saves/Jul31_22.45.04_finetune_vost_s3/Jul31_22.45.04_finetune_vost_s3_25000.pth  \
                --output ./output/vost \
                --dataset VOST \
                --split val
else
    echo "data is neither 'roves' nor 'vost'"
fi
