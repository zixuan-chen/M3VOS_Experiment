#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

python scripts/convert_burst_to_vos_train.py --json_path /home/bingxing2/home/scx8ah2/dataset/BURST/train/train.json --frames_path  /home/bingxing2/home/scx8ah2/dataset/BURST/frames/train --output_path /home/bingxing2/home/scx8ah2/dataset/BURST/train-vos

