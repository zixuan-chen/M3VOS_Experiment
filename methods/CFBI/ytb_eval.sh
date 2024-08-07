#!/bin/bash
source activate LMMSeg_cp310

module load anaconda/2021.11  compilers/cuda/11.8  cudnn/8.8.1.3_cuda11.x  cmake/3.26.3  compilers/gcc/9.3.0 
config="configs.resnet101_cfbi"
datasets="youtubevos"
ckpt_path="./pretrain_models/resnet101_cfbi.pth"
python tools/eval_net.py --config ${config} --dataset ${datasets} --ckpt_path ${ckpt_path}