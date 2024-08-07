#!/bin/bash
module load anaconda/2021.11  compilers/cuda/11.8  cudnn/8.8.1.3_cuda11.x  cmake/3.26.3  compilers/gcc/9.3.0
source activate LLMSeg_cp310

python tools/train.py --exp_name debug_LLM --model deaot_LLM --gpu_num 1 --batch_size 4  --stage  train_vost_LLM
# python test_llava.py
