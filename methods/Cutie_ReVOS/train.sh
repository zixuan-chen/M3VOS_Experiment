#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

# OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=mega_v2 model=base data=mega weights=output/mega_v2/mega_v2_main_training_175000.pth


OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=mega_v4 model=base data=mega checkpoint=output/mega_v4/mega_v4_main_training_ckpt_last.pth