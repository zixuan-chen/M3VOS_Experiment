#!/bin/bash

OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=mega_v4 model=base data=mega checkpoint=output/mega_v4/mega_v4_main_training_ckpt_last.pth