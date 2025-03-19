#!/bin/bash

OMP_NUM_THREADS=1 torchrun --master_port 25357 --nproc_per_node=1 cutie/train.py exp_id=mega_v4 model=base data=mega weights=cutie/weights/cutie-base-mega.pth