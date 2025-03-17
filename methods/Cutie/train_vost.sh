#!/bin/bash

OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=cutie-base-vost model=base data=with-vost