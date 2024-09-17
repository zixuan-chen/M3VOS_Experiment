#!/bin/bash
source activate zixuan
module load compilers/cuda/11.6

python -m torch.distributed.run --master_port 25763 --nproc_per_node=2 train.py --exp_id finetune_vost --stage 3 --load_network saves/XMem.pth
