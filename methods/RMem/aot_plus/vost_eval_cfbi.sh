config="configs.resnet101_cfbi"
datasets="VOST"
ckpt_path="./pretrain_models/resnet101_cfbi.pth"
python tools/eval.py --config ${config} --dataset ${datasets} --ckpt_path ${ckpt_path}