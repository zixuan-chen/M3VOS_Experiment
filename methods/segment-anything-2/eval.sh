#! /bin/bash
dataset_root="/path/to/your_dataset"

python ./tools/vos_inference.py \
  --sam2_cfg sam2_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2_hiera_base_plus.pt \
  --base_video_dir  ${dataset_root}/JPEGImages \
  --input_mask_dir  ${dataset_root}/Annotations \
  --video_list_file  ${dataset_root}/ImageSets/val.txt \
  --output_mask_dir ./outputs/name_of_mask


results_path=""

dataset="roves"

cd ../evaluation

python  ./evaluation_method.py  --results_path ${results_path} --dataset_path ${dataset} --re --week_num ${week_num}