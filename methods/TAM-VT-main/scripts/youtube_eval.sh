#!/bin/sh

module load compilers/cuda/11.6
module load compilers/gcc/9.3.0
module load cudnn/8.6.0.163_cuda11.x
module load nccl/2.17.1-1_cuda11.6
module load anaconda/2021.11

conda activate activate tam_vt


export MASTER_PORT=$((29500))  # 将端口号设置为 29500 + (week % 7)
echo $MASTER_PORT
cd /home/bingxing2/home/scx8ah2/jiaxin/DeformVOS/methods/TAM-VT-main

export OUTPUT_DIR_CHECKPOINT="checkpoints/pretrain_vost/eval_youtube"


python main_youtube_memory.py \
    --config_path ./config/youtube_multi_scale_memory.yaml  \
    ema=True \
    resume=checkpoints/tamvt_vost.pth \
    val_tasks.names="['youtube']"  \
    tasks.names="['vost']" epochs=100 eval_skip=5 lr_drop=50 num_workers=1 \
    train_flags.print_freq=5 \
    resolution=624 model.vost.video_max_len=12 \
    output_dir=${OUTPUT_DIR_CHECKPOINT} \
    sted=False model.use_score_per_frame=False aux_loss=True \
    data.youtube.use_test_transform=False \
    model.vost.memory.clip_length=2 model.vost.memory.bank_size=9 \
    model.name.tubedetr=tubedetr_vost_multi_scale_memory_ms model.name.transformer=transformer_vost_multi_scale_memory_last_layer_too_ms \
    lr=0.0001 lr_backbone=2.0e-05 \
    use_rand_reverse=True use_ignore_threshold=True \
    model.vost.memory.rel_time_encoding=True model.vost.memory.rel_time_encoding_type="embedding_dyna_mul" \
    eval_flags.vost.eval_first_window_only=False \
    eval=True eval_flags.plot_pred=True eval_flags.vost.vis_only_no_cache=True  \
    eval_flags.vost.vis_only_pred_mask=True \
    eval_flags.youtube.eval_first_window_only=False \
    eval_flags.youtube.vis_only_no_cache=True  \
    eval_flags.youtube.vis_only_pred_mask=True \
    backbone=resnet50 \
    loss_coef.vost.reweighting_tau=1.0 loss_coef.vost.reweighting=True 

# torchrun --nproc_per_node=1 main_roves_memory.py \
#     --config_path ./config/roves_multi_scale_memory.yaml  \
#     ema=True \
#     data.roves.week=ROVES_week_${week} \
#     resume=checkpoints/tamvt_vost.pth \
#     val_tasks.names="['roves']"  \
#     tasks.names="['vost']" epochs=100 eval_skip=5 lr_drop=50 num_workers=1 \
#     train_flags.print_freq=5 \
#     resolution=624 model.vost.video_max_len=12 \
#     output_dir=${OUTPUT_DIR_CHECKPOINT} \
#     sted=False model.use_score_per_frame=False aux_loss=True \
#     data.roves.use_test_transform=False \
#     model.vost.memory.clip_length=2 model.vost.memory.bank_size=9 \
#     model.name.tubedetr=tubedetr_vost_multi_scale_memory_ms model.name.transformer=transformer_vost_multi_scale_memory_last_layer_too_ms \
#     lr=0.0001 lr_backbone=2.0e-05 \
#     use_rand_reverse=True use_ignore_threshold=True \
#     model.vost.memory.rel_time_encoding=True model.vost.memory.rel_time_encoding_type="embedding_dyna_mul" \
#     eval_flags.vost.eval_first_window_only=False \
#     eval=True eval_flags.plot_pred=True eval_flags.vost.vis_only_no_cache=True  \
#     eval_flags.vost.vis_only_pred_mask=True \
#     eval_flags.roves.eval_first_window_only=False \
#     eval_flags.roves.vis_only_no_cache=True  \
#     eval_flags.roves.vis_only_pred_mask=True \
#     backbone=resnet50 \
#     loss_coef.vost.reweighting_tau=1.0 loss_coef.vost.reweighting=True &
# wait
