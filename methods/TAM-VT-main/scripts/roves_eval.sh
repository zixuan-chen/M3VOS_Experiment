#!/bin/sh




export MASTER_PORT=$((29500 + week % 7)) 
echo $MASTER_PORT
cd /path/to/TAM-VT-main

export OUTPUT_DIR_CHECKPOINT="checkpoints/pretrain_vost/roves_eval_week_${week}"


python main_roves_memory.py \
    --config_path ./config/roves_multi_scale_memory.yaml  \
    ema=True \
    data.roves.week=ROVES_week_${week} \
    resume=checkpoints/tamvt_vost.pth \
    val_tasks.names="['roves']"  \
    tasks.names="['vost']" epochs=100 eval_skip=5 lr_drop=50 num_workers=1 \
    train_flags.print_freq=5 \
    resolution=624 model.vost.video_max_len=12 \
    output_dir=${OUTPUT_DIR_CHECKPOINT} \
    sted=False model.use_score_per_frame=False aux_loss=True \
    data.roves.use_test_transform=False \
    model.vost.memory.clip_length=2 model.vost.memory.bank_size=9 \
    model.name.tubedetr=tubedetr_vost_multi_scale_memory_ms model.name.transformer=transformer_vost_multi_scale_memory_last_layer_too_ms \
    lr=0.0001 lr_backbone=2.0e-05 \
    use_rand_reverse=True use_ignore_threshold=True \
    model.vost.memory.rel_time_encoding=True model.vost.memory.rel_time_encoding_type="embedding_dyna_mul" \
    eval_flags.vost.eval_first_window_only=False \
    eval=True eval_flags.plot_pred=True eval_flags.vost.vis_only_no_cache=True  \
    eval_flags.vost.vis_only_pred_mask=True \
    eval_flags.roves.eval_first_window_only=False \
    eval_flags.roves.vis_only_no_cache=True  \
    eval_flags.roves.vis_only_pred_mask=True \
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
