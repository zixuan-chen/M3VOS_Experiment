
export OUTPUT_DIR_CHECKPOINT="checkpoints/vost_train"

torchrun --nproc_per_node=4 main_vost_memory.py \
    ema=True \
    load=checkpoints/tamvt_static.pth \
    auto_resume=${OUTPUT_DIR_CHECKPOINT} \
    tasks.names="['vost']" epochs=150 eval_skip=5 lr_drop=60 \
    train_flags.print_freq=10 \
    resolution=624 model.vost.video_max_len=12 \
    output_dir=${OUTPUT_DIR_CHECKPOINT} \
    sted=False model.use_score_per_frame=False aux_loss=True \
    eval_flags.vost.eval_first_window_only=False \
    data.vost.use_test_transform=False \
    model.vost.memory.clip_length=2 model.vost.memory.bank_size=9 \
    model.name.tubedetr=tubedetr_vost_multi_scale_memory_ms model.name.transformer=transformer_vost_multi_scale_memory_last_layer_too_ms \
    lr=0.0001 lr_backbone=2.0e-05 train_flags.vost.memory.multiple_indices_for_memory=True \
    use_rand_reverse=True use_ignore_threshold=True \
    model.vost.memory.rel_time_encoding=True model.vost.memory.rel_time_encoding_type="embedding_dyna_mul" \
    backbone=resnet50 \
    schedule="multistep_linear_with_warmup" fraction_warmup_steps=0.01 lr_refine_epochs=10 \
    loss_coef.vost.reweighting_tau=1.0 loss_coef.vost.reweighting=True &
wait
