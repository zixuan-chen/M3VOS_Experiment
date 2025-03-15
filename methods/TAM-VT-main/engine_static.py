# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
# import os
# import json
# import shutil
import random
import math
import sys
# import copy
from typing import Dict, Iterable, Optional
from collections import OrderedDict
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

import torch
import torch.nn
import torch.optim
import torch.nn.functional as F

import util.dist as dist
# from datasets.vidstg_eval import VidSTGEvaluator
# from datasets.hcstvg_eval import HCSTVGEvaluator
# from datasets.vq2d_eval import VQ2DEvaluator
# from datasets.vq2d_orig_eval import VQ2DOrigEvaluator
# from datasets.nlq_orig_eval import NLQOrigEvaluator
# from datasets.mq_orig_eval import MQOrigEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema
# from scipy.signal import find_peaks, medfilt
from util.misc import targets_to, NestedTensor

# from pathlib import Path
# from PIL import Image


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    data_loader: Dict,
    weight_dict: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    writer=None,
):
    model.train()
    if criterion is not None:
        criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = args.train_flags.print_freq
    num_training_steps = int(len(data_loader) * args.epochs)

    for i, batch_dict in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # if args.debug: import ipdb; ipdb.set_trace()
        # if args.debug:
        # print(f"Fetched batch {i} | Rank: {dist.get_rank()} | "
        #     f"Task: {batch_dict['task_name']} | Samples shape: {batch_dict['samples'].tensors.shape}")

        task_name = batch_dict['task_name'][0]
        curr_step = epoch * len(data_loader) + i

        samples = batch_dict["samples"].to(device)
        if "samples_fast" in batch_dict:
            samples_fast = batch_dict["samples_fast"].to(device)
        else:
            samples_fast = None

        durations = batch_dict["durations"]
        targets = batch_dict["targets"]
        targets = targets_to(targets, device)

        # memory initialization
        # kwargs = {}

        memory_encoded = OrderedDict()
        memory = OrderedDict({
            _k: {
                "image": NestedTensor(batch_dict["memory_images"].tensors[_k: _k + 1], batch_dict["memory_images"].mask[_k: _k + 1]).to(device),
                "mask": NestedTensor(batch_dict["memory_masks"].tensors[_k: _k + 1], batch_dict["memory_masks"].mask[_k: _k + 1]).to(device),
                }
            for _k in range(batch_dict["memory_images"].tensors.shape[0])
        })

        # print(f"Rank: {dist.get_rank()} | memory_encoded: {memory_encoded.keys()}"
        #       f" | memory: {memory.keys()}")

        # if args.debug: import ipdb; ipdb.set_trace()

        #######################################################################
        if args.train_flags.static.multi_object.enable:
            raise NotImplementedError()
            outputs = {}
            # print(f"Num objects: {kwargs['reference_crop'].tensors.shape[0]}")
            for _i_reference_crop in range(kwargs["reference_crop"].tensors.shape[0]):
                reference_crop = NestedTensor(
                    kwargs["reference_crop"].tensors[_i_reference_crop][None, ...],
                    kwargs["reference_crop"].mask[_i_reference_crop][None, ...]
                )

                # forward
                memory_cache_ref = model(
                    task_name,
                    samples,
                    durations,
                    captions,
                    encode_and_save=True,
                    samples_fast=samples_fast,
                    reference_crop=reference_crop,
                    reference_box_coord=targets[0]['boxes'][_i_reference_crop],
                )
                outputs_ref = model(
                    task_name,
                    samples,
                    durations,
                    captions,
                    encode_and_save=False,
                    memory_cache=memory_cache_ref,
                    reference_crop=reference_crop
                )

                if 'pred_masks' in outputs:
                    outputs['pred_masks'].append(outputs_ref['pred_masks'])
                else:
                    outputs['pred_masks'] = [outputs_ref['pred_masks']]

                if "aux_outputs" in outputs_ref:
                    if "aux_outputs" not in outputs:
                        outputs["aux_outputs"] = {
                            "pred_masks": [{"pred_masks": [_e["pred_masks"]]} for _e in outputs_ref["aux_outputs"]["pred_masks"]]
                        }
                    else:
                        for _e1, _e2 in zip(
                            outputs["aux_outputs"]["pred_masks"],
                            outputs_ref["aux_outputs"]["pred_masks"]
                        ):
                            _e1["pred_masks"].append(_e2["pred_masks"])

            outputs['pred_masks'] = torch.stack(outputs['pred_masks'])  # num_obj, seq, 2 (classes), H, W

            if "aux_outputs" in outputs:
                outputs["aux_outputs"]["pred_masks"] = [
                    {"pred_masks": torch.stack(e["pred_masks"])}
                    for e in outputs["aux_outputs"]["pred_masks"]
                ]

        else:
            window_step_size = args.model.static.memory.clip_length
            outputs = {}
            for ind_start in range(0, durations[0], window_step_size):
                # if args.debug: import ipdb; ipdb.set_trace()
                ind_end = min(durations[0], ind_start + args.model.static.memory.clip_length)

                samples_window = NestedTensor(
                    samples.tensors[ind_start: ind_end],
                    samples.mask[ind_start: ind_end]
                )

                # forward
                memory_cache, memory_encoded, memory = model(
                    task_name,
                    samples_window,
                    [ind_end - ind_start],
                    encode_and_save=True,
                    samples_fast=None,
                    memory_encoded=memory_encoded,
                    memory=memory,
                    # **kwargs
                )
                outputs_window = model(
                    task_name,
                    samples_window,
                    [ind_end - ind_start],
                    encode_and_save=False,
                    memory_cache=memory_cache,
                    # **kwargs
                )

                # accumulate output
                if 'pred_masks' in outputs:
                    outputs['pred_masks'].append(outputs_window['pred_masks'])
                else:
                    outputs['pred_masks'] = [outputs_window['pred_masks']]

                if "aux_outputs" in outputs_window:
                    if "aux_outputs" not in outputs:
                        outputs["aux_outputs"] = {
                            "pred_masks": [{"pred_masks": [_e["pred_masks"]]} for _e in outputs_window["aux_outputs"]["pred_masks"]]
                        }
                    else:
                        for _e1, _e2 in zip(
                            outputs["aux_outputs"]["pred_masks"],
                            outputs_window["aux_outputs"]["pred_masks"]
                        ):
                            _e1["pred_masks"].append(_e2["pred_masks"])

                # propagate memory
                if args.model.static.memory.teacher_forcing.enable and random.random() < (100 - epoch) / 100:
                    # if args.debug: import ipdb; ipdb.set_trace()
                    _pred_last = targets[ind_end - 1]['masks'][None].float()
                else:
                    _pred_last = F.interpolate(
                        outputs_window['pred_masks'][-1][None], size=samples.tensors.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).sigmoid()

                # if ("detach_predictions" in args.model.static.memory
                #     and args.model.static.memory.detach_predictions
                # ):
                #     _pred_last = _pred_last.detach()

                _mem_image_forward = samples_window.tensors[-1].unsqueeze(1)  # (C, N, H, W)
                _mem_image_forward = NestedTensor.from_tensor_list([_mem_image_forward.to(device)])

                _mem_mask_forward = _pred_last.detach()  # (C, N, H, W)
                _mem_mask_forward = NestedTensor.from_tensor_list([_mem_mask_forward.to(device)])

                if len(memory) >= args.model.static.memory.bank_size:
                    _key_to_remove = [*memory.keys()][1]  # keep first frame fixed
                    assert _key_to_remove != 0

                    # print(f"[PREV] Rank: {dist.get_rank()} | memory_encoded: {memory_encoded.keys()}"
                    #       f" | memory: {memory.keys()}")

                    # remove entries from memory and encoded memory
                    del memory[_key_to_remove]
                    del memory_encoded[_key_to_remove]

                    # print(f"[AFTER] Rank: {dist.get_rank()} | memory_encoded: {memory_encoded.keys()}"
                    #       f" | memory: {memory.keys()}")

                    # memory.popitem(last=False)
                    # memory = OrderedDict({k - 1: v for k, v in memory.items()})

                memory.update({max(memory.keys()) + 1: {
                    "image": _mem_image_forward, "mask": _mem_mask_forward
                }})


            # if args.debug: import ipdb; ipdb.set_trace()

            # Concat output together
            outputs['pred_masks'] = torch.cat(outputs['pred_masks'])  # seq, 1, H, W

            if "aux_outputs" in outputs:
                outputs["aux_outputs"]["pred_masks"] = [
                    {"pred_masks": torch.cat(e["pred_masks"])}
                    for e in outputs["aux_outputs"]["pred_masks"]
                ]

        #######################################################################

        # if args.debug: import ipdb; ipdb.set_trace()
        inter_idx = None
        b = len(durations)
        targets = [
            x for x in targets if len(x["boxes"])
        ]  # keep only targets in the annotated moment

        # mask with padded positions set to False for loss computation
        if args.sted:
            time_mask = torch.zeros(b, outputs["pred_sted"].shape[1]).bool().to(device)
            for i_dur, duration in enumerate(durations):
                time_mask[i_dur, :duration] = True
        else:
            time_mask = None

        # compute losses
        loss_dict = {}
        # if args.debug: import ipdb; ipdb.set_trace()
        if criterion is not None:
            # loss_dict.update(criterion(outputs, targets, inter_idx, time_mask, segment_type_selected, task_name))
            loss_dict.update(criterion(outputs, targets))

        # loss scaling
        for k, v in loss_dict.items():
            loss_dict[k] = v * args.joint.scale_loss[task_name]

        # if args.debug: import ipdb; ipdb.set_trace()

        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        loss_dict_scaled = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
        # print(f"task_name: {task_name}, losses: {losses}")

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {
        #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        # }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        loss_dict_reduced_scaled_task = {f"{k}_{task_name}": v for k, v in loss_dict_scaled.items()}
        # loss_dict_reduced_unscaled_task = {f"{k}_{task_name}": v for k, v in loss_dict.items()}

        # if args.debug: import ipdb; ipdb.set_trace()
        metric_logger.update(
            **{"loss_total": loss_value}, **{f"loss_{task_name}": losses}, **loss_dict_reduced_scaled_task,
            # **loss_dict_reduced_unscaled_task
        )

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
