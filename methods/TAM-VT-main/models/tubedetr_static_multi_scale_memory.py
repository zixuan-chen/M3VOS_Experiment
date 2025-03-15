# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR model and criterion classes.
"""
from typing import Dict, Optional

# import json
# import numpy as np
import torch
# import torchvision
import torch.distributed
import torch.nn.functional as F
from torch import nn
# import math

import util.dist as dist
from util import box_ops
from pathlib import Path
from util.misc import NestedTensor, interpolate

# from models.segmentation_detr import MHAttentionMap, MaskHeadSmallConv, MaskHeadSmallSmallConv
from models.segmentation_detr import dice_loss, sigmoid_focal_loss


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x


class TubeDETR(nn.Module):
    """This is the TubeDETR module that performs spatio-temporal video grounding"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        args=None,
    ):
        """
        :param backbone: visual backbone model
        :param transformer: transformer model
        :param num_queries: number of object queries per frame
        :param aux_loss: whether to use auxiliary losses at every decoder layer
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param guided_attn: whether to use guided attention loss
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param sted: whether to predict start and end proba
        """
        super().__init__()
        assert args is not None
        self.args = args
        self.backbone = backbone
        self.transformer = transformer

        # common variables to all tasks
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if self.args.model.use_single_query_embed:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        else:
            self.query_embed = nn.ModuleDict({
                "visual": nn.Embedding(self.num_queries, hidden_dim),
                "text": nn.Embedding(self.num_queries, hidden_dim),
            })

        if self.args.model.static.use_second_last_feature:
            backbone.num_channels = backbone.num_channels // 2

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # common
        if self.args.sted:
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.2)
        if self.args.model.use_score_per_frame:
            self.score_per_frame_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.2)

        # if args.debug: import ipdb; ipdb.set_trace()
        if "static" in self.args.tasks.names:
            if self.args.model.static.use_projection_layer:
                if self.args.model.static.get("separate_input_proj_for_all"):
                    self.input_proj_mem_image = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
                    self.input_proj_mem_mask = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
                else:
                    self.input_proj_query = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

            if self.args.train_flags.static.multi_object.enable:
                self.out_conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1)

    def _task_specific_variables_for_forward(self, task_name):
        """
        Setting up task-specific variables
        - stride
        """
        if self.training:
            flags_task_specific = self.args.train_flags
        else:
            flags_task_specific = self.args.eval_flags

        assert task_name in flags_task_specific.keys()
        stride = flags_task_specific[task_name].stride

        return stride

    def forward(
        self,
        task_name: str,
        samples,
        durations,
        encode_and_save=True,
        memory_cache=None,
        samples_fast=None,
        **kwargs,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched frames, of shape [n_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [n_frames x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.

        ipdb> pp [e.tensors.shape for e in features]
        [torch.Size([2, 256, 112, 150]),
        torch.Size([2, 512, 56, 75]),
        torch.Size([2, 1024, 28, 38]),
        torch.Size([2, 2048, 14, 19])]
        ipdb> samples.tensors.shape
        torch.Size([2, 3, 448, 597])
        """
        stride = self._task_specific_variables_for_forward(task_name)

        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            b = len(durations)
            t = max(durations)

            features, pos = self.backbone(samples)

            # if self.args.debug: import ipdb; ipdb.set_trace()

            if isinstance(features[-1], dict):
                src, mask = features[-1]['tensors'], features[-1]['mask']
            else:
                src, mask = features[-1].decompose()  # src (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32)); mask (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))

            # if self.args.debug: import ipdb; ipdb.set_trace()
            # encode reference crop
            if task_name == "static":
                for _ind_time, _mem in kwargs["memory"].items():
                    if _ind_time in kwargs["memory_encoded"]:
                        continue

                    # mem image
                    features_mem_image, pos_mem_image = self.backbone(_mem["image"])
                    src_mem_image, mask_mem_image = features_mem_image[-1].decompose()
                    # if self.args.debug: import ipdb; ipdb.set_trace()

                    if self.args.model.static.use_projection_layer and self.args.model.static.get("separate_input_proj_for_all"):
                        src_mem_image = self.input_proj_mem_image(src_mem_image)  # torch.Size([8, 256, 5, 6])
                    elif self.args.model.static.use_projection_layer and not self.args.model.static.reuse_input_proj_for_image_in_memory:
                        src_mem_image = self.input_proj_query(src_mem_image)  # torch.Size([8, 256, 5, 6])
                    else:
                        src_mem_image = self.input_proj(src_mem_image)  # torch.Size([8, 256, 5, 6])

                    # mem mask
                    _mem_mask_inflated = NestedTensor(_mem["mask"].tensors.repeat(1, 3, 1, 1), _mem["mask"].mask)
                    features_mem_mask, pos_mem_mask = self.backbone(_mem_mask_inflated)
                    src_mem_mask, mask_mem_mask = features_mem_mask[-1].decompose()
                    if self.args.model.static.use_projection_layer:
                        if self.args.model.static.get("separate_input_proj_for_all"):
                            src_mem_mask = self.input_proj_mem_mask(src_mem_mask)  # torch.Size([8, 256, 5, 6])
                        else:
                            src_mem_mask = self.input_proj_query(src_mem_mask)  # torch.Size([8, 256, 5, 6])
                    else:
                        src_mem_mask = self.input_proj(src_mem_mask)  # torch.Size([8, 256, 5, 6])

                    # updating memory
                    _mem_encoded = {"image": {}, "mask": {}}
                    _mem_encoded["image"]["src_last_layer"] = src_mem_image
                    _mem_encoded["image"]["pos_last_layer"] = pos_mem_image[-1]

                    _mem_encoded["mask"]["src_last_layer"] = src_mem_mask
                    _mem_encoded["mask"]["pos_last_layer"] = pos_mem_mask[-1]

                    kwargs["memory_encoded"][_ind_time] = _mem_encoded

            # if self.args.debug: import ipdb; ipdb.set_trace()

            # temporal padding pre-encoder
            src = self.input_proj(src)  # torch.Size([8, 256, 5, 6])
            _, f, h, w = src.shape
            f2 = pos[-1].size(1)
            device = src.device
            tpad_mask_t = None
            fast_src = None

            if not stride:
                tpad_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask = torch.ones(b, t, h, w).bool().to(device)
                pos_embed = torch.zeros(b, t, f2, h, w).to(device)
                cur_dur = 0
                for i_dur, dur in enumerate(durations):
                    tpad_src[i_dur, :dur] = src[cur_dur: cur_dur + dur]
                    tpad_mask[i_dur, :dur] = mask[cur_dur: cur_dur + dur]
                    pos_embed[i_dur, :dur] = pos[-1][cur_dur: cur_dur + dur]
                    cur_dur += dur
                tpad_src = tpad_src.view(b * t, f, h, w)
                tpad_mask = tpad_mask.view(b * t, h, w)
                tpad_mask[:, 0, 0] = False  # avoid empty masks
                pos_embed = pos_embed.view(b * t, f2, h, w)
            else:
                raise NotImplementedError()

            # query embed
            if self.args.model.use_single_query_embed:
                query_embed = self.query_embed.weight
            else:
                if task_name in ['static']:
                    query_embed = self.query_embed['visual'].weight
                else:
                    raise ValueError("Task name: {task_name} not recognized")

            kwargs['features'] = features
            kwargs['pos_features'] = pos

            # if self.args.model.static.pixel_decoder.decode_mask_image_too:
            #     kwargs['features_reference'] = features_reference_crop
            #     kwargs['pos_features_reference'] = pos_reference_crop

            # video-text encoder
            memory_cache = self.transformer(
                task_name,
                tpad_src,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                tpad_mask,  # (n_clips)x(math.ceil(H/32))x(math.ceil(W/32))
                query_embed,  # num_queriesxF
                pos_embed,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                encode_and_save=True,
                durations=durations,  # list of length batch_size
                tpad_mask_t=tpad_mask_t,  # (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
                fast_src=fast_src,  # (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32))
                **kwargs
            )
            # if task_name in ['static']:
            #     memory_cache['features'] = features

            return memory_cache, kwargs["memory_encoded"], kwargs["memory"]

        else:
            assert memory_cache is not None
            # space-time decoder
            hs = self.transformer(
                task_name=task_name,
                img_memory=memory_cache[
                    "img_memory"
                ],  # (math.ceil(H/32)*math.ceil(W/32) + n_tokens)x(BT)xF
                mask=memory_cache[
                    "mask"
                ],  # (BT)x(math.ceil(H/32)*math.ceil(W/32) + n_tokens)
                pos_embed=memory_cache["pos_embed"],  # n_tokensx(BT)xF
                query_embed=memory_cache["query_embed"],  # (num_queries)x(BT)xF
                query_mask=memory_cache["query_mask"],  # Bx(Txnum_queries)
                encode_and_save=False,
                text_memory=memory_cache["text_memory"],
                text_mask=memory_cache["text_attention_mask"],
                pixel_decoder_out=memory_cache["pixel_decoder_out"],
                **kwargs
            )
            out = {}

            # hs = hs.flatten(1, 2)  # n_layersxbxtxf -> n_layersx(b*t)xf
            out["pred_masks"] = hs["pred_masks"]

            # if self.args.debug: import ipdb; ipdb.set_trace()

            # auxiliary outputs
            if self.aux_loss:
                out["aux_outputs"] = {"pred_masks": hs["aux_outputs"]}
                # for i_aux in range(len(out["aux_outputs"])):
                #     pass

            if self.args.train_flags.static.multi_object.enable:
                out["pred_masks"] = self.out_conv(out["pred_masks"])
                if self.aux_loss:
                    for e in out["aux_outputs"]["pred_masks"]:
                        e["pred_masks"] = self.out_conv(e["pred_masks"])

            return out


class SetCriterion(nn.Module):
    """This class computes the loss for TubeDETR."""

    def __init__(self, args, losses, sigma=1):
        """Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            sigma: standard deviation for the Gaussian targets in the start and end Kullback Leibler divergence loss
        """
        super().__init__()
        self.args = args
        self.losses = losses
        self.sigma = sigma

    def loss_masks(self, outputs, targets, num_boxes, device):
        if self.args.train_flags.static.multi_object.enable:
            return self.loss_masks_ce(outputs, targets, num_boxes, device)
        else:
            return self.loss_masks_detr(outputs, targets, num_boxes, device)

    def loss_masks_detr(self, outputs, targets, num_boxes, device):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if "pred_masks" not in outputs:
            return {"loss_mask": torch.tensor(0.0).to(device),
                    "loss_dice": torch.tensor(0.0).to(device)}
        assert "pred_masks" in outputs

        # import ipdb; ipdb.set_trace()
        # since num queries = 1 and we don't have to do any kind matching
        assert outputs["pred_masks"].shape[1] == 1
        src_masks = outputs["pred_masks"].squeeze(1)

        # src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        # since num queries = 1 and we don't have to do any kind matching
        assert target_masks.shape[1] == 1
        target_masks = target_masks.squeeze(1)

        # if self.args.debug: import ipdb; ipdb.set_trace()

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        losses = {
            "loss_mask": sigmoid_focal_loss(
                src_masks, target_masks, num_boxes,
                alpha=self.args.loss_coef.static.mask_alpha,
            ),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_masks_ce(self, outputs, targets, num_boxes, device):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        def Soft_aggregation(ps, K, device):
            num_objects, seq, H, W = ps.shape
            em = torch.zeros(1, K, seq, H, W).to(device)
            em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
            em[0, 1:num_objects + 1] = ps  # obj prob
            em = torch.clamp(em, 1e-7, 1 - 1e-7)
            logit = torch.log((em / (1 - em)))
            return logit

        if "pred_masks" not in outputs:
            return {"loss_ce": torch.tensor(0.0).to(device)}
        assert "pred_masks" in outputs

        max_num_objects = len(targets[0]['id_object_considered_true'])

        # if self.args.debug: import ipdb; ipdb.set_trace()
        # since num queries = 1 and we don't have to do any kind matching
        # assert outputs["pred_masks"].shape[1] == 1
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[:max_num_objects]

        masks = [t["masks"][:max_num_objects] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        num_objects = src_masks.shape[0]
        src_masks = [
            interpolate(src_masks[o], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
            for o in range(num_objects)
        ]
        src_masks = torch.stack(src_masks)  # num_obj, seq, 2 (classes), H, W

        # From STM
        ps = F.softmax(src_masks, dim=2)[:, :, 1]  # num_obj, seq, h, w
        logit = Soft_aggregation(ps, num_objects + 1, device)  # 1, K, seq, H, W

        # target to one-hot
        target_masks_zero_prepended = torch.cat(
            (torch.zeros((target_masks.shape[0], 1, target_masks.shape[2], target_masks.shape[3])).to(device),
            target_masks), dim=1
        )
        target_masks_one_hot = torch.argmax(target_masks_zero_prepended, dim=1)

        losses = {"loss_ce": F.cross_entropy(logit, target_masks_one_hot[None, ...])}

        return losses

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        num_boxes,
        device,
        # **kwargs,
    ):
        loss_map = {
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_boxes, device)  # for bbox

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == n_annotated_frames.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inter_idx: list of [start index of the annotated moment, end index of the annotated moment] for each video
             time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the loss computation
        """
        # if self.args.debug: import ipdb; ipdb.set_trace()
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # device = time_mask.device
        device = targets[0]['masks'].device
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # if self.args.debug: import ipdb; ipdb.set_trace()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    num_boxes,
                    device,
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]["pred_masks"]):
                for loss in self.losses:
                    # kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        num_boxes,
                        device,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # if self.args.debug: import ipdb; ipdb.set_trace()
        return losses


def build_tubedetr(args, backbone, transformer):
    device = torch.device(args.device)

    model = TubeDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args,
    )
    weight_dict = {}

    losses = []
    losses.append("masks")
    weight_dict["loss_mask"] = args.loss_coef.static.mask
    weight_dict["loss_dice"] = args.loss_coef.static.dice
    weight_dict["loss_ce"] = 1

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        args=args,
        losses=losses,
        sigma=args.sigma,
    )
    criterion.to(device)

    return model, criterion, weight_dict
