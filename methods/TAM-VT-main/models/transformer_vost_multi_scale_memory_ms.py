# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
# import json
from typing import List, Optional
# from pathlib import Path

import torch
# import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
# from transformers import RobertaModesl, RobertaTokenizerFast
from detectron2.layers import Conv2d
import math

from .mask2former_layers_ms import build_pixel_decoder
from .position_encoding import TimeEmbeddingSine, TimeEmbeddingLearned

from util.visual_token_matching import VTMMatchingModule


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        no_tsa=False,
        return_weights=False,
        fast=False,
        fast_mode="",
        learn_time_embed=False,
        rd_init_tsa=False,
        no_time_embed=False,
        args=None,
    ):
        """
        :param d_model: transformer embedding dimension
        :param nhead: transformer number of heads
        :param num_encoder_layers: transformer encoder number of layers
        :param num_decoder_layers: transformer decoder number of layers
        :param dim_feedforward: transformer dimension of feedforward
        :param dropout: transformer dropout
        :param activation: transformer activation
        :param return_intermediate_dec: whether to return intermediate outputs of the decoder
        :param pass_pos_and_query: if True tgt is initialized to 0 and position is added at every layer
        :param text_encoder_type: Hugging Face name for the text encoder
        :param freeze_text_encoder: whether to freeze text encoder weights
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param no_tsa: whether to use temporal self-attention
        :param return_weights: whether to return attention weights
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param learn_time_embed: whether to learn time encodings
        :param rd_init_tsa: whether to randomly initialize temporal self-attention weights
        :param no_time_embed: whether to use time encodings
        """
        super().__init__()

        return_weights = False

        self.args = args
        self.pass_pos_and_query = pass_pos_and_query

        # encoder_layer = TransformerEncoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation,
        #     attention_type=self.args.model.vost.attention_encoder
        # )
        # encoder_norm = None
        # self.encoder = TransformerEncoder(
        #     encoder_layer, num_encoder_layers, encoder_norm, return_weights=True,
        # )
        self.encoder = VTMMatchingModule(dim_w=256, dim_z=256, n_attn_heads=4, n_levels=2)

        self.pixel_decoder = build_pixel_decoder(args)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            no_tsa=no_tsa,
            attention_type=self.args.model.vost.attention_decoder,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_weights=return_weights,
        )

        self._reset_parameters()

        self.return_weights = return_weights

        self.learn_time_embed = learn_time_embed
        self.use_time_embed = not no_time_embed
        if self.use_time_embed:
            self.time_embed = nn.ModuleDict()
            for task in self.args.tasks.names:
                if learn_time_embed:
                    self.time_embed.update({
                        task: TimeEmbeddingLearned(self.args.model[task].video_max_len, d_model)
                    })
                    # self.time_embed = TimeEmbeddingLearned(video_max_len, d_model)
                else:
                    self.time_embed.update({
                        task: TimeEmbeddingSine(self.args.model[task].video_max_len, d_model)
                    })
                    # self.time_embed = TimeEmbeddingSine(video_max_len, d_model)

        self.time_embed_memory = TimeEmbeddingSine(self.args.model[task].memory.bank_size, d_model)

        self.fast = fast
        self.fast_mode = fast_mode

        self.rd_init_tsa = rd_init_tsa
        self._reset_temporal_parameters()

        # VOS
        # if "vost" in self.args.tasks.names:
        #     if self.args.model.vost.use_projection_layer:
        #         self.resizer_vq2d = FeatureResizer(
        #             input_feat_size=d_model,
        #             output_feat_size=d_model,
        #             dropout=0.1,
        #         )

        self.d_model = d_model
        self.nhead = nhead
        # self.video_max_len = video_max_len
        # self.stride = stride

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_temporal_parameters(self):
        for n, p in self.named_parameters():
            if "fast_encoder" in n and self.fast_mode == "transformer":
                if "norm" in n and "weight" in n:
                    nn.init.constant_(p, 1.0)
                elif "norm" in n and "bias" in n:
                    nn.init.constant_(p, 0)
                else:
                    nn.init.constant_(p, 0)

            if self.rd_init_tsa and "decoder" in n and "self_attn" in n:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            if "fast_residual" in n:
                nn.init.constant_(p, 0)
            if self.fast_mode == "gating" and "fast_encoder" in n:
                nn.init.constant_(p, 0)

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
        task_name,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        encode_and_save=True,
        durations=None,
        tpad_mask_t=None,
        fast_src=None,
        img_memory=None,
        query_mask=None,
        text_memory=None,
        text_mask=None,
        memory_mask=None,
        **kwargs
    ):
        stride = self._task_specific_variables_for_forward(task_name)

        if encode_and_save:
            # flatten n_clipsxCxHxW to HWxn_clipsxC
            tot_clips, c, h, w = src.shape
            device = src.device

            # nb of times object queries are repeated
            if durations is not None:
                t = max(durations)
                b = len(durations)
                bs_oq = tot_clips if (not stride) else b * t
            else:
                bs_oq = tot_clips

            # if self.args.debug: import ipdb; ipdb.set_trace()
            # src = src.flatten(2).permute(2, 0, 1)
            # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            query_embed = query_embed.unsqueeze(1).repeat(
                1, bs_oq, 1
            )  # n_queriesx(b*t)xf

            n_queries, _, f = query_embed.shape
            query_embed = query_embed.view(
                n_queries * t,
                b,
                f,
            )
            if self.use_time_embed:  # add temporal encoding to init time queries
                time_embed = self.time_embed[task_name](t).repeat(n_queries, b, 1)
                query_embed = query_embed + time_embed

            # prepare time queries mask
            query_mask = None
            if stride:
                query_mask = (
                    torch.ones(
                        b,
                        n_queries * t,
                    )
                    .bool()
                    .to(device)
                )
                query_mask[:, 0] = False  # avoid empty masks
                for i_dur, dur in enumerate(durations):
                    query_mask[i_dur, : (dur * n_queries)] = False

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                raise ValueError()
                # src, tgt, query_embed, pos_embed = (
                #     src + 0.1 * pos_embed,
                #     query_embed,
                #     None,
                #     None,
                # )

            if task_name == "vost":
                text_memory_resized = {}
                text_attention_mask = {}

                # if self.args.debug: import ipdb; ipdb.set_trace()

                # B T N C H W
                # src (T, 256, 15, 25)
                # pos_embed (T, 256, 15, 25)
                # W_Qs = (src + pos_embed).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, 256, 15, 20]

                W_Qs = [
                    (__src + __pos).unsqueeze(0).unsqueeze(2)
                    for __src, __pos in zip(
                        kwargs["curr_image_encoded"]["src_last_two_layers"],
                        kwargs["curr_image_encoded"]["pos_last_two_layers"]
                    )
                ]
                W_Ss_all_levels = []
                Z_Ss_all_levels = []
                for __level in range(2):
                    W_Ss = []
                    Z_Ss = []
                    for _ind_time, _mem in kwargs["memory_encoded"].items():
                        W_Ss.append((_mem['image']['src_last_two_layers'][__level] + _mem['image']['pos_last_two_layers'][__level]))  # [1, 256, 15, 25]
                        Z_Ss.append((_mem['mask']['src_last_two_layers'][__level] + _mem['mask']['pos_last_two_layers'][__level]))  # [1, 256, 15, 25]

                    W_Ss = torch.cat(W_Ss)  # [N, 256, 15, 25]
                    Z_Ss = torch.cat(Z_Ss)  # [N, 256, 15, 25]

                    W_Ss = W_Ss.unsqueeze(0).unsqueeze(0)
                    Z_Ss = Z_Ss.unsqueeze(0).unsqueeze(0)

                    W_Ss_all_levels.append(W_Ss)
                    Z_Ss_all_levels.append(Z_Ss)

                # aa = self.encoder([W_Qs], [W_Ss], [Z_Ss])
                Z_Qs = []
                for _t in range(W_Qs[0].shape[1]):
                    # Z_Qs.append(self.encoder([W_Qs[:, _t][None]], W_Ss_all_levels, Z_Ss_all_levels)[0])
                    Z_Qs.append(self.encoder([W_Qs[0][:, _t][None], W_Qs[1][:, _t][None]], W_Ss_all_levels, Z_Ss_all_levels))

                Z_Qs = [torch.stack([e[__level] for e in Z_Qs]).squeeze(1).squeeze(1).squeeze(1) for __level in range(2)]
                # Z_Qs = torch.stack(Z_Qs)
                # Z_Qs = Z_Qs.squeeze(1).squeeze(1).squeeze(1)  # (tot_clips, c, h, w)

            # if self.args.debug: import ipdb; ipdb.set_trace()
            img_only_memory_reshaped = Z_Qs[0]

            # PIXEL DECODER
            assert len(self.args.model.vost.pixel_decoder.in_features) == len(kwargs['features'])
            # dict_features = {'res5': img_only_memory_reshaped}
            dict_features = {'res5': Z_Qs[0], 'res4': Z_Qs[1]}
            dict_features.update({
                k: v.tensors
                for k, v in zip(self.args.model.vost.pixel_decoder.in_features[:-2], kwargs['features'][:-2])
            })
            mask_features, _, multi_scale_features \
                = self.pixel_decoder.forward_features(dict_features)

            pixel_decoder_out = {
                "mask_features": mask_features, "multi_scale_features": multi_scale_features,
                "pos_features": kwargs["pos_features"],
            }

            # if self.args.model.vost.pixel_decoder.decode_mask_image_too:
            #     ref_memory_split = img_memory[h * w:].split(h * w)
            #     assert len(ref_memory_split) == len(kwargs["memory_encoded"])

            #     multi_scale_features_ref = []
            #     pos_features_ref = []
            #     for __k in kwargs["memory_encoded"].keys():
            #         # ref_only_memory_reshaped = img_memory[h * w:].permute(1, 2, 0).view(tot_clips, c, h, w)
            #         ref_only_memory_reshaped = ref_memory_split[__k].permute(1, 2, 0).view(tot_clips, c, h, w)

            #         dict_features_ref = {'res5': ref_only_memory_reshaped}
            #         dict_features_ref.update({
            #             k: v.tensors
            #             for k, v in zip(self.args.model.vost.pixel_decoder.in_features[:-1], kwargs["memory_encoded"][__k]['features'][:-1])
            #         })

            #         _, _, __multi_scale_features_ref \
            #             = self.pixel_decoder.forward_features(dict_features)

            #         # if self.args.debug: import ipdb; ipdb.set_trace()
            #         multi_scale_features_ref.append(__multi_scale_features_ref)
            #         pos_features_ref.append(kwargs["memory_encoded"][__k]['pos'])

            #     pixel_decoder_out.update({
            #         "multi_scale_features_ref": multi_scale_features_ref,
            #         # "pos_features_ref": kwargs["pos_features_reference"]
            #         "pos_features_ref": pos_features_ref
            #     })

            memory_cache = {
                "text_memory_resized": text_memory_resized,  # before encoder (repeated) [torch.Size([266, 2, 256])]
                "text_memory": text_memory,  # after encoder [torch.Size([266, 2, 256])]
                "text_attention_mask": text_attention_mask,  # all False [torch.Size([2, 266])]
                # "tokenized": tokenized,  # batch first
                "img_memory": img_memory,  # seq first
                "img_only_memory_reshaped": img_only_memory_reshaped,  # seq first ([6, 256, 5, 7])
                "mask": mask,  # all False [torch.Size([2, 532])]
                "pos_embed": pos_embed,  # [torch.Size([532, 2, 256])]
                "query_embed": query_embed,  # [torch.Size([2, 1, 256])]
                "query_mask": query_mask,  # None
                "pixel_decoder_out": pixel_decoder_out,
            }

            return memory_cache

        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = (
                    src + 0.1 * pos_embed,
                    query_embed,
                    None,
                    None,
                )

            # if self.args.debug: import ipdb; ipdb.set_trace()

            # time-space-text attention
            hs = self.decoder(
                tgt,  # n_queriesx(b*t)xF
                img_memory,  # ntokensx(b*t)x256
                memory_key_padding_mask=mask,  # (b*t)xn_tokens
                pos=pos_embed,  # n_tokensx(b*t)xF
                query_pos=query_embed,  # n_queriesx(b*t)xF
                tgt_key_padding_mask=query_mask,  # bx(t*n_queries)
                text_memory=text_memory,
                text_memory_mask=text_mask,
                memory_mask=memory_mask,
                pixel_decoder_out=kwargs["pixel_decoder_out"],
            )  # n_layersxn_queriesx(b*t)xF
            if self.return_weights:
                hs, weights, cross_weights = hs

            return hs


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        return_weights=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

        hidden_dim = 256
        mask_dim = 256

        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        text_memory=None,
        text_memory_mask=None,
        pixel_decoder_out=None
    ):

        x = pixel_decoder_out["multi_scale_features"]
        pos_x = pixel_decoder_out["pos_features"]

        # check if we wish to incorporate multi-scale ref features as well
        flag_ref = "multi_scale_features_ref" in pixel_decoder_out
        if flag_ref:
            x_ref = pixel_decoder_out["multi_scale_features_ref"]
            pos_x_ref = pixel_decoder_out["pos_features_ref"]

        # remove pos for the last layer (res5) which has the same resolution as `mask_features`
        # and this resolution is absent from `multi_scale_features`
        assert len(pos_x) == 4
        pos_x = pos_x[::-1][:self.num_feature_levels]

        if flag_ref:
            assert len(pos_x_ref[0]) == 4
            # pos_x_ref = pos_x_ref[::-1][:self.num_feature_levels]
            pos_x_ref = [
                _pos_x_ref[::-1][:self.num_feature_levels]
                for _pos_x_ref in pos_x_ref
            ]

        src = []
        pos_src = []

        for i in range(self.num_feature_levels):
            pos_src.append(pos_x[i].flatten(2))
            src.append(x[i].flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos_src[-1] = pos_src[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        if flag_ref:
            src_ref = []
            pos_src_ref = []

            # for i in range(self.num_feature_levels):
            #     pos_src_ref.append(pos_x_ref[i].repeat(x_ref[i].shape[0], 1, 1, 1).flatten(2))
            #     src_ref.append(x_ref[i].flatten(2) + self.level_embed.weight[i][None, :, None])

            #     # flatten NxCxHxW to HWxNxC
            #     pos_src_ref[-1] = pos_src_ref[-1].permute(2, 0, 1)
            #     src_ref[-1] = src_ref[-1].permute(2, 0, 1)

            for i in range(self.num_feature_levels):
                _src_ref = []
                _pos_src_ref = []
                for _x_ref, _pos_x_ref in zip(x_ref, pos_x_ref):
                    _pos_src_ref.append(_pos_x_ref[i].repeat(_x_ref[i].shape[0], 1, 1, 1).flatten(2))
                    _src_ref.append(_x_ref[i].flatten(2) + self.level_embed.weight[i][None, :, None])

                    # import ipdb; ipdb.set_trace()
                    # flatten NxCxHxW to HWxNxC
                    _pos_src_ref[-1] = _pos_src_ref[-1].permute(2, 0, 1)
                    _src_ref[-1] = _src_ref[-1].permute(2, 0, 1)

                pos_src_ref.append(torch.cat(_pos_src_ref))
                src_ref.append(torch.cat(_src_ref))

        # import ipdb; ipdb.set_trace()

        output = tgt
        predictions_mask = []

        for i_layer, layer in enumerate(self.layers):
            level_index = i_layer % self.num_feature_levels

            assert not memory_key_padding_mask.any().item()

            output, weights, cross_weights = layer(
                output,
                src[level_index] if not flag_ref else torch.cat((src[level_index], src_ref[level_index])),  # memory
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=None,
                pos=pos_src[level_index] if not flag_ref else torch.cat((pos_src[level_index], pos_src_ref[level_index])),  # pos
                query_pos=query_pos,
                text_memory=text_memory,
                text_memory_mask=text_memory_mask,
            )

            predictions_mask.append(
                self.forward_prediction_heads(output, pixel_decoder_out["mask_features"])
            )

        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }
        return out

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.norm(output)
        # decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        no_tsa=False,
        attention_type="cross_attention",
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.no_tsa = no_tsa

        self.attention_type = attention_type

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        text_memory=None,
        text_memory_mask=None,
    ):
        # import ipdb; ipdb.set_trace()

        q = k = self.with_pos_embed(tgt, query_pos)

        # Temporal Self attention
        if self.no_tsa:
            t, b, _ = tgt.shape
            n_tokens, bs, f = memory.shape
            tgt2, weights = self.self_attn(
                q.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                k.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                value=tgt.transpose(0, 1).reshape(bs * b, -1, f).transpose(0, 1),
                attn_mask=tgt_mask,
                key_padding_mask=None,
            )
            tgt2 = tgt2.view(b, t, f).transpose(0, 1)
        else:
            tgt2, weights = self.self_attn(
                q,
                k,
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Time Aligned Cross attention
        t, b, _ = tgt.shape
        n_tokens, bs, f = memory.shape

        if self.attention_type == "cross_attention":
            tgt_cross = (
                tgt.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
            )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf
            query_pos_cross = (
                query_pos.transpose(0, 1).reshape(bs, -1, f).transpose(0, 1)
            )  # txbxf -> bxtxf -> (b*t)x1xf -> 1x(b*t)xf

            tgt2, cross_weights = self.cross_attn_image(
                query=self.with_pos_embed(tgt_cross, query_pos_cross),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )

        #################################################################
        else:
            k_ = self.with_pos_embed(memory, pos).transpose(0, 1).reshape((bs * n_tokens, 1, f))
            v_ = memory.transpose(0, 1).reshape((bs * n_tokens, 1, f))

            if self.attention_type == "full":
                attn_mask_ = None
            elif self.attention_type == "cross_attention_using_attn_mask":
                attn_mask_ = ~torch.eye(bs, bs).repeat_interleave(n_tokens).reshape(bs, bs * n_tokens).bool().to(tgt.device)
            elif self.attention_type == "local_neighbourhood":
                _mask_neigh = torch.diag(torch.ones(bs), 1)[:bs, :bs] + torch.diag(torch.ones(bs), 0) + torch.diag(torch.ones(bs), -1)[:bs, :bs]
                attn_mask_ = ~_mask_neigh.repeat_interleave(n_tokens).reshape(bs, bs * n_tokens).bool().to(tgt.device)
            elif self.attention_type == "local_neighbourhood_only_past":
                _mask_neigh = torch.diag(torch.ones(bs), 0) + torch.diag(torch.ones(bs), -1)[:bs, :bs]
                attn_mask_ = ~_mask_neigh.repeat_interleave(n_tokens).reshape(bs, bs * n_tokens).bool().to(tgt.device)
            else:
                raise NotImplementedError(f"{self.attention_type}")

            tgt2, cross_weights = self.cross_attn_image(
                query=self.with_pos_embed(tgt, query_pos),
                key=k_,
                value=v_,
                attn_mask=attn_mask_,
            )
            tgt2 = tgt2.transpose(0, 1)
        #################################################################

        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights, cross_weights


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        # video_max_len=args.video_max_len_train,
        # stride=args.stride,
        no_tsa=args.no_tsa,
        return_weights=args.guided_attn,
        fast=args.fast,
        fast_mode=args.fast_mode,
        learn_time_embed=args.learn_time_embed,
        rd_init_tsa=args.rd_init_tsa,
        no_time_embed=args.no_time_embed,
        args=args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
