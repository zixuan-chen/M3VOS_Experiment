import torch
import torch.nn as nn

from networks.layers.transformer import DualBranchGPM
from networks.models.aot import AOT
from networks.decoders import build_decoder
from timm.models.layers import trunc_normal_


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


import re

class DeAOT_LLM(AOT):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__(cfg, encoder, decoder)


        self.init_LLM_model(cfg)

        if self.use_temporal_pe:
            self.temporal_CA = Cross_Attention_Block(cfg.MODEL_ENCODER_EMBEDDING_DIM, cfg.MODEL_HIDDEN_DIM)
        self.frame_CA = Cross_Attention_Block(cfg.MODEL_ENCODER_EMBEDDING_DIM, cfg.MODEL_HIDDEN_DIM)


        self.LSTT = DualBranchGPM(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True)

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * \
            (cfg.MODEL_LSTT_NUM * 2 +
             1) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_ENCODER_EMBEDDING_DIM * 2

        self.decoder = build_decoder(
            decoder,
            in_dim=decoder_indim,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)

        self.id_norm = nn.LayerNorm(cfg.MODEL_ENCODER_EMBEDDING_DIM)

        self._init_weight()

        self.use_temporal_pe = cfg.USE_TEMPORAL_POSITIONAL_EMBEDDING
        if self.cfg.USE_TEMPORAL_POSITIONAL_EMBEDDING:
            self.cur_pos_emb = nn.Parameter(torch.randn(1, cfg.MODEL_ENCODER_EMBEDDING_DIM //2) * 0.05)
            if self.cfg.TEMPORAL_POSITIONAL_EMBEDDING_SLOT_4:
                self.mem_pos_emb = nn.Parameter(torch.randn(4, cfg.MODEL_ENCODER_EMBEDDING_DIM //2) * 0.05)
            else:
                self.mem_pos_emb = nn.Parameter(torch.randn(2, cfg.MODEL_ENCODER_EMBEDDING_DIM //2) * 0.05)
            trunc_normal_(self.cur_pos_emb, std=.05)
            trunc_normal_(self.mem_pos_emb, std=.05)
        else:
            self.temporal_encoding = None

        

    def init_LLM_model(self, cfg):
        # Model
        disable_torch_init()

        self.LLM_model_name = get_model_name_from_path(cfg.LLM_MODEL_PATH)

        self.LLM_tokenizer, self.LLM_model, self.LLM_image_processor, _ = load_pretrained_model(
        cfg.LLM_MODEL_PATH, cfg.LLM_MODEL_BASE, self.LLM_model_name
        )

    


    def get_embedding_from_LLM(self,query, frames):
        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if self.LLM_model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.LLM_model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            
        if "llama-2" in self.LLM_model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.LLM_model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.LLM_model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.LLM_model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.LLM_model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"


        conv_mode = conv_mode
            
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = frames
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.LLM_image_processor,
            self.LLM_model.config
        ).to(self.LLM_model.device, dtype=torch.float16)


        input_ids = (
            tokenizer_image_token(prompt, self.LLM_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        output = self.LLM_model(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            output_hidden_states = True,
            use_cache=True,

        )

        return output["hidden_states"]



    


    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, -1).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts)
        return pred_logit

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        id_emb = self.id_dropout(id_emb)
        return id_emb

    def LSTT_forward(
        self,
        curr_embs,
        curr_id_emb=None,
        pos_emb=None,
        size_2d=(30, 30),
        temporal_encoding=None,
        is_outer_memory=False,
        outer_long_memories=None,
        outer_short_memories=None,
        save_atten_weights=False,
    ):

        curr_emb = bchw_2_lbc(curr_embs[-1])
        lstt_embs = self.LSTT(
            curr_emb,
            curr_id_emb,
            pos_emb,
            size_2d=size_2d,
            temporal_encoding=temporal_encoding,
            is_outer_memory=is_outer_memory,
            outer_long_memories=outer_long_memories,
            outer_short_memories=outer_short_memories,
            save_atten_weights=save_atten_weights,
        )
        return lstt_embs



class Cross_Attention_Block(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(Cross_Attention_Block, self).__init__()
        self.query_proj = nn.Linear(emb_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, emb, hidden_state):
        # Step 2: Compute query, key, value
        query = self.query_proj(emb)
        key = self.key_proj(hidden_state)
        value = self.value_proj(hidden_state)

        # Step 3: Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)

        # Step 4: Apply attention weights
        context_vector = torch.matmul(attention_weights, value)

        # Step 5: Optional - Add residual connection
        output = context_vector + emb

        return output