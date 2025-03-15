
import torch
import torch.nn as nn
import math

from einops import rearrange, repeat


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_v, dim_o, num_heads=4, act_fn=nn.GELU,
                 dr=0.1, pre_ln=True, ln=True, residual=True, dim_k=None, args=None):
        super().__init__()

        if dim_k is None:
            dim_k = dim_q

        # heads and temperature
        self.args = args
        self.task = args.tasks.names[0]
        self.num_heads = num_heads
        self.dim_split_q = dim_q // num_heads
        self.dim_split_v = dim_o // num_heads
        self.temperature = math.sqrt(dim_o)
        self.residual = residual
        
        # projection layers
        self.fc_q = nn.Linear(dim_q, dim_q, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_q, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=False)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=False)
        
        # nonlinear activation and dropout
        self.activation = act_fn()
        self.attn_dropout = nn.Dropout(dr)
        
        # layernorm layers
        if pre_ln:
            if dim_q == dim_k:
                self.pre_ln_q = self.pre_ln_k = nn.LayerNorm(dim_q)
            else:
                self.pre_ln_q = nn.LayerNorm(dim_q)
                self.pre_ln_k = nn.LayerNorm(dim_k)
        else:
            self.pre_ln_q = self.pre_ln_k = nn.Identity()
        self.ln = nn.LayerNorm(dim_o) if ln else nn.Identity()

    def compute_attention_scores(self, Q, K, mask=None, **kwargs):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)

        # scaled dot-product attention with mask and dropout
        A = Q_.bmm(K_.transpose(1, 2)) / self.temperature
        A = A.clip(-1e4, 1e4)
        if mask is not None:
            A.masked_fill(mask, -1e38)
        A = A.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        return A
    
    def project_values(self, V):
        # linear projection
        O = self.fc_v(V)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        return O

    def forward(self, Q, K, V, mask=None, get_attn_map=False, disconnect_self_image=False, H=None, W=None, **kwargs):
        # pre-layer normalization
        Q = self.pre_ln_q(Q)
        K = self.pre_ln_k(K)
        
        # lienar projection
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # split into multiple heads
        Q_ = torch.cat(Q.split(self.dim_split_q, 2), 0)
        K_ = torch.cat(K.split(self.dim_split_q, 2), 0)
        V_ = torch.cat(V.split(self.dim_split_v, 2), 0)
        
        # TODO: Relative position embedding:
        # adding encoding K_ 
        K_t = K_.transpose(1, 2)

        if kwargs.get('rel_time_encoding_proj', None) is not None:
            if self.args.model.vost.memory.rel_time_encoding_type == "embedding_add_shaw":
                K_t = K_t + kwargs['rel_time_encoding_proj'].squeeze(-1)[None][..., 0]

        # scaled dot-product attention with mask and dropout
        L = Q_.bmm(K_t) / self.temperature
        L = L.clip(-1e4, 1e4)
        
        # mask
        if mask is not None:
            mask = mask.transpose(1, 2).expand_as(L)
        elif disconnect_self_image:
            assert Q_.size(1) == K_.size(1)
            assert H is not None and W is not None
            N = Q_.size(1) // (H*W)
            mask = torch.block_diag(*[torch.ones(H*W, H*W, device=Q.device) for _ in range(N)]).bool()
        
        if mask is not None:
            L.masked_fill(mask, -1e38)

        if kwargs.get('rel_time_encoding_proj', None) is not None:
            if self.args.model.vost.memory.rel_time_encoding_type == "linear_add":
                L = L + kwargs['rel_time_encoding_proj'][None][..., 0]
            elif self.args.model.vost.memory.rel_time_encoding_type == "linear_mul":
                L = L * kwargs['rel_time_encoding_proj'][None][..., 0]
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_add":
                L = L + kwargs['rel_time_encoding_proj'].squeeze(-1)[None][..., 0]
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_mul":
                L = L * kwargs['rel_time_encoding_proj'].squeeze(-1)[None][..., 0]   
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_dyna_add": 
                L = L + kwargs['rel_time_encoding_proj'].squeeze(-1)[None][..., 0]
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_dyna_mul": 
                L = L * kwargs['rel_time_encoding_proj'].squeeze(-1)[None][..., 0]   
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_add_shaw":
                pass
            else:
                raise NotImplementedError
            
        A = L.softmax(dim=2)
        if mask is not None:
            A.masked_fill(mask, 0)
        A = self.attn_dropout(A)
        
        # apply attention to values
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        # layer normalization
        O = self.ln(O)
        
        # residual connection with non-linearity
        if self.residual:
            O = O + self.activation(self.fc_o(O))
        else:
            O = self.fc_o(O)
            
        if get_attn_map:
            return O, A
        else:
            return O

class AggAttention(nn.Module):

    def __init__(self, dim, num_heads, drop_prob=0.1, batch_first=True):
        super(AggAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=batch_first)
        # self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, y):
        # 1. compute self attention
        _x = x
        x, _ = self.attention(x, y, y)
        
        # 2. add and norm
        x = self.dropout(x)
        x = x + _x
        
        return x

class VTMMatchingModule(nn.Module):
    def __init__(self, dim_w, dim_z, n_attn_heads, n_levels, args=None):
        super().__init__()
        self.args = args
        self.use_spatialtemporal_attn = False
        self.use_bg_matching = False

        # task
        assert len(args.tasks.names) == 1
        self.task = args.tasks.names[0]

        self.matching = nn.ModuleList(
            [
                CrossAttention(dim_w, dim_z, dim_z, num_heads=n_attn_heads, args=args)
                for i in range(n_levels)
            ]
        )
        self.n_levels = n_levels

        if self.use_spatialtemporal_attn:
            # self.spatialtemporal_attn = nn.ModuleList(
            #     [
            #         AggAttention(dim_w, num_heads=n_attn_heads, batch_first=True)
            #         for i in range(n_levels)
            #     ]
            # )
            self.aggregation = nn.Linear(dim_z*2, dim_z)


        if self.use_bg_matching:
            self.matching_bg = nn.ModuleList(
                [
                    CrossAttention(dim_w, dim_z, dim_z, num_heads=n_attn_heads)
                    for i in range(n_levels)
                ]
            )
            self.aggregation = nn.Linear(dim_z*2, dim_z)

        if self.args.model.vost.memory.rel_time_encoding:
            if self.args.model.vost.memory.rel_time_encoding_type == "linear_add" : 
                self.rel_time_theta = nn.Linear(1, 1, bias=True)
                nn.init.zeros_(self.rel_time_theta.weight)
                nn.init.zeros_(self.rel_time_theta.bias)
            elif self.args.model.vost.memory.rel_time_encoding_type == "linear_mul" :
                self.rel_time_theta = nn.Linear(1, 1, bias=True)
                nn.init.zeros_(self.rel_time_theta.weight)
                nn.init.ones_(self.rel_time_theta.bias)
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_add" :
                self.rel_time_theta = nn.Embedding(self.args.model.vost.memory.bank_size+1, 1) 
                nn.init.zeros_(self.rel_time_theta.weight)
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_add_shaw" :
                self.rel_time_theta = nn.Embedding(self.args.model.vost.memory.bank_size+1, 1) 
                nn.init.zeros_(self.rel_time_theta.weight)
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_mul" :
                self.rel_time_theta = nn.Embedding(self.args.model.vost.memory.bank_size+1, 1) 
                nn.init.ones_(self.rel_time_theta.weight)
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_dyna_add" :
                total_num_code = sum([ii+1 for ii in range(self.args.model.vost.memory.bank_size)])
                self.rel_time_theta = nn.Embedding(total_num_code, 1)
                nn.init.zeros_(self.rel_time_theta.weight)
            elif self.args.model.vost.memory.rel_time_encoding_type == "embedding_dyna_mul" :
                total_num_code = sum([ii+1 for ii in range(self.args.model.vost.memory.bank_size)])
                self.rel_time_theta = nn.Embedding(total_num_code, 1)
                nn.init.ones_(self.rel_time_theta.weight)
            else:
                raise NotImplementedError

        if self.args.model.vost.memory.abs_time_encoding:
            self.abs_time_theta = nn.Sequential(
                nn.Linear(1, 1, bias=False),
                nn.ReLU(),
                nn.Linear(1, 1, bias=False),
            )
            nn.init.zeros_(self.abs_time_theta[0].weight)
            nn.init.zeros_(self.abs_time_theta[2].weight)

    def forward(self, W_Qs, W_Ss, Z_Ss, attn_mask=None, abs_time_encoding=None, Z_Ss_bg=None):
        # B, T, N, _, H, W = W_Ss[-1].size()
        assert len(W_Qs) == self.n_levels

        if attn_mask is not None:
            attn_mask = from_6d_to_3d(attn_mask)
        
        if self.use_spatialtemporal_attn:
            raise NotImplementedError
        if self.use_bg_matching:
            raise NotImplementedError

        Z_Qs = []
        for level in range(self.n_levels):
            # B is the batch size, T is the number of frames, N is the number of tokens

            B, T, N, _, H, W = W_Ss[level].size()

            Q = from_6d_to_3d(W_Qs[level])
            K = from_6d_to_3d(W_Ss[level])
            V = from_6d_to_3d(Z_Ss[level])

            time_encoding_proj = None
            # time encoding
            if self.args.model.vost.memory.rel_time_encoding:
                rel_time_encoding = self.get_rel_time_encoding(B, T, N, H, W)
                rel_time_encoding = rel_time_encoding.to(Q.device)
                if self.args.model.vost.memory.rel_time_encoding_type in ["embedding_add", "embedding_add_shaw", "embedding_mul", "embedding_dyna_add", "embedding_dyna_mul"]:
                    rel_time_encoding = rel_time_encoding.long()
                if self.args.model.vost.memory.rel_time_encoding_type in ["embedding_dyna_add", "embedding_dyna_mul"]:
                    code_shift = sum([ii for ii in range(N)])
                    rel_time_encoding = rel_time_encoding + code_shift
                    time_encoding_proj = self.rel_time_theta(rel_time_encoding-1)
                else:
                    time_encoding_proj = self.rel_time_theta(rel_time_encoding)

            if self.args.model.vost.memory.abs_time_encoding:
                abs_time_encoding_expanded = self.get_abs_time_encoding(abs_time_encoding, B, T, N, H, W)
                abs_time_encoding_expanded = abs_time_encoding_expanded.to(Q.device)
                time_encoding_proj = self.abs_time_theta(abs_time_encoding_expanded)


            # TODO: this should be cleaned up

            O = self.matching[level](
                Q, K, V, N=N, H=H, mask=attn_mask,
                rel_time_encoding_proj=time_encoding_proj
            )
                
            Z_Q = from_3d_to_6d(O, B=B, T=T, H=H, W=W)
            Z_Qs.append(Z_Q)

        return Z_Qs

    def get_rel_time_encoding(self, B, T, N, H, W):
        aa = torch.arange(N, 0, step=-1)[None, None, ..., None, None, None].float()
        bb = aa.repeat((B, T, 1, 1, H, W))  # (B T N C H W)

        cc = from_6d_to_3d(bb)

        return cc

    def get_abs_time_encoding(self, abs_time_encoding, B, T, N, H, W):
        aa = torch.as_tensor(abs_time_encoding)[None, None, ..., None, None, None].float()
        bb = aa.repeat((B, T, 1, 1, H, W))  # (B T N C H W)
        cc = from_6d_to_3d(bb)

        return cc


def get_reshaper(pattern):
    def reshaper(x, contiguous=False, **kwargs):
        if isinstance(x, torch.Tensor):
            x = rearrange(x, pattern, **kwargs)
            if contiguous:
                x = x.contiguous()
            return x
        elif isinstance(x, dict):
            return {key: reshaper(x[key], contiguous=contiguous, **kwargs) for key in x}
        elif isinstance(x, tuple):
            return tuple(reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x)
        elif isinstance(x, list):
            return [reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x]
        else:
            return x

    return reshaper


from_6d_to_4d = get_reshaper('B T N C H W -> (B T N) C H W')
from_4d_to_6d = get_reshaper('(B T N) C H W -> B T N C H W')

from_6d_to_3d = get_reshaper('B T N C H W -> (B T) (N H W) C')
from_3d_to_6d = get_reshaper('(B T) (N H W) C -> B T N C H W')


def parse_BTN(x):
    if isinstance(x, torch.Tensor):
        B, T, N = x.size()[:3]
    elif isinstance(x, (tuple, list)):
        B, T, N = x[0].size()[:3]
    elif isinstance(x, dict):
        B, T, N = x[list(x.keys())[0]].size()[:3]
    else:
        raise ValueError(f'unsupported type: {type(x)}')

    return B, T, N


def forward_6d_as_4d(func, x, t_idx=None, **kwargs):
    B, T, N = parse_BTN(x)
        
    x = from_6d_to_4d(x, contiguous=True)
    
    if t_idx is not None:
        t_idx = repeat(t_idx, 'B T -> (B T N)', N=N)
        x = func(x, t_idx=t_idx, **kwargs)
    else:
        x = func(x, **kwargs)
    
    x = from_4d_to_6d(x, B=B, T=T)
