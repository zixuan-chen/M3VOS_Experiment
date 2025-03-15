from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F



class Voter(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super(Voter, self).__init__()
        self.image_size = model_cfg.voter.image_size
        self.hidden_dim = model_cfg.voter.hidden_dim
        self.scale_frame = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.scale_logits = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        # self.scale_logits = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=1+1+3, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.hidden_dim)
        self.conv3 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=1, stride=1)
    
    def forward(self, frame, forward_logits, backward_logits):
        _, num_objects = forward_logits.shape[:2]
        f_logits = F.interpolate(forward_logits, scale_factor=1/4, mode="bilinear", align_corners=False)
        b_logits = F.interpolate(backward_logits, scale_factor=1/4, mode="bilinear", align_corners=False)
        frame = F.interpolate(frame, scale_factor=1/4, mode="bilinear", align_corners=False)
        out_x = []
        for i in range(num_objects):
            xi = torch.cat([self.scale_frame(frame), 
                            self.scale_logits(f_logits[:, i].unsqueeze(1)), 
                            self.scale_logits(b_logits[:, i].unsqueeze(1))], dim=1)
            out_xi = self.relu(self.bn1(self.conv1(xi)))
            out_xi = self.relu(self.bn2(self.conv2(out_xi)))
            out_x.append(self.conv3(out_xi))
        out_x = torch.cat(out_x, dim=1)
        logits = F.interpolate(out_x, scale_factor=4, mode='bilinear', align_corners=False)
        # logits = forward_logits
        prob = F.softmax(logits, dim=1)
        return logits, prob
    
# class Voter(nn.Module):
#     def __init__(self, model_cfg: DictConfig):
#         super(Voter, self).__init__()
#         self.image_size = model_cfg.voter.image_size
#         self.patch_size = model_cfg.voter.patch_size
#         self.hidden_dim = model_cfg.voter.hidden_dim
        
#         self.attention = nn.MultiheadAttention(
#             num_heads=model_cfg.voter.num_heads,
#             embed_dim=self.hidden_dim,
#             dropout=model_cfg.voter.dropout,
#             batch_first=True
#         )
#         # self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, self.hidden_dim).normal_(std=0.02))
#         self.conv_proj = nn.Conv2d(
#                 in_channels=4, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
#             )
#         self.conv_proj_pix = nn.Conv2d(
#                 in_channels=3, out_channels=self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size
#             )
#         self.conv_unproj = nn.Conv2d(
#                 in_channels=self.hidden_dim, out_channels=4, kernel_size=1, stride=1
#             )
        

#     def _process_input(self, x: torch.Tensor, use_pix=False) -> torch.Tensor:
#         n, c, h, w = x.shape
#         p = self.patch_size
#         torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
#         torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
#         n_h = h // p
#         n_w = w // p

#         # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
#         if use_pix:
#             x = self.conv_proj_pix(x)
#         else:
#             x = self.conv_proj(x)
#         # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
#         x = x.reshape(n, self.hidden_dim, n_h * n_w)

#         # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
#         # The self attention layer expects inputs in the format (N, S, E)
#         # where S is the source sequence length, N is the batch size, E is the
#         # embedding dimension
#         x = x.permute(0, 2, 1)

#         return x

#     def forward(self, frame, forward_logits, backward_logits):
#         # frame: B*C*H*W  [4, 3, 480, 480]
#         # forward_logits: B*num_objects*H*W [4, 4, 480, 480]  [4, 480, 480]
#         # backward_logits: B*num_objects*H*W [4, 4, 480, 480]
#         forward_logits = F.interpolate(forward_logits, scale_factor=1/4, mode="bilinear", align_corners=False)
#         backward_logits = F.interpolate(backward_logits, scale_factor=1/4, mode="bilinear", align_corners=False)

#         n, c, h, w = forward_logits.shape

#         n_h, n_w = h // self.patch_size, w // self.patch_size
#         batch_size = len(forward_logits)
        
#         query = self._process_input(forward_logits) # N, L, E
#         key =   self._process_input(frame, use_pix=True)
#         value = self._process_input(backward_logits)

#         out = self.attention(query, key, value) # N, L, E

#         out = out.permute(0, 2, 1).reshape(n, self.hidden_dim, n_h, n_w)

#         out = self.conv_unproj(out)

#         logits = F.interpolate(out, scale_factor=self.patch_size, mode='bilinear') # N, , E

#         logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
#         prob = F.softmax(logits, dim=1)

#         return logits, prob

        