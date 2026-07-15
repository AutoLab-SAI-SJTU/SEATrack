import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.timm_compat import DropPath, Mlp, trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index
import math
from typing import Optional, List

# Low-Rank Projection
class LoRP(nn.Module):
    def __init__(self, in_dim=768, hid_dim=4, out_dim=768, drp1=0.1, drp2=0.1, use_bias=True, if_act=False):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty(in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(hid_dim, out_dim))
        self.use_bias = use_bias
        self.if_act = if_act
        if use_bias == True:
            self.bias_a = nn.Parameter(torch.empty(hid_dim))
            self.bias_b = nn.Parameter(torch.empty(out_dim))
        self.act = nn.GELU() if if_act else nn.Identity()
        for n, p in self.named_parameters():
            if 'lora' in n:
                nn.init.xavier_normal_(p, gain=math.sqrt(2))          
            else:
                bound = 1 / math.sqrt(in_dim) if 'a' in n else 1 / math.sqrt(hid_dim)
                nn.init.uniform_(p, -bound, bound)
            
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.view(1, 1, -1)
            x = self.drop1(self.act(x))
            return self.drop2(torch.matmul(x, self.lora_b) + self.bias_b.view(1, 1, -1))
        else:
            return self.drop2(self.drop1(self.act(x @ self.lora_a)) @ self.lora_b)

class MultiExpertLinear(nn.Module):
    def __init__(self, num_experts, in_dim, hid_dim, drp1, drp2, use_bias=True, if_act=False):
        super(MultiExpertLinear, self).__init__()
        self.lora_a = nn.Parameter(torch.empty(num_experts, in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(num_experts, hid_dim, in_dim))
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)
        self.act = nn.GELU() if if_act else nn.Identity()
        self.use_bias = use_bias
        if self.use_bias:
            self.bias_a = nn.Parameter(torch.empty(num_experts, hid_dim))
            self.bias_b = nn.Parameter(torch.empty(num_experts, in_dim))
        else:
            self.register_parameter("bias", None)

        for n, p in self.named_parameters():
            nn.init.xavier_normal_(p, gain=math.sqrt(2))             

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.unsqueeze(1).unsqueeze(0) # (num_experts, output_dim) -> (1, num_experts, output_dim) -> (1, num_experts, 1, output_dim)
            x = torch.matmul(self.drop1(self.act(x)), self.lora_b) + self.bias_b.unsqueeze(1).unsqueeze(0) # (num_experts, output_dim) -> (1, num_experts, output_dim) -> (1, num_experts, 1, output_dim)
            return self.drop2(x)
        else:
            return self.drop2(self.drop1(self.act((x @ self.lora_a))) @ self.lora_b)

class HMoE(nn.Module):
    def __init__(self, dim, experts, slots, hid_dim, if_norm=True):
        super().__init__()
        self.size_slots = slots
        self.size_experts = experts
        thi = torch.empty(dim // slots, experts*slots)         
        std = math.sqrt(1.0 / thi.size(0))         
        thi = nn.init.normal_(thi, mean=0, std=std)
        self.norm = nn.LayerNorm(dim) if if_norm else nn.Identity()
        self.gate_thi = nn.Parameter(thi)
        self.linear1 = LoRP(hid_dim=hid_dim)
        # self.experts = nn.ModuleList([CAdapter(in_dim= dim // slots, out_dim=dim // slots, hid_dim=hid_dim) for _ in range(experts)])
        self.experts = MultiExpertLinear(experts, dim // slots, hid_dim, 0.1, 0.1, use_bias=True)
        self.D_temp = nn.Parameter(torch.zeros(1)+1.0)
        self.C_temp = nn.Parameter(torch.zeros(1)+1.0)
        self.linear2 = LoRP(hid_dim=hid_dim)
        self.router_stats = {}

    def forward(self, x, mode=None):
        B, N, D = x.shape
        self.router_stats = {}
        
        thi = self.gate_thi.unsqueeze(0).expand(B, -1, -1)
        
        x = self.linear1(self.norm(x)).reshape(B, N*self.size_slots, D//self.size_slots)

        logits = torch.bmm(x, thi) # (B, N*slots, D//slots) x (B, D//slots, E*S) -> (B, N*slots, E*S)
        Dispatch = F.softmax(logits/self.D_temp, dim=1) # (B, N*slots, E*S)列向量是一个sequence的所有token关于一个slot的得分，所有token根据列向量的值做加权和，作为该slot的输入
        Combine = logits.reshape(B, N, self.size_slots, self.size_slots*self.size_experts).sum(dim=2).reshape(B, N, self.size_experts, self.size_slots).sum(dim=-1) # (B, N, E)
        Combine = F.softmax(Combine/self.C_temp, dim=-1) # (B, N, E)行向量是一个token对一个expert的贡献度，所有expert的输出根据贡献度加权和得到该token的最终输出
        with torch.no_grad():
            combine_detached = Combine.detach().clamp_min(1e-9)
            entropy = -(combine_detached * combine_detached.log()).sum(dim=-1)
            entropy = entropy / math.log(max(self.size_experts, 2))
            expert_load = combine_detached.mean(dim=(0, 1))
            self.router_stats = {
                "Router/entropy": entropy.mean(),
                "Router/expert_load_max": expert_load.max(),
                "Router/expert_load_std": expert_load.std(unbiased=False),
            }

        experts_inputs = torch.bmm(Dispatch.transpose(1, 2), x).reshape(B, self.size_experts, self.size_slots, D//self.size_slots) # (B, E*S, D) -> (B, E, S, D)
        # experts_outputs = torch.stack([self.experts[i](experts_inputs[:, i]) for i in range(self.size_experts)], dim=1).reshape(B, self.size_experts, -1) # (B, E, D)
        experts_outputs = self.experts(experts_inputs).reshape(B,self.size_experts, -1)
        experts_outputs = self.linear2(experts_outputs)
        moe_out = torch.bmm(Combine, experts_outputs) # (B, N, D)
        return moe_out


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))) # (r*n, dim)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)) # ()
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.xavier_normal_(self.lora_A, gain=math.sqrt(2))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self, mode=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        '''
        input (batch, in_channels, seq_len)
        weight (total_out_channels, filter_channels_per_group=in_channels // groups, kernel_size)
        groups
        requires：
        input.shape[1] % groups == 0 && weight.shape[0] == weight.shape[1] * groups
        '''
        if mode is not None:
            with torch.no_grad():
                delta_w = F.conv1d(
                                    self.lora_A.unsqueeze(0), # (1, 16, 768)
                                    self.lora_B.unsqueeze(-1), # (1536, 8, 1)
                                    groups=sum(self.enable_lora) # 2
                                ).squeeze(0)
        else:
            delta_w = F.conv1d(
                                    self.lora_A.unsqueeze(0), # (1, 16, 768)
                                    self.lora_B.unsqueeze(-1), # (1536, 8, 1)
                                    groups=sum(self.enable_lora) # 2
                                ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB(mode) #* self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB(mode) #* self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x @ T(self.merge_AB().T)) #* self.scaling
            return result

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14, layer=None, lora_layers=[], amglora_rank=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        if layer in lora_layers:
            self.qkv = MergedLinear(dim, dim * 3, r=amglora_rank, lora_alpha=1, lora_dropout=0.1,
                                bias=qkv_bias, enable_lora=[False, True, True])
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, H, N, D//H=64)
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, N, N)
        unnorm_attn = attn

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),) # mask (B, N) -> (B, 1, 1, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        if return_attention:
            return x, attn, unnorm_attn
        else:
            return x

class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
