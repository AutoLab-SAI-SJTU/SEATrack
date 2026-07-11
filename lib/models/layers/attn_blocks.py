import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.timm_compat import DropPath, Mlp, lecun_normal_, trunc_normal_
from lib.models.layers.attn import Attention, LoRP, HMoE
from lib.models.layers.bilift import BiLift
from lib.models.seatrack.utils import combine_tokens, recover_tokens, token2feature, feature2token


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    '''
    math.ceil() 是一个Python标准库中的函数
    返回大于或等于所传参数的最小整数,它将参数向上取整到最接近的整数,如果参数已经是整数，则返回该整数。
    '''
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    '''
    (B, 12, 64, 256)
    取的是templates关于search region的attention权重
    '''
    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        '''
        对于central token的情况：
        unsqueeze(1): (B, 64) -> (B, 1, 64)
        unsqueeze(-1): (B, 1, 64) -> (B, 1, 64, 1)
        expand:(B, 1, 64, 1) -> (B, 12, 64, 256)
        扩充仍然从最右侧dim3开始，(64,1)->(64,256)，然后是dim1，将(64,256)看作整体进行复制
        其结果是：生成了关于attn_t的掩码矩阵，对于每个头的central token行，其掩码全为true
        '''
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1]) # (B, 1 64, 1) -> (B, 12, 64, 256)
        # attn_t = attn_t[:, :, box_mask_z, :]
        '''
        进行bool掩码索引，只返回attn_t中对应位置在box_mask_z中值为true的元素
        返回每个头的central token关于search region的attention权重
        (B, 12, 1, 256)
        '''
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        '''
        对于mean操作，返回一个降维tensor，指定的dim被消除
        ***对于指定了dim的操作，其操作单位是指定dim右边dim构成的整体张量***的attention权重
        (B, 12, 1, 256)
        对于mean操作，返回一个降维tensor，指定的dim被消除
        ***对于指定了dim的操作，其操作单位是指定dim右边dim构成的整体张量***
        mean(dim=2):(1, 12, 1, 256) -> (1, 12, 256)
        mean(dim=1):(1, 12, 256) -> (1, 256)
        将所有头中central token的attention求均值作为central token的最终相似度
        '''
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    '''
    将searchs与central token的相似度按降序排列，返回排序结果sorted_attn及其原索引indices
    '''
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    '''
    分别按顺序记录保留的token索引和删除的token索引
    '''
    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    '''
    top_idx -> (1, 180)
    unsqueeze(-1) -> (1, 180, 1)，索引由行向量变为列向量
    expand(B, -1, C) -> (1, 180, 768)=index.shanpe=output.shape，每行的元素都相同（同一个索引）
    output[i][j][k] = tokens[i][index[i][j][k]][k]
    通过gather来获取tokens中index对应的那些token值
    '''
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    '''
    tokens_new就是要送到下一层encoder的新tokens
    '''
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index

class CEBlock_AP(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,
                 layer=None, lora_layers=[], moe_layers=[], amglora_rank=None, hmoe_rank=None,
                 gra_enabled=False, gra_diagnostics=False, gra_layers=None, gra_rgae_enabled=True,
                 gra_rho_min=0.1, gra_detach_rho=False, amg_enabled=True, hmoe_enabled=True,
                 bilift_enabled=False, bilift_rank=8, bilift_reverse=False,
                 bilift_dropout=0.0, bilift_diagnostics=False):
        super().__init__()
        amglora_rank = 8 if amglora_rank is None else amglora_rank
        hmoe_rank = 4 if hmoe_rank is None else hmoe_rank
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop, layer=layer, lora_layers=lora_layers,
                              amglora_rank=amglora_rank)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search
        self.layer = layer
        self.lora_layers = lora_layers
        self.moe_layers = moe_layers
        self.amg_enabled = amg_enabled
        self.hmoe_enabled = hmoe_enabled
        self.hmoe_active = self.hmoe_enabled and layer in self.moe_layers
        self.bilift_active = bilift_enabled
        self.gra_enabled = gra_enabled
        self.gra_diagnostics = gra_diagnostics
        self.gra_layers = list(lora_layers if gra_layers is None else gra_layers)
        self.gra_active = (self.gra_enabled or self.gra_diagnostics) and layer in self.gra_layers
        self.gra_rgae_enabled = gra_rgae_enabled
        self.gra_rho_min = gra_rho_min
        self.gra_detach_rho = gra_detach_rho
        self.gratrack_stats = {}
        self.bilift_stats = {}

        if self.amg_enabled and layer in lora_layers:
            self.r2dte_scaling = nn.Parameter(torch.zeros(1) + 1)
            self.dte2r_scaling = nn.Parameter(torch.zeros(1) + 1)
        if self.gra_active and self.gra_enabled and self.gra_rgae_enabled:
            self.rgae_x2r_scaling = nn.Parameter(torch.ones(1, num_heads, 1, 1))
            self.rgae_r2x_scaling = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        if self.hmoe_active:
            self.attn_moe = HMoE(dim, 4, 2, hmoe_rank)
            self.ffn_moe = HMoE(dim, 8, 2, hmoe_rank)
        if self.bilift_active:
            self.bilift = BiLift(
                dim=dim,
                rank=bilift_rank,
                dropout=bilift_dropout,
                reverse=bilift_reverse,
                diagnostics=bilift_diagnostics,
            )

    def cal_qkv(self, x, layer=None):
        B, N, C = x.shape
        qkv = self.attn.qkv(x) 
        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return (q @ k.transpose(-2, -1)) * self.attn.scale, v
    
    def amglora_attn(self, attn, v, shape, guidance=None, mode=None, cls_token=None, layer=None):
        B, N, C = shape
        if mode == 'r2dte':
            attn = attn + self.r2dte_scaling*(guidance - attn)

        elif mode == 'dte2r':
            attn = attn + self.dte2r_scaling*(guidance - attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.attn.proj_drop(self.attn.proj(output)), attn

    def compute_gra_stats(self, attn_logits, lens_t, eps=1e-6):
        attn = attn_logits.softmax(dim=-1)
        zs = attn[:, :, :lens_t, lens_t:]
        response = zs.mean(dim=2)
        prob = response / response.sum(dim=-1, keepdim=True).clamp_min(eps)

        lens_s = prob.shape[-1]
        confidence = (lens_s * (prob ** 2).sum(dim=-1) - 1.0) / max(lens_s - 1, 1)
        confidence = confidence.clamp(0.0, 1.0)
        prob_mean = prob.mean(dim=1)
        return {
            "prob": prob,
            "prob_mean": prob_mean,
            "confidence": confidence,
        }

    def compute_cross_response_gate(self, rgb_stats, x_stats, eps=1e-6):
        p_r = rgb_stats["prob_mean"]
        p_x = x_stats["prob_mean"]
        lens_s = p_r.shape[-1]
        agreement = (lens_s * (p_r * p_x).sum(dim=-1) - 1.0) / max(lens_s - 1, 1)
        agreement = agreement.clamp(0.0, 1.0)

        c_r = rgb_stats["confidence"].mean(dim=1)
        c_x = x_stats["confidence"].mean(dim=1)
        rho_raw = agreement * torch.sqrt((c_r * c_x).clamp_min(eps))
        rho = self.gra_rho_min + (1.0 - self.gra_rho_min) * rho_raw
        return {
            "rho": rho,
            "rho_raw": rho_raw,
            "agreement": agreement,
            "c_rgb": c_r,
            "c_x": c_x,
            "p_rgb": p_r,
            "p_x": p_x,
        }

    def response_gated_attention(self, self_logits, other_logits, v, shape, rho, mode):
        B, N, C = shape
        if self.gra_detach_rho:
            rho = rho.detach()

        if mode == "x2r":
            scale = torch.sigmoid(self.rgae_x2r_scaling)
        elif mode == "r2x":
            scale = torch.sigmoid(self.rgae_r2x_scaling)
        else:
            raise ValueError("Unsupported RGAE mode: {}".format(mode))

        gate = rho[:, None, None, None] * scale
        logits = self_logits + gate * (other_logits - self_logits)
        attn = logits.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        output = self.attn.proj_drop(self.attn.proj(output))
        return output, attn, gate

    def _record_gra_stats(self, gate, gate_x2r=None, gate_r2x=None):
        stats = {
            "GRA/rho_mean": gate["rho"].detach().mean(),
            "GRA/rho_raw_mean": gate["rho_raw"].detach().mean(),
            "GRA/agreement_mean": gate["agreement"].detach().mean(),
            "GRA/c_rgb_mean": gate["c_rgb"].detach().mean(),
            "GRA/c_x_mean": gate["c_x"].detach().mean(),
        }
        if gate_x2r is not None:
            stats["Gate/x2r_mean"] = gate_x2r.detach().mean()
        if gate_r2x is not None:
            stats["Gate/r2x_mean"] = gate_r2x.detach().mean()
        self.gratrack_stats.update(stats)

    def _record_router_stats(self, prefix, module):
        if not self.gra_active:
            return
        for name, value in getattr(module, "router_stats", {}).items():
            self.gratrack_stats["{}/{}".format(prefix, name)] = value.detach().mean()

    def forward(self, x, global_index_template, global_search_idx, mask=None, ce_template_mask=None, keep_ratio_search=None):
        self.gratrack_stats = {}
        self.bilift_stats = {}
        lens_t = global_index_template.shape[1]
        lens_x = global_search_idx[0].shape[1]

        # AMG-LoRA for Attention maps alignment
        brgb_attn, rgb_v = self.cal_qkv(self.norm1(x[0]), self.layer)
        bdte_attn, dte_v = self.cal_qkv(self.norm1(x[1]), self.layer)

        if self.gra_active:
            rgb_stats = self.compute_gra_stats(brgb_attn, lens_t)
            x_stats = self.compute_gra_stats(bdte_attn, lens_t)
            gate = self.compute_cross_response_gate(rgb_stats, x_stats)

            if self.gra_enabled and self.gra_rgae_enabled:
                xrgb_attn, _, gate_x2r = self.response_gated_attention(
                    brgb_attn, bdte_attn, rgb_v, x[0].shape, gate["rho"], mode="x2r")
                xdte_attn, _, gate_r2x = self.response_gated_attention(
                    bdte_attn, brgb_attn, dte_v, x[1].shape, gate["rho"], mode="r2x")
                self._record_gra_stats(gate, gate_x2r=gate_x2r, gate_r2x=gate_r2x)
            elif self.amg_enabled and self.layer in self.lora_layers:
                xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape, guidance=bdte_attn, mode='dte2r', layer=self.layer)
                xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape, guidance=brgb_attn, mode='r2dte', layer=self.layer)
                self._record_gra_stats(gate)
            else:
                xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape)
                xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape)
                self._record_gra_stats(gate)
        elif self.amg_enabled and self.layer in self.lora_layers:
            xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape, guidance=bdte_attn, mode='dte2r', layer=self.layer)
            xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape, guidance=brgb_attn, mode='r2dte', layer=self.layer)
        else:
            xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape)
            xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape)

        x[0] = x[0] + self.drop_path(xrgb_attn)
        x[1] = x[1] + self.drop_path(xdte_attn)

        if self.bilift_active:
            x[0], x[1] = self.bilift(x[0], x[1])
            self.bilift_stats = dict(self.bilift.last_stats)

        # HMoE for cross template and search region fusion
        if self.hmoe_active:
            mix_z = self.attn_moe(torch.cat([x[0][:, :lens_t], x[1][:, :lens_t]], dim=1), mode = 'template')
            self._record_router_stats("AttnMoE/template", self.attn_moe)
            mix_x = self.attn_moe(torch.cat([x[0][:, lens_t:], x[1][:, lens_t:]], dim=1), mode = 'search')
            self._record_router_stats("AttnMoE/search", self.attn_moe)
            x[0] = x[0] + self.drop_path(torch.cat([mix_z[:, :lens_t], mix_x[:, :lens_x]], dim=1))
            x[1] = x[1] + self.drop_path(torch.cat([mix_z[:, lens_t:], mix_x[:, lens_x:]], dim=1))
        
        removed_rgbsearch_idx = None
        removed_dtesearch_idx = None

        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x[0], global_search_idx[0], removed_rgbsearch_idx = candidate_elimination(x[0], x[0], lens_t, keep_ratio_search, global_search_idx[0], ce_template_mask)
            x[1], global_search_idx[1], removed_dtesearch_idx = candidate_elimination(x[1], x[1], lens_t, keep_ratio_search, global_search_idx[1], ce_template_mask)
            lens_x = global_search_idx[0].shape[1]

        x[0] = x[0] + self.drop_path(self.mlp(self.norm2(x[0]))) 
        x[1] = x[1] + self.drop_path(self.mlp(self.norm2(x[1]))) 

        # HMoE for cross template and search region fusion
        if self.hmoe_active:
            mix_z = self.ffn_moe(torch.cat([x[0][:, :lens_t], x[1][:, :lens_t]], dim=1), mode = 'template')
            self._record_router_stats("FfnMoE/template", self.ffn_moe)
            mix_x = self.ffn_moe(torch.cat([x[0][:, lens_t:], x[1][:, lens_t:]], dim=1), mode = 'search')
            self._record_router_stats("FfnMoE/search", self.ffn_moe)
            x[0] = x[0] + self.drop_path(torch.cat([mix_z[:, :lens_t], mix_x[:, :lens_x]], dim=1))
            x[1] = x[1] + self.drop_path(torch.cat([mix_z[:, lens_t:], mix_x[:, lens_x:]], dim=1))

        return x, global_index_template, global_search_idx, [removed_rgbsearch_idx, removed_dtesearch_idx]

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
