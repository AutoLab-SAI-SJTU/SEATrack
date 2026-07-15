---
title: "GRATrack Formula and SEATrack Code Integration Plan"
translated_title: "GRATrack 公式与 SEATrack 代码接入方案"
type: "implementation-design"
topic: "多模态目标跟踪"
created: 2026-07-09
status: "proposal"
source_repo: ".tmp/SEATrack-src"
---

# GRATrack 公式与 SEATrack 代码接入方案

## 1. 目标与边界

本文档面向 **GRATrack: Gini Response Agreement Guided Sparse Expert Routing for Efficient RGB-X Tracking** 的代码落地。目标是在本地 SEATrack 官方源码结构上设计一个可逐步替换/增强的实现路线，使方法满足四个约束：

1. **紧贴 tracking**：所有 gate、trust、token sparse mask 都来自 template-search response，而不是全局图像质量或分类 token。
2. **不新增大 cross-attention**：不引入 RGB-X token-level dense cross-attention，只复用已有 attention logits/output 做轻量门控。
3. **保持 PEFT 兼容**：冻结主干和 head，只训练 LoRA/MoE/scaling/GRA gate/router bias/sparse weights 等小参数。
4. **计算量持平甚至降低**：V0 只加轻量统计；V1 只加 router bias；V2 用 response-aware token sparsification 和 top-k expert routing 抵消新增开销。

当前本地代码仓库：

```text
E:\Code\research-codex\daily_paper\.tmp\SEATrack-src
```

本轮仅做静态源码设计，不直接修改 SEATrack 源码。

## 2. 当前源码关键事实

### 2.1 文件与职责

| 文件 | 当前职责 | GRATrack 接入建议 |
|---|---|---|
| `.tmp/SEATrack-src/lib/models/layers/attn_blocks.py` | `CEBlock_AP` 串联双流 attention、AMG-like mutual guidance、HMoE、candidate elimination | 核心接入文件。新增 GRA 统计、RGAE gated exchange、response-aware sparsification；在 `CEBlock_AP.forward()` 中组织数据流 |
| `.tmp/SEATrack-src/lib/models/layers/attn.py` | `Attention`、`MultiExpertLinear`、`HMoE`、LoRA/router/expert 实现 | 改造 `HMoE.forward()` 支持 trust bias、top-k expert、可选 token mask |
| `.tmp/SEATrack-src/lib/models/seatrack/vit_ci.py` | 双流 RGB/DTE backbone，token 拼接，block 循环，裁剪恢复 | 传入 GRA 配置；维护 `global_index_s`、`removed_indexes_s`；保证 sparsification 后恢复原 search 网格 |
| `.tmp/SEATrack-src/lib/models/seatrack/seatrack.py` | `build_seatrack()` 和 head wrapper | 从 cfg 读取 GRA 参数并传给 backbone；不改 `forward_head()` |
| `.tmp/SEATrack-src/lib/train/base_functions.py` | PEFT optimizer 参数过滤 | 扩展 trainable name 白名单：`gra/rgae/gate/trust/router_bias/sparse` |
| `.tmp/SEATrack-src/lib/train/actors/seatrack.py` | tracking loss 入口 | 主损失不改；可把 GRA 诊断量加入 `status` 或 `aux_dict` 用于 ablation |

### 2.2 当前 `CEBlock_AP` 数据流

`attn_blocks.py::CEBlock_AP` 中关键结构：

```python
brgb_attn, rgb_v = self.cal_qkv(self.norm1(x[0]), self.layer)
bdte_attn, dte_v = self.cal_qkv(self.norm1(x[1]), self.layer)
xrgb_attn, _ = self.amglora_attn(
    brgb_attn, rgb_v, x[0].shape,
    guidance=bdte_attn, mode='dte2r', layer=self.layer
)
xdte_attn, _ = self.amglora_attn(
    bdte_attn, dte_v, x[1].shape,
    guidance=brgb_attn, mode='r2dte', layer=self.layer
)
```

`cal_qkv()` 返回未 softmax 的 attention logits 和 value：

```python
qkv = self.attn.qkv(x)
qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads)
qkv = qkv.permute(2, 0, 3, 1, 4)
q, k, v = qkv.unbind(0)
return (q @ k.transpose(-2, -1)) * self.attn.scale, v
```

这对 GRA 很关键：当前实现已经显式 materialize `[B, H, N, N]` logits，因此 V0 可以直接从 `brgb_attn/bdte_attn` 的 `template -> search` 子矩阵计算 response，不需要新增 `QK^T`。

### 2.3 当前 HMoE 数据流

`attn.py::HMoE.forward()` 中 router 结构：

```python
x = self.linear1(self.norm(x)).reshape(B, N*self.size_slots, D//self.size_slots)
logits = torch.bmm(x, thi)
Dispatch = F.softmax(logits/self.D_temp, dim=1)
Combine = logits.reshape(B, N, self.size_slots, self.size_slots*self.size_experts)
Combine = Combine.sum(dim=2).reshape(B, N, self.size_experts, self.size_slots).sum(dim=-1)
Combine = F.softmax(Combine/self.C_temp, dim=-1)
experts_inputs = torch.bmm(Dispatch.transpose(1, 2), x)
experts_outputs = self.experts(experts_inputs).reshape(B, self.size_experts, -1)
moe_out = torch.bmm(Combine, experts_outputs)
```

最小改动点是：在 `Combine` softmax 前加 response/trust bias，再可选 top-k mask。

### 2.4 当前 PEFT 参数过滤

`base_functions.py::get_optimizer_scheduler()` 中：

```python
if cfg.TRAIN.PEFT:
    trainable_params = ['moe', 'scaling', 'lora']
```

新增参数如果要在 PEFT 下训练，参数名必须进入白名单。建议扩展为：

```python
trainable_params = [
    'moe', 'scaling', 'lora',
    'gra', 'rgae', 'gate', 'trust', 'router_bias', 'sparse'
]
```

注意：`gate` 会捕获 HMoE 的 `gate_thi`，这通常可接受，因为 MoE 本来就在 PEFT 可训练范围中；但不要把普通 backbone 参数命名成含 `gate/trust/sparse` 的名字。

## 3. 统一符号

设：

- batch size: \(B\)
- head 数: \(H\)
- template token 数: \(N_z\)
- search token 数: \(N_s\)
- 总 token 数: \(N=N_z+N_s\)
- hidden dim: \(C\)
- head dim: \(d=C/H\)
- 模态流：\(m \in \{r,x\}\)，其中 \(r\) 表示 RGB，\(x\) 表示 auxiliary modality，如 thermal/depth/event
- 第 \(l\) 层 attention logits：\(S_{m,l}\in\mathbb{R}^{B\times H\times N\times N}\)
- softmax 后 attention：\(A_{m,l}=\mathrm{softmax}(S_{m,l}, \mathrm{dim}=-1)\)
- template-search attention block：\(A_{m,l}^{zs}=A_{m,l}[:,:, :N_z, N_z:]\)

当前 `CAT_MODE='direct'` 时 token 顺序为 `[template, search]`。GRA 的切片依赖这个假设。

## 4. 模块公式

### 4.1 Gini Response Agreement (GRA)

从 template-to-search attention 中提取 response：

\[
p_{m,l}(j)=
\frac{1}{HN_z}
\sum_{h=1}^{H}
\sum_{i=1}^{N_z}
A_{m,l}^{h,zs}(i,j).
\]

归一化：

\[
\bar p_{m,l}(j)=
\frac{p_{m,l}(j)}
{\sum_{k=1}^{N_s}p_{m,l}(k)+\epsilon}.
\]

单模态响应集中度：

\[
c_{m,l}=
\left[
\frac{N_s\|\bar p_{m,l}\|_2^2-1}{N_s-1}
\right]_0^1.
\]

跨模态区域一致性：

\[
a_l=
\left[
\frac{
N_s\bar p_{r,l}^{\top}\mathcal{P}_\sigma(\bar p_{x,l})-1
}
{N_s-1}
\right]_0^1.
\]

其中 \(\mathcal{P}_\sigma\) 可以是 1D/2D 平滑、窗口池化或轻量 spatial tolerance，使 RGB/X 响应不必逐 token 完全一致，只需目标区域一致。

最终 gate：

\[
\rho_l=a_l\sqrt{c_{r,l}c_{x,l}}.
\]

解释：

- \(c_r,c_x\) 回答“各自是否看得清”。
- \(a_l\) 回答“两者是否看向同一区域”。
- \(\rho_l\) 回答“当前是否适合强互导/强融合”。

### 4.2 Response-Gated Attention Exchange (RGAE)

当前 logits exchange 可抽象为：

\[
S_r' = S_r + \alpha_{x\rightarrow r}(S_x-S_r),
\]

\[
S_x' = S_x + \alpha_{r\rightarrow x}(S_r-S_x).
\]

RGAE 将固定或静态的 \(\alpha\) 替换为 response-conditioned gate：

\[
S_r' = S_r + \rho_l\lambda_{x\rightarrow r,l}(S_x-S_r),
\]

\[
S_x' = S_x + \rho_l\lambda_{r\rightarrow x,l}(S_r-S_x).
\]

其中 \(\lambda_{x\rightarrow r,l}\)、\(\lambda_{r\rightarrow x,l}\) 可为：

1. V0: 每层标量参数；
2. V0-b: 每层每 head 参数；
3. V1: 由 \(q_l=[c_r,c_x,\rho,c_r-c_x]\) 预测的方向性 gate。

V0 建议先用 per-head gate：

\[
\lambda_{x\rightarrow r,l,h}=\sigma(b_{x\rightarrow r,l,h}),
\quad
\lambda_{r\rightarrow x,l,h}=\sigma(b_{r\rightarrow x,l,h}).
\]

### 4.3 Trust-Biased Sparse MoE (TB-SMoE)

构造路由状态：

\[
q_l=[c_{r,l},c_{x,l},\rho_l,c_{r,l}-c_{x,l}].
\]

给 HMoE 的 combine logits 加 expert-specific bias：

\[
z'_{l,e}=z_{l,e}+u_e^\top q_l.
\]

稀疏专家激活：

\[
\pi_l=\mathrm{TopK\text{-}Softmax}(z_l'),
\]

\[
Y_l=
\sum_{e\in\mathrm{TopK}(\pi_l)}
\pi_{l,e}E_e(\tilde X_l).
\]

注意：如果只加一个全局 scalar bias，softmax 后不会改变 expert 分布；必须是 **expert-specific bias**，即 \(u_e\) 对不同专家不同。

### 4.4 Response-Aware Token Sparsification (RATS)

融合双流 response：

\[
p_{f,l}
=
w_r c_{r,l}\bar p_{r,l}
+w_x c_{x,l}\bar p_{x,l}
+w_a \min(\bar p_{r,l},\bar p_{x,l}).
\]

保留高响应 search token：

\[
M_l=\mathrm{TopK}(p_{f,l},K_s),
\quad
K_s=\lceil \alpha N_s\rceil.
\]

专家计算只作用于 \(M_l\)：

\[
Y_{l,M_l}
=
\sum_{e\in\mathrm{TopK}(\pi_l)}
\pi_{l,e}E_e(\tilde X_{l,M_l}),
\]

低响应 token 走残差：

\[
Y_{l,\bar M_l}=\tilde X_{l,\bar M_l}.
\]

默认建议 RGB/X 共享 `keep_idx`，避免两流 token 数不一致导致 HMoE 和最终 fusion 复杂化。

## 5. 伪代码

### 5.1 `compute_gra_stats`

```python
def compute_gra_stats(attn_logits, lens_t, smooth=None, eps=1e-6):
    # attn_logits: [B, H, N, N], raw logits
    # lens_t: template token count
    # return:
    #   prob: [B, H, Ns]
    #   prob_mean: [B, Ns]
    #   confidence: [B, H]

    B, H, N, _ = attn_logits.shape
    attn = attn_logits.softmax(dim=-1)
    zs = attn[:, :, :lens_t, lens_t:]        # [B,H,Nz,Ns]

    response = zs.mean(dim=2)                # [B,H,Ns]
    prob = response / response.sum(dim=-1, keepdim=True).clamp_min(eps)

    Ns = prob.shape[-1]
    confidence = (Ns * (prob ** 2).sum(dim=-1) - 1.0) / max(Ns - 1, 1)
    confidence = confidence.clamp(0.0, 1.0)  # [B,H]

    prob_mean = prob.mean(dim=1)             # [B,Ns]
    if smooth is not None:
        prob_for_agree = smooth(prob_mean)
    else:
        prob_for_agree = prob_mean

    return prob, prob_mean, confidence, prob_for_agree
```

### 5.2 `compute_cross_response_gate`

```python
def compute_cross_response_gate(rgb_stats, x_stats, eps=1e-6):
    # rgb_stats/x_stats: outputs of compute_gra_stats
    p_r, p_r_mean, c_r, p_r_smooth = rgb_stats
    p_x, p_x_mean, c_x, p_x_smooth = x_stats

    Ns = p_r_mean.shape[-1]
    # region-level agreement, [B]
    agreement = (Ns * (p_r_mean * p_x_smooth).sum(dim=-1) - 1.0) / max(Ns - 1, 1)
    agreement = agreement.clamp(0.0, 1.0)

    # head-averaged confidence, [B]
    c_r_bar = c_r.mean(dim=1)
    c_x_bar = c_x.mean(dim=1)

    rho = agreement * torch.sqrt((c_r_bar * c_x_bar).clamp_min(eps))
    return {
        "rho": rho,                         # [B]
        "agreement": agreement,             # [B]
        "c_r": c_r_bar,                     # [B]
        "c_x": c_x_bar,                     # [B]
        "p_r": p_r_mean,                    # [B,Ns]
        "p_x": p_x_mean,                    # [B,Ns]
    }
```

### 5.3 `response_gated_attention`

```python
def response_gated_attention(self, self_logits, other_logits, v, shape, rho, mode):
    # self_logits/other_logits: [B,H,N,N]
    # rho: [B]
    # mode: 'x2r' or 'r2x'
    B, N, C = shape

    if mode == "x2r":
        scale = self.rgae_x2r_scaling
    else:
        scale = self.rgae_r2x_scaling

    gate = rho[:, None, None, None] * scale
    logits = self_logits + gate * (other_logits - self_logits)

    attn = logits.softmax(dim=-1)
    attn = self.attn.attn_drop(attn)
    output = (attn @ v).transpose(1, 2).reshape(B, N, C)
    output = self.attn.proj_drop(self.attn.proj(output))
    return output, attn
```

### 5.4 `trust_bias_router`

```python
def trust_bias_router(combine_logits, q, router_bias):
    # combine_logits: [B,N,E]
    # q: [B,4] = [c_r,c_x,rho,c_r-c_x]
    # router_bias: [E,4]
    # output: biased logits [B,N,E]
    bias = torch.einsum("bf,ef->be", q, router_bias)  # [B,E]
    return combine_logits + bias[:, None, :]
```

### 5.5 `build_response_token_mask`

```python
def build_response_token_mask(p_r, p_x, c_r, c_x, keep_ratio):
    # p_r/p_x: [B,Ns], c_r/c_x: [B]
    score = c_r[:, None] * p_r + c_x[:, None] * p_x + torch.minimum(p_r, p_x)
    K = math.ceil(keep_ratio * score.shape[-1])
    keep = torch.topk(score, k=K, dim=-1).indices
    return keep, score
```

## 6. 具体源码改造方案

### 6.1 `attn_blocks.py::CEBlock_AP.__init__`

建议新增参数：

```python
gra_enabled=False,
gra_layers=[],
gra_use_proxy=False,
gra_smooth_kernel=3,
gra_sparse_keep_ratio=1.0,
gra_moe_topk=None,
```

建议新增可训练参数：

```python
if layer in gra_layers:
    self.gra_enabled = True
    self.rgae_x2r_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))
    self.rgae_r2x_scaling = nn.Parameter(torch.ones(num_heads, 1, 1))
    self.gra_router_bias = nn.Parameter(torch.zeros(num_experts, 4))  # 若本层接 MoE
    self.sparse_w_rgb = nn.Parameter(torch.ones(1))
    self.sparse_w_x = nn.Parameter(torch.ones(1))
    self.sparse_w_agree = nn.Parameter(torch.ones(1))
```

命名原则：

- `gra_*`：GRA 统计与 router bias；
- `rgae_*`：attention exchange gate；
- `sparse_*`：response-aware token sparsification；
- `*_scaling`：保留 PEFT 白名单兼容。

### 6.2 `attn_blocks.py::CEBlock_AP.forward`

V0 推荐流程：

```python
brgb_attn, rgb_v = self.cal_qkv(self.norm1(x[0]), self.layer)
bdte_attn, dte_v = self.cal_qkv(self.norm1(x[1]), self.layer)

if self.layer in self.gra_layers:
    rgb_stats = self.compute_gra_stats(brgb_attn, lens_t)
    x_stats = self.compute_gra_stats(bdte_attn, lens_t)
    gate = self.compute_cross_response_gate(rgb_stats, x_stats)

    xrgb_attn, rgb_attn_map = self.response_gated_attention(
        brgb_attn, bdte_attn, rgb_v, x[0].shape, gate["rho"], mode="x2r"
    )
    xdte_attn, x_attn_map = self.response_gated_attention(
        bdte_attn, brgb_attn, dte_v, x[1].shape, gate["rho"], mode="r2x"
    )
else:
    xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape)
    xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape)
```

注意：

- 第一版不要保存完整 `attn_map` 到 `aux_dict`，只保存 `rho/agreement/c_r/c_x` 的 detach 值。
- V0 不改变 token 长度，风险最低。
- 如果 `rho` 过小导致退化，可用 residual floor：

\[
\rho'=\rho_{\min}+(1-\rho_{\min})\rho.
\]

例如 `rho_min=0.1`。

### 6.3 `attn.py::HMoE.forward`

当前签名：

```python
def forward(self, x, mode=None):
```

建议 V1 改为：

```python
def forward(self, x, mode=None, gra_q=None, token_response=None,
            token_mask=None, topk=None, return_router=False):
```

其中：

- `gra_q`: `[B,4]`，即 `[c_r,c_x,rho,c_r-c_x]`
- `token_response`: `[B,N]`，可选 search response score
- `token_mask`: `[B,N]`，可选 high-response token mask
- `topk`: expert top-k

在 `Combine` softmax 前：

```python
Combine_logits = ...
if gra_q is not None:
    bias = torch.einsum("bf,ef->be", gra_q, self.gra_router_bias)
    Combine_logits = Combine_logits + bias[:, None, :]

if topk is not None:
    topk_idx = Combine_logits.topk(topk, dim=-1).indices
    mask = torch.zeros_like(Combine_logits, dtype=torch.bool)
    mask.scatter_(-1, topk_idx, True)
    Combine_logits = Combine_logits.masked_fill(~mask, float("-inf"))

Combine = F.softmax(Combine_logits / self.C_temp, dim=-1)
```

V1 可以先不开 top-k，只加 `gra_router_bias`，观察 expert load 和 router entropy。

### 6.4 `vit_ci.py`

`VisionTransformerCE.__init__()` 增加配置参数：

```python
gra_enabled=False,
gra_layers=None,
gra_use_proxy=False,
gra_sparse_keep_ratio=1.0,
gra_moe_topk=None,
```

构建 block 时传入：

```python
CEBlock_AP(
    ...,
    gra_enabled=gra_enabled,
    gra_layers=gra_layers or lora_layers,
    gra_use_proxy=gra_use_proxy,
    gra_sparse_keep_ratio=gra_sparse_keep_ratio,
    gra_moe_topk=gra_moe_topk,
)
```

`forward_features()` 末端保持：

```python
x_rgb = recover_tokens(...)
x_dte = recover_tokens(...)
x_fusion = x_rgb + x_dte
```

除非 V2 已完成 search token scatter/recovery 验证，否则不要改最终 `x_fusion`。

### 6.5 `seatrack.py::build_seatrack`

从 cfg 读取：

```python
gra_enabled = cfg.MODEL.GRA.ENABLED
gra_layers = cfg.MODEL.GRA.LAYERS
gra_sparse_keep_ratio = cfg.MODEL.GRA.SPARSE.KEEP_RATIO
gra_moe_topk = cfg.MODEL.GRA.MOE.TOPK
```

传给 `vit_base_patch16_224_ce()`。保持 `SEATrack.forward()` 和 `forward_head()` 不变。

### 6.6 配置建议

在 `lib/config/seatrack/config.py` 中增加：

```yaml
MODEL:
  GRA:
    ENABLED: true
    LAYERS: [1, 3, 5, 7, 9, 11]
    USE_PROXY: false
    RHO_MIN: 0.1
    SPARSE:
      ENABLED: false
      KEEP_RATIO: 1.0
      MIN_KEEP_RATIO: 0.5
      SHARED_MASK: true
    MOE:
      TRUST_BIAS: false
      TOPK: null
      HIGH_RESPONSE_ONLY: false
```

V0 yaml：

```yaml
MODEL:
  GRA:
    ENABLED: true
    USE_PROXY: false
    RHO_MIN: 0.1
    SPARSE:
      ENABLED: false
      KEEP_RATIO: 1.0
    MOE:
      TRUST_BIAS: false
      TOPK: null
```

V1 yaml：

```yaml
MODEL:
  GRA:
    ENABLED: true
    MOE:
      TRUST_BIAS: true
      TOPK: null
```

V2 yaml：

```yaml
MODEL:
  GRA:
    ENABLED: true
    SPARSE:
      ENABLED: true
      KEEP_RATIO: 0.5
      SHARED_MASK: true
    MOE:
      TRUST_BIAS: true
      TOPK: 1
```

## 7. PEFT 策略

当前 PEFT 只训练名字中包含 `moe/scaling/lora` 的参数。GRATrack 建议扩展：

```python
trainable_params = [
    'moe', 'scaling', 'lora',
    'gra', 'rgae', 'gate', 'trust', 'router_bias', 'sparse'
]
```

冻结策略：

- 冻结原始 backbone `qkv.weight/proj/mlp/patch_embed/pos_embed`。
- 保持 LoRA 参数可训练。
- 保持 HMoE 参数可训练。
- 新增 GRA/RGAE gate、trust bias、router bias、sparse weights 可训练。
- 默认不训练 bbox head；如果后续需要训练 head，应单独设置配置，不建议混入第一版。

诊断量如 `rho/agreement/c_r/c_x/router_entropy/expert_load` 应使用 `detach()` 存入 `aux_dict`，不要作为 `nn.Parameter`。

## 8. 三阶段实现路线

### V0: GRA + RGAE

改动：

- 在 `CEBlock_AP.forward()` 中计算 GRA response。
- 用 \(\rho_l\) 乘到双向 logits exchange 上。
- 不改 HMoE。
- 不改 token 长度。
- 不改 loss。

预期收益：

- 最小验证 GRA 是否有价值。
- shape 风险最低。
- 仍保持 PEFT。

主要风险：

- `rho` 初始过低导致跨模态互导被压死。建议加 `rho_min`。
- attention logits 显存增加诊断保存风险。不要保存完整 attention。
- 如果 gate 参与反传导致不稳定，可尝试 `rho.detach()` 做 ablation。

### V1: Trust-Biased Router

改动：

- `HMoE.forward()` 支持 `gra_q`。
- 对 `Combine` logits 加 expert-specific `gra_router_bias`。
- 先不开 top-k，只观察 router entropy 和 expert load。

预期收益：

- 专家路由更贴近 tracking response。
- 不改变 token 长度，风险仍较低。

主要风险：

- expert collapse。需要 `router_entropy/expert_load` 诊断。
- 如果 `gra_router_bias` 太强，会压制原 router。建议初始化为 0。

### V2: Response-Aware Token Sparsification

改动：

- 使用 fused response score 选 high-response search token。
- 默认 RGB/X 共享 keep index。
- 只对 high-response token 激活 MoE 或进入后续 block。
- `vit_ci.forward_features()` 末端必须 scatter 回原始 \(N_s\)。

预期收益：

- 抵消 GRA/TB-SMoE 额外计算。
- 在 MoE 较重时可能使整体速度持平甚至提升。

主要风险：

- top-k mask 过早裁掉目标 token。
- `global_index_s` 与 removed index shape 出错会破坏恢复。
- bbox head 对完整 search grid 敏感，必须断言恢复后 token 数等于 `feat_len_s`。

## 9. 复杂度分析

### 9.1 V0 新增开销

使用已 materialized attention logits：

\[
O(L_g B H N_z N_s).
\]

当前 `cal_qkv()` 已经计算完整：

\[
O(BH(N_z+N_s)^2d).
\]

因此 GRA 统计不是主开销，真正风险是 attention logits 的显存访问和是否保存完整 attention。

如果使用 pooled query proxy：

\[
\bar q_{m,l}^h=
\frac{1}{N_z}
\sum_{i\in Z}q_{m,l,i}^h,
\]

\[
s_{m,l}(j)=
\frac{1}{H}
\sum_h
\frac{
\bar q_{m,l}^h(k_{m,l,j}^h)^\top
}{\sqrt d},
\]

\[
\bar p_{m,l}=\mathrm{softmax}(s_{m,l}).
\]

则开销接近：

\[
O(L_gBH N_s d).
\]

源码当前已显式算 logits，V0 先用 logits 版本更直接；若显存或速度不稳，再切换 `gra_use_proxy=True`。

### 9.2 V1 新增开销

router bias：

\[
O(BE\cdot 4)
\]

并广播到 \([B,N,E]\)。相比 HMoE 的 expert 前向非常小。

### 9.3 V2 抵消开销

原 MoE 近似：

\[
O(L_gK(N_z+N_s)Dd_{ff}).
\]

使用 top-k expert 和保留比例 \(\alpha\) 后：

\[
O(L_gK'(N_z+\alpha N_s)Dd_{ff}).
\]

若 \(K=2,K'=1,\alpha=0.5,N_z=64,N_s=256\)，MoE 部分相对成本：

\[
\frac{K'(N_z+\alpha N_s)}{K(N_z+N_s)}
=
\frac{1(64+128)}{2(64+256)}
=0.3.
\]

端到端速度不会线性提升，因为 attention/MLP 仍占成本，但足够抵消 GRA 统计。若 V2 进一步裁 token 进入后续 attention，attention 复杂度也从：

\[
O((N_z+N_s)^2)
\]

降到：

\[
O((N_z+\alpha N_s)^2).
\]

### 9.4 不使用 FFT 的原因

不建议引入 FFT：

- 当前 response 已由 attention logits 或 q/k proxy 直接给出，无需频域相关。
- ViT token response 不是规则卷积相关，不满足 FFT 加速最有利的平移不变假设。
- FFT 会引入 transform、padding、复数/实数转换开销。
- 本方法目标是减少 token/expert 计算，而不是新增独立相关分支。
- response agreement 已经是 \(O(N_s)\) 的向量归约，FFT 的 \(O(N_s\log N_s)\) 不划算。

## 10. 算法一致性检查

| 目标 | 当前方案是否满足 | 说明 |
|---|---|---|
| 简洁有效 | 是 | V0 只加 GRA 统计和 gate，不动主 loss/head |
| 紧贴 tracking | 是 | 所有信号来自 template-search response |
| 不新增大 cross-attention | 是 | 只在已有 logits/output 上做 gated exchange |
| PEFT 兼容 | 是 | 新参数可通过命名进入白名单 |
| 计算持平 | V0 基本持平，V2 可降低 | V0 额外统计很小；V2 token/expert sparse 回收计算 |
| 逻辑闭环 | 是 | confidence 决定信谁，agreement 决定是否互导，MoE 决定怎么融合 |

最符合当前思路的默认实现是：

1. V0：GRA + RGAE，验证 response agreement 是否带来稳定增益。
2. V1：增加 HMoE router bias，验证专家选择是否更可解释。
3. V2：增加 token sparsification，专门回答计算量持平甚至降低的问题。

不要第一版就同时上 V0/V1/V2，否则难以判断收益来自哪里。

## 11. 风险与验证

### 11.1 Attention Map Materialization

风险：

- `brgb_attn/bdte_attn` shape 为 `[B,H,N,N]`。
- search size 256、stride 16 时 \(N_s=256\)，template 128 时 \(N_z=64\)，\(N=320\)。
- batch 较大时双流 logits 显存明显。

验证：

- profiler 记录 `CEBlock_AP` 前后 max memory。
- 比较 `gra_use_proxy=false/true`。
- 不要把每层完整 attention 存入 `aux_dict`。

### 11.2 Shape 验证

每个版本至少检查：

```text
x[0].shape == x[1].shape
x[0].shape[1] == lens_t + current_lens_s
global_index_s[0].shape == global_index_s[1].shape
final x_fusion.shape == [B, N_z + original_N_s, C]
pred_boxes.shape == [B, 1, 4]
score_map.shape 与 head 预期一致
```

### 11.3 Template/Search Token Index

风险点：

- `lens_t = global_index_template.shape[1]`。
- 默认 `CAT_MODE='direct'`，token 顺序是 `[template, search]`。
- 若未来启用 `add_cls_token=True`，GRA 的 `:lens_t` 切片会错位；当前配置为 false，V0/V1/V2 均应显式假设不支持 cls token。

验证：

- 对 `CE_TEMPLATE_RANGE='CTR_POINT'/'GT_BOX'/'ALL'` 分别做 smoke test。
- 检查 `ce_template_mask.shape == [B, N_z]`。

### 11.4 Top-K Mask 对 BBox Head 的影响

风险：

- bbox head 使用最后 `feat_len_s` 个 token reshape 成 feature map。
- 若未 scatter 回原始 search 网格，head 会错。
- 若 pruned token 用零填充，可能影响 score map 边界区域。

验证：

- `forward_head()` 前断言 search token 数等于 `feat_len_s`。
- 可视化 sparse keep mask 与 GT bbox 中心关系。
- 统计目标中心 token 被保留比例。

### 11.5 Gate 行为验证

需要记录：

```text
GRA/rho_mean
GRA/agreement_mean
GRA/c_rgb_mean
GRA/c_x_mean
Gate/x2r_mean
Gate/r2x_mean
Router/entropy
Router/expert_load
Sparse/keep_ratio
Sparse/target_center_keep_rate
```

### 11.6 Ablation 顺序

| 实验 | GRA | RGAE | Router Bias | Top-k MoE | Token Sparse |
|---|---|---|---|---|---|
| Baseline | 否 | 原始互导 | 原 HMoE | 否 | 否 |
| V0-a | 是 | sample scalar gate | 否 | 否 | 否 |
| V0-b | 是 | per-head gate | 否 | 否 | 否 |
| V1 | 是 | per-head gate | 是 | 否 | 否 |
| V1-topk | 是 | per-head gate | 是 | 是 | 否 |
| V2 | 是 | per-head gate | 是 | 是 | 是 |

指标：

- tracking: Success/AUC、Precision、Normalized Precision、PR/SR、F-score 等。
- loss: giou、l1、focal、IoU。
- efficiency: FPS、GPU memory、FLOPs、profiler block time。
- diagnostics: gate 分布、agreement、confidence、router entropy、expert load、keep ratio、target token keep rate。

## 12. 推荐默认落地方案

最推荐的实现路线：

1. **V0**：只在 `CEBlock_AP` 内做 GRA + RGAE，不动 HMoE，不动 token 长度，不动 loss。  
   目的：验证核心假设“target response agreement 是否能更稳地控制双向互导”。

2. **V1**：在 `HMoE` 的 `Combine` logits 上加 expert-specific `gra_router_bias`，先不开 top-k。  
   目的：验证 response statistics 是否能让专家路由更符合“信 RGB/信 X/信融合”的语义。

3. **V2**：实现 response-aware shared search token mask，替换或扩展 candidate elimination，保证 scatter recovery。  
   目的：回答“计算量持平甚至降低”的效率 claim。

第一篇主文可以把 V0+V1 作为核心方法，V2 作为效率增强；如果 V2 结果稳定，再把 Response-Aware Token Sparsification 上升为完整贡献点。

## 13. 最小代码修改清单

推荐最小文件改动：

```text
lib/models/layers/attn_blocks.py
  - add compute_gra_stats()
  - add compute_cross_response_gate()
  - add response_gated_attention()
  - modify CEBlock_AP.__init__()
  - modify CEBlock_AP.forward()

lib/models/layers/attn.py
  - modify HMoE.__init__() with gra_router_bias
  - modify HMoE.forward(..., gra_q=None, topk=None, token_mask=None)

lib/models/seatrack/vit_ci.py
  - pass gra config to CEBlock_AP
  - keep aux gra_stats if needed

lib/models/seatrack/seatrack.py
  - pass cfg.MODEL.GRA to backbone builder

lib/train/base_functions.py
  - extend PEFT trainable_params whitelist

lib/config/seatrack/config.py
  - add MODEL.GRA config block
```

不建议第一版改：

```text
lib/models/layers/head.py
lib/train/actors/seatrack.py main loss
lib/test/tracker/seatrack.py online state
```

这些位置保持不动，有利于控制变量。

## 14. 与当前算法思路的一致性结论

该接入方案与 GRATrack 的论文逻辑一致：

- GRA 从 template-search response 中估计单模态 confidence 和跨模态 consistency。
- RGAE 用 \(\rho_l\) 控制跨模态 attention exchange。
- TB-SMoE 用 \([c_r,c_x,\rho,c_r-c_x]\) 控制 expert routing。
- RATS 用 response mask 将专家计算集中到高价值 search token。

它保持了方法的“简单优美”属性：attention 负责目标响应，Gini agreement 负责可信协同，MoE 负责条件融合，token sparsification 负责回收计算量。第一版实现不需要新 cross-attention、不需要新 loss、不需要改 head，是一个可控、可消融、可解释的替换路径。
