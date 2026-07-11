---
title: "GRATrack 实验实施方案"
translated_title: "Experimental Implementation Plan for GRATrack"
type: "experiment-plan"
topic: "多模态目标跟踪"
created: 2026-07-09
status: "proposal"
source_docs:
  - "knowledge_base/GRATrack-论文撰写方案.md"
  - "knowledge_base/GRATrack-公式与代码接入方案.md"
---

# GRATrack 实验实施方案

## 1. 实验总目标

GRATrack 的实验不应只证明最终指标涨点，而应系统回答四个问题：

1. **性能**：GRATrack 是否在 RGB-T、RGB-D、RGB-E 跟踪基准上优于同骨干、同训练协议的 baseline。
2. **机制**：Gini Response Agreement (GRA)、Response-Gated Attention Exchange (RGAE)、Trust-Biased Sparse MoE (TB-SMoE)、Response-Aware Token Sparsification (RATS) 分别是否有效。
3. **鲁棒性**：当 RGB 或 X 模态退化、错位、缺失时，模型是否会根据 template-search response 改变跨模态交换和专家路由。
4. **效率**：稀疏专家和高响应 token 计算是否能抵消 GRA/TB-SMoE 的新增开销，使 FPS、显存、activated FLOPs 持平或下降。

一句话主张：

> GRATrack uses target-conditioned template-search response to decide when to exchange information, which modality to trust, and which experts or tokens to activate, and this should be validated by performance, ablation, degradation, routing behavior, and efficiency evidence.

## 2. 术语与版本锁定

| 术语 | 定义 | 实验中如何使用 |
|---|---|---|
| GRATrack | 完整方法名 | 最终模型或主方法行 |
| GRA | Gini Response Agreement | 从 template-search attention 中计算 \(c_r,c_x,a,\rho\) |
| RGAE | Response-Gated Attention Exchange | 用 \(\rho_l\lambda\) 控制双向 attention exchange |
| TB-SMoE | Trust-Biased Sparse MoE | 用 \([c_r,c_x,\rho,c_r-c_x]\) 给专家路由加 bias |
| RATS | Response-Aware Token Sparsification | 只对高响应 search token 激活专家或重计算 |
| V0 | GRA + RGAE | 验证 response agreement 是否能控制互导 |
| V1 | V0 + TB-SMoE | 验证 response statistics 是否能影响专家路由 |
| V1-topk | V1 + top-k expert | 验证专家稀疏化的效率收益 |
| V2 | V1-topk + RATS | 验证高响应 token 稀疏计算 |

默认结论策略：

- 若 V2 不稳定，第一篇主文以 **V1 作为核心 GRATrack**，V2 作为效率增强或补充实验。
- 若 V2 在 RGB-T、RGB-D、RGB-E 上均稳定，主文将 **V2 作为完整 GRATrack**。
- 不允许第一轮同时上 V0/V1/V2，否则无法判断收益来源。

## 3. 实验主线

实验推进顺序固定为：

```text
Baseline
  -> instrumentation only
  -> V0: GRA + RGAE
  -> V1: + Trust-Biased Router
  -> V1-topk: + Top-k Expert
  -> V2: + Response-Aware Token Sparsification
  -> controlled degradation
  -> full benchmarks
  -> ablation package
  -> visualization and failure analysis
```

每个阶段都必须有：

1. **目标**：该阶段回答哪个论文问题。
2. **操作**：具体改哪些文件、跑哪些训练或评测。
3. **验收**：进入下一阶段的最低标准。
4. **应对措施**：失败时先改什么、不能改什么。

## 4. 统一实验记录

每次训练和评测必须保存一个 `run_manifest.yaml`，至少包含：

```yaml
method: V0-b
config: experiments/seatrack/rgbt_gratrack_v0b.yaml
checkpoint: models/checkpoints/train/seatrack/rgbt_gratrack_v0b/SEATrack_epXXXX.pth.tar
commit: "<git commit or local diff id>"
seed: 0
gpu: "A6000 x 2"
dataset_train:
  - LasHeR_train
  - DepthTrack_train
  - VisEvent
dataset_eval:
  - LasHeR
  - RGBT234
metrics:
  tracking:
    - Success/AUC
    - Precision
    - Normalized Precision
  efficiency:
    - FPS
    - latency_ms
    - gpu_memory_mb
    - activated_flops
  diagnostics:
    - GRA/rho_mean
    - GRA/agreement_mean
    - GRA/c_rgb_mean
    - GRA/c_x_mean
    - Gate/x2r_mean
    - Gate/r2x_mean
    - Router/entropy
    - Router/expert_load
    - Sparse/keep_ratio
    - Sparse/target_center_keep_rate
```

命名建议：

```text
rgbt_baseline_seed0
rgbt_gratrack_v0a_seed0
rgbt_gratrack_v0b_seed0
rgbt_gratrack_v1_seed0
rgbt_gratrack_v1_topk_seed0
rgbt_gratrack_v2_seed0
```

## 5. 阶段 0：基线复现与数据闭环

### 目标

证明当前 SEATrack 训练、评测、数据路径和 checkpoint 流程可用，避免后续把环境问题误判为方法问题。

### 操作步骤

1. 检查 `lib/train/admin/local.py` 中的 `workspace_dir`、`lasher_dir`、`depthtrack_dir`、`visevent_dir`。
2. 使用最小配置先跑 smoke training，例如 `experiments/seatrack/rgbt_smoke.yaml`。
3. 跑 RGB-T baseline：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script seatrack --config rgbt --save_dir ./models --mode multiple
```

4. 跑 RGB-D baseline：

```bash
python tracking/train.py --script seatrack --config rgbd --save_dir ./models --mode multiple
```

5. 跑 RGB-E baseline：

```bash
python tracking/train.py --script seatrack --config rgbe --save_dir ./models --mode multiple
```

6. RGB-T 评测：

```bash
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name seatrack --dataset_name LasHeR --yaml_name rgbt --num_gpus 1
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name seatrack --dataset_name RGBT234 --yaml_name rgbt --num_gpus 1
```

7. RGB-D 评测：

```bash
cd Depthtrack_workspace
vot evaluate --workspace ./ rgbd
vot analysis --name rgbd
cd ..

cd VOT22RGBD_workspace
vot evaluate --workspace ./ rgbd
vot analysis --name rgbd
cd ..
```

8. RGB-E 评测：

```bash
CUDA_VISIBLE_DEVICES=0 python ./RGBE_workspace/test_rgbe_mgpus.py --script_name seatrack --yaml_name rgbe --num_gpus 1
```

### 验收标准

- 训练 forward 无 shape error。
- loss 正常下降，无持续 NaN。
- 至少一个 RGB-T、一个 RGB-D、一个 RGB-E 评测入口能跑通。
- baseline 指标、FPS、显存、日志路径完整保存。

### 应对措施

| 问题 | 应对 |
|---|---|
| 数据路径错误 | 先只启用一个已确认存在的数据集，路径稳定后再恢复完整训练集 |
| 显存不足 | 降低 batch size 和 num_workers，不改模型结构 |
| VOT 评测失败 | 先固定单 GPU 和单 workspace，确认 tracker 名称与 `trackers.ini` 一致 |
| baseline 指标异常 | 检查 checkpoint epoch、yaml 名称、输入尺寸、数据 split 和预训练权重 |

## 6. 阶段 1：诊断日志接入，不改变模型行为

### 目标

加入实验诊断体系，但不改变模型输出。该阶段是 V0/V1/V2 的观测基础。

### 操作步骤

1. 在 `CEBlock_AP.forward()` 和 `HMoE.forward()` 预留 `aux_dict` 或 `status` 收集接口。
2. 只记录标量或低维统计，禁止保存完整 `[B,H,N,N]` attention。
3. 默认记录项：

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

4. 所有诊断项必须使用 `.detach()`，写入日志前可 `.cpu()`。
5. 加入基础 shape assert：

```text
x[0].shape == x[1].shape
x[0].shape[1] == lens_t + current_lens_s
global_index_s[0].shape == global_index_s[1].shape
final x_fusion.shape == [B, N_z + original_N_s, C]
pred_boxes.shape == [B, 1, 4]
score_map.shape 与 head 预期一致
```

### 验收标准

- 与 baseline 指标基本一致。
- 显存不明显增加。
- 日志能写出默认诊断项，即使此时部分值为空或默认值。

### 应对措施

| 问题 | 应对 |
|---|---|
| 显存上涨 | 检查是否保存完整 attention；只保存 mean、std、histogram |
| 速度明显变慢 | 降低日志频率，例如每 50 或 100 iter 记录一次 |
| 日志污染训练图 | 将诊断项独立到 `diagnostics.csv` 或 TensorBoard 独立 namespace |

## 7. 阶段 2：V0，GRA + RGAE

### 目标

验证核心假设：template-search response agreement 能否更稳地控制 RGB-X 双向互导。

### 代码操作

1. 在 `lib/models/layers/attn_blocks.py` 中新增：

```text
compute_gra_stats()
compute_cross_response_gate()
response_gated_attention()
```

2. 从 `brgb_attn`、`bdte_attn` 中切出 template-to-search block：

```text
A[:, :, :lens_t, lens_t:]
```

3. 计算：

```text
c_r, c_x, agreement, rho
```

4. 用 RGAE 替代原始固定互导：

```text
S_r' = S_r + rho * lambda_x2r * (S_x - S_r)
S_x' = S_x + rho * lambda_r2x * (S_r - S_x)
```

5. 默认启用 `rho_min=0.1`：

```text
rho' = rho_min + (1 - rho_min) * rho
```

6. V0 不改 HMoE、不改 token 长度、不改 loss、不改 head。

### 实验配置

| 版本 | GRA | RGAE | 说明 |
|---|---|---|---|
| Baseline | 否 | 原始互导 | 复现基线 |
| V0-a | 是 | sample scalar gate | 最小风险版本 |
| V0-b | 是 | per-head gate | 默认候选 |
| V0-detach | 是 | per-head gate, `rho.detach()` | 判断 gate 反传是否不稳定 |
| fixed-gate | 否 | fixed learned gate | 验证动态 \(\rho\) 是否必要 |

推荐先在 RGB-T 跑完，再扩展到 RGB-D 和 RGB-E。

### 验收标准

- `rho` 不长期塌缩到 0 或 1。
- V0-b 在至少一个主 benchmark 上不低于 baseline。
- 退化样本中 `agreement` 和 `rho` 有可解释变化。
- 显存和速度没有不可接受开销。

### 应对措施

| 问题 | 应对 |
|---|---|
| `rho` 过小导致互导被压死 | 提高 `rho_min` 到 0.2，或减少启用层数 |
| 训练不稳定 | 尝试 `rho.detach()`，确认不稳定是否来自 GRA 反传 |
| 指标下降 | 从后半层启用 GRA；先只在 `[7,9,11]` 开启 |
| gate 无响应 | 检查 `CAT_MODE='direct'` 和 `lens_t` 切片是否正确 |
| 显存上涨 | 不保存 attention map；必要时启用 `gra_use_proxy=True` |

## 8. 阶段 3：V1，Trust-Biased Sparse MoE

### 目标

验证 response statistics 是否能让专家路由更符合“信 RGB、信 X、信融合或抑制冲突”的语义。

### 代码操作

1. 修改 `lib/models/layers/attn.py::HMoE.forward()` 签名：

```python
def forward(self, x, mode=None, gra_q=None, token_response=None,
            token_mask=None, topk=None, return_router=False):
```

2. 构造路由状态：

```text
gra_q = [c_r, c_x, rho, c_r - c_x]
```

3. 在 `Combine` softmax 前加 expert-specific bias：

```text
bias = einsum("bf,ef->be", gra_q, gra_router_bias)
Combine_logits = Combine_logits + bias[:, None, :]
```

4. `gra_router_bias` 必须初始化为 0。
5. 第一轮不开 top-k，只观察 `router_entropy` 和 `expert_load`。

### 实验配置

| 版本 | 目的 |
|---|---|
| V0-b | 对照 |
| V1 | 完整 trust bias |
| V1-rho-only | 只用 \(\rho\) |
| V1-c-only | 只用 \(c_r,c_x\) |
| V1-random-q | 随机打乱 `gra_q` |
| w/o trust bias | 去掉路由偏置 |

### 验收标准

- expert load 不坍缩到少数专家。
- RGB 退化时，路由统计向 X 或融合相关专家偏移；X 退化时反向偏移。
- V1 指标不低于 V0-b，至少在退化或困难场景上表现更稳定。

### 应对措施

| 问题 | 应对 |
|---|---|
| expert collapse | 降低 bias scale，加入 load balance，或前若干 epoch 冻结 `gra_router_bias` |
| bias 无效 | 检查是否是 expert-specific bias；单个 scalar bias 对 softmax 分布无效 |
| 指标下降 | 先只在高层 MoE 加 bias，或使用 `gra_q.detach()` |
| 路由解释不清 | 增加专家使用直方图、按退化类型分组统计 |

## 9. 阶段 4：V1-topk，专家稀疏激活

### 目标

单独验证 top-k expert 是否能降低专家计算，同时保持跟踪精度。

### 操作步骤

1. 在 V1 基础上打开 `MOE.TOPK`。
2. 从 `topk=2` 开始，再尝试 `topk=1`。
3. 先只在后半层启用 top-k expert。
4. 记录：

```text
activated_experts
expert_load
router_entropy
FPS
latency_ms
gpu_memory_mb
```

### 实验配置

| 版本 | 说明 |
|---|---|
| V1 | 无 top-k |
| V1-topk2 | 每 token 激活 2 个专家 |
| V1-topk1 | 每 token 激活 1 个专家 |
| V1-topk1-late | 只在后半层 top-k1 |

### 验收标准

- FPS 或显存有可观察收益。
- top-k 不引发 expert collapse。
- 性能下降在可接受范围内；若 top-k1 掉点明显，保留 top-k2。

### 应对措施

| 问题 | 应对 |
|---|---|
| 精度下降明显 | 从 top-k2 开始，或只在后半层启用 |
| 路由过硬 | 提高 softmax temperature，或训练前期不开 top-k |
| expert load 不均衡 | 加 load balance loss 或使用更保守的 top-k2 |

## 10. 阶段 5：V2，Response-Aware Token Sparsification

### 目标

回答“计算量持平甚至降低”的效率 claim，同时证明高响应 search token 选择不会破坏 bbox head。

### 代码操作

1. 构造 fused response score：

```text
p_f = w_r c_r p_r + w_x c_x p_x + w_a min(p_r, p_x)
```

2. 根据 `p_f` 选择 high-response search token：

```text
M_l = TopK(p_f, ceil(alpha * N_s))
```

3. RGB/X 默认共享 `keep_idx`。
4. high-response token 进入 MoE 或重计算路径，低响应 token 走 residual。
5. `vit_ci.forward_features()` 末端必须 scatter 回原始 search grid。
6. `forward_head()` 前断言 search token 数等于 `feat_len_s`。

### 实验配置

| 版本 | KEEP_RATIO | TOPK | 说明 |
|---|---:|---:|---|
| V1-topk | 1.0 | 1/2 | 无 token sparse |
| V2-0.75 | 0.75 | 1 | 保守稀疏 |
| V2-0.50 | 0.50 | 1 | 默认效率版本 |
| V2-0.35 | 0.35 | 1 | 激进效率版本 |
| V2-random | 0.50 | 1 | random token sparsification 对照 |

### 必须诊断

```text
Sparse/keep_ratio
Sparse/target_center_keep_rate
Sparse/keep_mask_gt_overlap
final_search_token_count
score_map_shape
```

### 验收标准

- `forward_head()` 输入 shape 完整。
- target center keep rate 足够高。
- FPS、latency 或 activated FLOPs 有明确收益。
- V2-0.50 指标不显著低于 V1-topk。

### 应对措施

| 问题 | 应对 |
|---|---|
| 裁掉目标 token | 提高 `KEEP_RATIO`，设置 `MIN_KEEP_RATIO=0.5`，响应不确定时保守保留 |
| scatter/recovery 出错 | 暂停 V2，保留 V1 作为主文核心 |
| score map 边缘异常 | 低响应 token 不用零填充，改为 residual 或原 token 回填 |
| 速度收益不明显 | 只声称 activated FLOPs 下降；若端到端 FPS 无收益，不夸大效率结论 |

## 11. 阶段 6：受控退化实验

### 目标

证明 GRATrack 不只是 benchmark 涨点，而是在模态质量变化时根据 response 改变交换和路由。

### 退化等级

```text
clean -> mild -> moderate -> severe -> missing
```

### 退化类型

| 类别 | 退化方式 |
|---|---|
| RGB 退化 | Gaussian blur、motion blur、low-light、over-exposure、color jitter、random occlusion、compression noise |
| Thermal 退化 | thermal washout、hot distractor、low contrast |
| Depth 退化 | missing holes、quantization noise、edge corruption |
| Event 退化 | event dropout、background event noise、temporal accumulation mismatch |
| 跨模态退化 | spatial shift、scale mismatch、temporal delay、partial modality missing、one modality all-zero/random noise |

### 输出结果

每类退化输出：

1. 性能退化曲线。
2. \(c_r,c_x,\rho\) 随退化强度变化曲线。
3. `Gate/x2r`、`Gate/r2x` 方向变化。
4. expert load 变化。
5. high-response token mask 与 GT bbox 关系。

### 预期行为

| 场景 | 预期 |
|---|---|
| RGB 清晰，X 退化 | \(c_r > c_x\)，路由偏向 RGB-preserving 或 RGB-to-X |
| X 清晰，RGB 退化 | \(c_x > c_r\)，路由偏向 X-preserving 或 X-to-RGB |
| 两模态一致且集中 | \(\rho_l\) 高，cross-modal exchange 增强 |
| 两模态冲突 | \(a_l\) 低，exchange 被抑制 |
| 响应分散 | \(c_r,c_x\) 低，专家激活减少或偏向保守路径 |
| 目标区域明确 | high-response token 被优先送入专家 |

### 应对措施

| 问题 | 应对 |
|---|---|
| 两模态一致看错目标 | 作为 failure case 报告，不把 \(\rho\) 说成 reliability oracle |
| 退化增强影响 clean benchmark | 将退化训练作为 robustness variant，不混入主模型 |
| 诊断曲线不符合预期 | 检查退化是否真的作用于对应模态；按序列类别分组统计 |

## 12. 阶段 7：主 benchmark 与公平比较

### 数据集安排

| 模态 | 主文优先 | 补充或扩展 |
|---|---|---|
| RGB-T | LasHeR、RGBT234 | VTUAV、GTOT |
| RGB-D | DepthTrack、VOT-RGBD | CDTB |
| RGB-E | VisEvent | COESOT |

### 主表行

主表至少包含：

```text
Baseline tracker
Dense fusion variant
Sparse MoE w/o GRA
V0-b
V1
V1-topk
V2 / GRATrack
```

### 主表列

```text
Method
Backbone
Training Data
RGB-T metric
RGB-D metric
RGB-E metric
FPS
Params
FLOPs / Activated FLOPs
Activated Tokens
Activated Experts
```

### 公平性要求

- 使用相同 backbone。
- 使用相同输入尺寸。
- 使用相同训练数据和训练 epoch。
- 使用相同 PEFT 策略，除非某个对照实验专门研究训练策略。
- 测速必须固定 GPU、batch size、warmup 次数和评测序列。

## 13. 阶段 8：消融实验包

### 核心消融矩阵

| 实验 | GRA | RGAE | Router Bias | Top-k MoE | Token Sparse |
|---|---|---|---|---|---|
| Baseline | 否 | 原始互导 | 原 HMoE | 否 | 否 |
| V0-a | 是 | sample scalar gate | 否 | 否 | 否 |
| V0-b | 是 | per-head gate | 否 | 否 | 否 |
| V1 | 是 | per-head gate | 是 | 否 | 否 |
| V1-topk | 是 | per-head gate | 是 | 是 | 否 |
| V2 | 是 | per-head gate | 是 | 是 | 是 |

### 机制消融

| 消融 | 回答的问题 |
|---|---|
| w/o GRA | response agreement 是否必要 |
| w/o concentration \(c\) | 单模态响应集中度是否有效 |
| w/o agreement \(a\) | 跨模态一致性是否有效 |
| raw attention agreement | 普通 attention similarity 是否不足 |
| fixed exchange gate | 动态 \(\rho_l\) 是否优于固定 gate |
| w/o trust bias in MoE | `gra_q` 注入 router 是否有效 |
| dense MoE for all tokens | token sparsification 的效率收益 |
| random token sparsification | response-aware token selection 是否必要 |
| different TopK experts | 专家数量敏感性 |
| different \(K_s\) | token budget 与性能/速度权衡 |
| entropy confidence vs Gini concentration | Gini concentration 是否优于 entropy |
| token-level vs region-level agreement | region-level consistency 是否更鲁棒 |

## 14. 阶段 9：可视化与失败分析

### 必备图

1. `rho/agreement/c_r/c_x` 随层和帧变化曲线。
2. RGB/X 退化时方向 gate 和 expert load 的变化。
3. high-response token mask 与预测框、GT 框叠加。
4. 成功案例、冲突案例、两模态共同看错的失败案例。
5. V0/V1/V2 的效率对比图：FPS、显存、activated FLOPs。

### 失败分析边界

论文中必须使用保守表述：

> GRA is a target-response consistency signal that modulates cross-modal exchange and routing. It reduces the probability of harmful interaction under modality conflict, but does not guarantee absolute correctness under shared distractors.

不要把 \(\rho_l\) 写成绝对可靠性判断。它只能证明“响应集中且跨模态一致”，不能证明两个模态一定没有共同看错目标。

## 15. 阶段 10：结果打包与论文写作输入

### 交付物

```text
results/
  main_results.csv
  ablation.csv
  degradation_curves.csv
  efficiency.csv
  diagnostics.csv
  run_manifest/
    rgbt_gratrack_v0b_seed0.yaml
    rgbt_gratrack_v1_seed0.yaml
    ...
figures/
  main_table.pdf
  ablation_table.pdf
  degradation_curve_rgbt.pdf
  routing_behavior.pdf
  sparse_mask_examples.pdf
  failure_cases.pdf
```

### 写作输入

| 论文 claim | 所需证据 | 状态 |
|---|---|---|
| GRATrack 提升 RGB-X tracking 性能 | 主 benchmark 表 | 待实验 |
| GRA 能控制有害互导 | V0 消融、退化曲线、gate 可视化 | 待实验 |
| TB-SMoE 使专家路由感知模态可靠性 | V1 消融、expert load、router entropy | 待实验 |
| RATS 降低不必要专家计算 | V2 效率表、target center keep rate | 待实验 |
| 方法在模态退化下更稳 | controlled degradation 曲线 | 待实验 |
| 方法有明确边界 | failure cases | 待实验 |

## 16. 决策规则

### 主方法选择

| 情况 | 论文中如何处理 |
|---|---|
| V1 稳定，V2 不稳定 | 主方法写 V1，V2 作为 efficiency attempt 或 appendix |
| V2 在 RGB-T/RGB-D/RGB-E 均稳定 | 主方法写完整 GRATrack，即 GRA + RGAE + TB-SMoE + RATS |
| V0 有效但 V1 无效 | 主贡献收缩为 response-guided exchange，MoE 路由只作为探索 |
| clean benchmark 提升小但退化优势明显 | 论文重点转向 robust RGB-X tracking under modality degradation |
| 退化和 clean 均无优势 | 回到 GRA 定义，检查 response 是否真正来自 target-conditioned template-search attention |

### 停止条件

任一阶段出现以下问题，应暂停进入下一阶段：

- baseline 未复现。
- shape assert 未通过。
- 训练 loss 持续 NaN。
- 诊断日志缺失关键项。
- V2 未能恢复完整 search grid。
- 评测协议不一致，无法公平比较。

## 17. 推荐执行顺序

1. `Baseline`：RGB-T 先跑通，再扩展 RGB-D/RGB-E。
2. `Instrumentation`：只加诊断，不改行为。
3. `V0-a`：sample scalar gate，检查 GRA 数值稳定性。
4. `V0-b`：per-head gate，作为默认 RGAE。
5. `V1`：加 trust-biased router，不开 top-k。
6. `V1-topk`：只研究 expert sparsity。
7. `V2`：研究 token sparsity 和 scatter recovery。
8. `Controlled degradation`：验证路由行为。
9. `Full benchmarks`：跑主结果。
10. `Ablation + visualization`：补齐论文证据链。

最终论文优先汇报：

```text
Main result -> degradation robustness -> core ablation -> efficiency -> routing visualization -> failure cases
```

这样可以先回答“是否有用”，再回答“为什么有用”，最后说明“在哪些条件下有效、在哪些条件下有限”。
