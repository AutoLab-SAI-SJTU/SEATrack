---
title: "GRATrack: Gini Response Agreement Guided Sparse Expert Routing for Efficient RGB-X Tracking"
translated_title: "GRATrack：面向高效 RGB-X 跟踪的 Gini 响应一致性引导稀疏专家路由"
type: "method-proposal"
topic: "多模态目标跟踪"
created: 2026-07-09
status: "proposal"
---

# GRATrack: Gini Response Agreement Guided Sparse Expert Routing for Efficient RGB-X Tracking

## 1. 一句话主张

**GRATrack 将 RGB-X tracking 的融合问题重新表述为由目标响应驱动的条件决策问题：当单模态响应明确且跨模态响应一致时交换信息，当模态置信度不对称时偏向可靠模态，并且只在高响应目标区域激活稀疏专家计算。**

更短的论文式表述：

> Multimodal tracking does not need heavier fusion; it needs target-response-aware decisions on when to exchange, which modality to trust, and which experts to activate.

## 2. 论文标题

推荐英文题目：

**GRATrack: Gini Response Agreement Guided Sparse Expert Routing for Efficient RGB-X Tracking**

推荐中文题目：

**GRATrack：面向高效 RGB-X 跟踪的 Gini 响应一致性引导稀疏专家路由**

备选中文题目：

1. **基于 Gini 响应一致性的稀疏专家路由 RGB-X 目标跟踪**
2. **让跟踪响应决定融合：面向 RGB-X 跟踪的稀疏专家路由方法**
3. **面向高效 RGB-X 跟踪的响应一致性驱动跨模态交互与专家选择**

## 3. 摘要雏形

RGB-X 目标跟踪需要在 RGB 与辅助模态之间动态利用互补信息，但不同模态在遮挡、低照度、热干扰、深度缺失、事件噪声或空间错位下往往呈现不稳定可靠性。现有多模态跟踪方法通常将融合视为一种固定或加重的特征增强过程，而忽略了跟踪任务本身已经提供了一个更直接的判据：template-search response。该响应反映了模板目标在搜索区域中的定位证据，因此比一般特征相似度或模态注意力更适合回答“何时交换信息、相信哪个模态、何时融合”。

本文提出 **GRATrack**，一种由 Gini Response Agreement 引导的高效 RGB-X 跟踪框架。我们从各模态的 template-search attention 中估计目标响应分布，并通过响应集中度衡量单模态置信度，通过跨模态区域一致性衡量两种模态是否指向同一候选目标。基于该响应信号，模型进一步执行三类条件计算：首先，Response-Gated Attention Exchange 仅在响应可信且跨模态一致时进行双向信息交换；其次，Trust-Biased Sparse MoE 将单模态置信度与一致性信号注入专家路由，使专家选择偏向当前更可靠的模态或交互模式；最后，Response-Aware Token Sparsification 仅对高响应搜索 token 激活专家计算，低响应区域保留轻量残差路径。

GRATrack 的核心不是构造更重的融合模块，而是让跟踪响应成为跨模态交互和稀疏计算的控制变量。实验计划将在 RGB-T、RGB-D 与 RGB-E 标准跟踪基准上验证整体性能，并进一步通过受控退化协议评估模型在单模态退化、跨模态错位和模态缺失条件下的可靠性选择能力。最终结果将报告精度、成功率、鲁棒性、专家激活比例、token 计算比例和速度开销等指标，以证明响应引导的稀疏路由能在保持或提升跟踪性能的同时降低不必要的跨模态计算。

结果占位句：

> 在 `[benchmark names]` 上，GRATrack 相比 `[baseline]` 在 `[metric]` 上提升 `[x]`，同时将专家激活 token 比例降低至 `[y%]`，验证了响应引导路由在准确性与效率之间的优势。

## 4. 研究背景

### 4.1 多模态跟踪不是普通 Fusion

RGB-X 跟踪中的辅助模态可以是 thermal、depth、event 或其他传感器信号。直觉上，多模态输入能够提供互补信息：RGB 保留纹理、颜色和语义细节；thermal 在弱光或外观伪装下可能更稳定；depth 提供几何结构；event 对高速运动和亮度变化更敏感。

但是，tracking 中的多模态融合并不是“信息越多越好”。原因有三点。

第一，模态可靠性是时变的。同一视频中，RGB 可能在白天表现稳定，却在低照度或运动模糊下失效；thermal 可能在热干扰、背景热源或低热对比条件下误导模型；depth 可能存在空洞、边界噪声和反射错误；event 可能在低运动区域缺少足够事件。固定融合策略无法判断当前帧、当前层、当前空间区域究竟应该信任哪个模态。

第二，跟踪关注的是目标定位，而不是一般语义表达。多模态融合若只追求全局特征增强，可能将背景、干扰物或不可靠模态的信息一并注入目标表示。对于 tracking，真正关键的问题不是“两个模态能不能融合”，而是“融合后的证据是否能帮助模板目标在搜索区域中被更准确地定位”。

第三，跨模态交互本身存在成本与风险。密集交互会增加计算量，也会放大错误模态的干扰。尤其在长序列跟踪中，错误信息一旦进入模板更新或 search representation，可能导致漂移。因此，多模态跟踪需要选择性交互，而不是默认更重的融合。

### 4.2 Tracking Response 是更合适的依据

在 tracking-by-matching 框架中，模型通过 template 与 search 的交互来定位目标。template-search response 直接描述模板目标对搜索区域 token 的响应强度，它天然包含任务相关信息：

- 它是目标条件化的，而非普通模态特征。
- 它定义在搜索区域上，直接对应跟踪输出空间。
- 它可以反映响应是否集中、是否存在多峰干扰、是否与另一模态指向同一区域。
- 它可以在不同层级动态计算，从而支持逐层控制跨模态交互与专家路由。

因此，本文将 tracking response 作为跨模态决策信号，而不是把 fusion 当作固定结构。

核心思想是：

> 让 template-search response 决定：该不该交换信息、该信哪个模态、该激活哪个专家。

## 5. 核心研究问题

本文围绕三个问题展开。

### Q1: When to Exchange?

模型不应在所有层、所有 token、所有帧上无条件交换模态信息。只有当单模态响应足够集中，且跨模态响应在目标区域上具有一致性时，跨模态交换才更可能带来收益。

### Q2: Which Modality to Trust?

当 RGB 与 X 模态产生冲突时，模型需要判断哪个模态当前更可靠。本文使用单模态响应集中度作为可靠性线索，并通过跨模态一致性调节双向信息流。

### Q3: When to Fuse?

融合不应发生在所有搜索 token 上。低响应 token 往往对应背景或不确定区域，对这些 token 激活复杂专家既浪费计算，也可能引入噪声。因此，本文只在高响应 search token 上执行专家计算和融合更新。

## 6. 方法总览

GRATrack 包含四个核心模块：

1. **Gini Response Agreement (GRA)**  
   从 template-search attention 中计算单模态响应集中度与跨模态响应一致性。

2. **Response-Gated Attention Exchange (RGAE)**  
   使用响应一致性控制 RGB 与 X 分支之间的信息交换强度。

3. **Trust-Biased Sparse MoE (TB-SMoE)**  
   将响应置信度与跨模态一致性注入专家路由，使专家激活随模态可靠性变化。

4. **Response-Aware Token Sparsification (RATS)**  
   仅对高响应 search token 使用 MoE 专家，低响应 token 走残差路径以节省计算。

整体逻辑闭环：

```text
target-conditioned response
        ↓
single-modal confidence
        ↓
cross-modal response consistency
        ↓
response-gated attention exchange
        ↓
trust-biased sparse expert routing
        ↓
high-response token expert computation
```

这条链条的重点是：**先有目标响应，再有模态信任，再有跨模态交换，再有专家路由**。这样 fusion 不再是无条件堆叠，而是目标状态驱动的条件计算。

## 7. Gini Response Agreement

### 7.1 Template-Search Response Extraction

设模态为 \(m \in \{r, x\}\)，其中 \(r\) 表示 RGB，\(x\) 表示辅助模态。第 \(l\) 层、第 \(h\) 个 attention head 的 template-to-search attention 记为：

\[
A_{m,l}^{h,zs} \in \mathbb{R}^{N_z \times N_s},
\]

其中 \(N_z\) 为 template token 数量，\(N_s\) 为 search token 数量。我们将所有 template token 与所有 heads 对 search token 的响应平均，得到模态 \(m\) 在第 \(l\) 层的 search response：

\[
p_{m,l}(j)=\frac{1}{H N_z}\sum_h \sum_i A_{m,l}^{h,zs}(i,j).
\]

随后对响应进行归一化：

\[
\bar p_{m,l} = \frac{p_{m,l}}{\sum_j p_{m,l}(j) + \epsilon}.
\]

\(\bar p_{m,l}\) 可以被视为模板目标在 search grid 上的响应分布。

### 7.2 单模态响应集中度

如果响应分布接近均匀，说明该模态无法明确定位目标；如果响应集中在少数 search token 上，说明该模态给出了更明确的目标证据。本文使用 Gini-style concentration 衡量这一点：

\[
c_{m,l}=
\left[
\frac{N_s \|\bar p_{m,l}\|_2^2 -1}{N_s-1}
\right]_0^1.
\]

其中 \([\cdot]_0^1\) 表示 clamp 到 \([0,1]\)。当响应均匀时，\(c_{m,l}\) 接近 0；当响应高度集中时，\(c_{m,l}\) 接近 1。

该指标不是一般意义上的分类置信度，而是目标响应在搜索区域上的空间集中度。它回答的是：**当前模态是否给出了明确的目标位置证据**。

### 7.3 跨模态响应一致性

仅有高集中度并不足够，因为两个模态可能分别集中在不同目标或干扰物上。因此，需要衡量 RGB 与 X 的响应是否在空间上指向同一区域。

定义跨模态响应一致性：

\[
a_l=
\left[
\frac{
N_s \bar p_{r,l}^{\top} P_\sigma(\bar p_{x,l}) -1
}{
N_s-1
}
\right]_0^1.
\]

其中 \(P_\sigma(\cdot)\) 表示空间对齐、平滑或区域级映射算子。它可以用于处理 RGB 与 X 模态之间的轻微错位，也可以将 token-level response 转换为 region-level response，从而避免过度依赖逐 token 精确重合。

该一致性指标的含义是：两个模态是否在搜索区域中支持相近的目标候选位置。

### 7.4 GRA 交互强度

最终的 Gini Response Agreement 定义为：

\[
\rho_l = a_l \sqrt{c_{r,l}c_{x,l}}.
\]

这一设计有明确含义：

- 如果任一模态响应不集中，则 \(\rho_l\) 降低。
- 如果两个模态响应不一致，则 \(\rho_l\) 降低。
- 只有当两个模态都给出明确响应，且响应区域一致时，\(\rho_l\) 才较高。

因此，\(\rho_l\) 不是单纯 attention similarity，而是由单模态目标证据和跨模态空间一致性共同决定的 response agreement。

## 8. Response-Gated Attention Exchange

传统多模态交互通常默认在每层执行跨模态 attention 或 feature fusion。GRATrack 改为使用 \(\rho_l\) 控制信息交换强度。

设第 \(l\) 层两个模态分支的输出为 \(O_r\) 与 \(O_x\)，则响应门控交换为：

\[
\tilde O_r = O_r + \rho_l \lambda_{x\rightarrow r,l} O_x,
\]

\[
\tilde O_x = O_x + \rho_l \lambda_{r\rightarrow x,l} O_r.
\]

其中 \(\lambda_{x\rightarrow r,l}\) 和 \(\lambda_{r\rightarrow x,l}\) 表示方向性信任权重。它们可以由响应特征预测：

\[
q_l = [c_{r,l}, c_{x,l}, \rho_l, c_{r,l}-c_{x,l}].
\]

当 \(c_{x,l} > c_{r,l}\) 时，模型可以增强 \(x \rightarrow r\) 的信息注入；当 \(c_{r,l} > c_{x,l}\) 时，模型可以增强 \(r \rightarrow x\) 的信息注入。若 \(\rho_l\) 较低，即使某一模态响应集中，跨模态交换也会受到抑制，避免将冲突信息强行融合。

这一模块回答了：

- 是否交换：由 \(\rho_l\) 控制。
- 谁帮助谁：由 \(\lambda_{x\rightarrow r,l}\)、\(\lambda_{r\rightarrow x,l}\) 控制。
- 交换多少：由 response agreement 与方向性信任共同决定。

## 9. Trust-Biased Sparse MoE

### 9.1 响应信号注入专家路由

MoE 的核心是条件计算：不同输入应激活不同专家。对于 RGB-X tracking，专家选择不应只依赖 token feature，还应依赖当前模态可靠性与跨模态一致性。

设基础 router 对第 \(e\) 个专家的 logit 为 \(z_{l,e}\)，本文使用响应特征 \(q_l\) 对专家路由进行偏置：

\[
q_l=[c_r,c_x,\rho,c_r-c_x],
\]

\[
z'_{l,e}=z_{l,e}+u_e^\top q_l,
\]

\[
\pi= \mathrm{TopK\text{-}Softmax}(z'),
\]

\[
Y=\sum_e \pi_e E_e(\tilde X).
\]

其中 \(E_e\) 为第 \(e\) 个专家，\(\pi_e\) 为稀疏激活权重。\(u_e^\top q_l\) 使专家路由显式感知当前层的模态置信度和响应一致性。

### 9.2 专家语义建议

专家不需要被硬编码为固定语义，但论文叙事中可以将其解释为条件功能分工：

| 专家类型 | 作用 |
|---|---|
| 模态保持专家 | 在单模态响应可靠时保留该模态判别特征 |
| 跨模态交互专家 | 在 \(\rho_l\) 高时加强一致区域的信息融合 |
| 冲突抑制专家 | 在 \(c_r\) 与 \(c_x\) 差异明显或一致性低时降低不可靠模态影响 |
| 背景过滤专家 | 对低响应或多峰响应区域进行干扰抑制 |
| 几何/边界细化专家 | 利用辅助模态补充目标边界或结构信息 |

论文中应谨慎表述专家语义：可以说“专家倾向于学习不同响应状态下的处理模式”，但不要在没有可视化或统计证据前声称每个专家必然对应某种固定语义。

## 10. Response-Aware Token Sparsification

MoE 若对所有 search token 激活专家，会带来较高计算成本，也会让背景 token 参与复杂融合。本文使用 response-aware token mask，仅对高响应 search token 激活专家：

\[
M_l= \mathrm{TopK}(\bar p_{r,l}+\bar p_{x,l},K_s).
\]

其中 \(K_s\) 为参与专家计算的 search token 数量或比例。对于 \(M_l\) 中的 token，执行 sparse expert routing；对于低响应 token，保留轻量残差路径：

```text
high-response search tokens -> sparse experts
low-response search tokens  -> residual / lightweight update
```

这一设计使计算集中在最可能包含目标的位置区域，避免将专家容量浪费在背景区域。它同时服务于效率与鲁棒性：高响应区域获得更强建模能力，低响应区域减少错误融合机会。

## 11. 复杂度与算法技巧

设 GRA 只在 \(L_g\) 个层启用。

### 11.1 Response Compression

GRA 不比较完整 \(N \times N\) attention，只压缩 template-search response：

\[
A^{zs}\rightarrow \bar p \in \mathbb{R}^{N_s}.
\]

新增统计成本为：

\[
O(L_gHN_zN_s).
\]

常见设置下：

\[
N_z=64,\quad N_s=256,\quad H=12,\quad L_g=6.
\]

两模态 response compression 的额外操作约为：

\[
2L_gHN_zN_s
=
2\times6\times12\times64\times256
\approx2.36M.
\]

相比几十 G MACs 级别的 ViT tracker，这个统计量很小。

### 11.2 Log-Free Agreement

GRA 不使用 entropy、KL 或 JS divergence，而使用二阶范数和点积：

\[
\|\bar p\|_2^2,\quad \bar p_r^\top \bar p_x.
\]

这些操作都是乘加，更适合 GPU reduction。这里不建议使用 FFT：response agreement 已经是 \(O(N_s)\) 的向量归约，FFT 的 \(O(N_s\log N_s)\) 反而不合适，也会偏离 tracking response 的直接判据。

### 11.3 Router Bias Instead of New Router

TB-SMoE 不新建复杂 gating network，只给已有 router 加：

\[
U[c_r,c_x,\rho,c_r-c_x].
\]

因此新增参数量极小，且路由解释性更强。

### 11.4 Sparse MoE / Early Skip

只在高响应 search token 上激活专家，低响应 token 残差跳过。若原 MoE 激活 \(K=2\) 个专家，现在激活 \(K'=1\)，并保留一半 search token，则 MoE 部分相对成本约为：

\[
\frac{
K'(N_z+\alpha N_s)
}{
K(N_z+N_s)
}
=
\frac{1(64+128)}{2(64+256)}
=0.3.
\]

端到端速度不会线性提升，因为 attention 仍是主成本，但足以抵消 GRA 统计开销，甚至在专家模块较重时更快。

## 12. 训练目标

主训练目标仍应以 tracking supervision 为核心：

\[
\mathcal{L}_{track}
=
\mathcal{L}_{cls}
+\lambda_{box}\mathcal{L}_{box}
+\lambda_{iou}\mathcal{L}_{iou}.
\]

MoE 部分可加入轻量正则：

\[
\mathcal{L}
=
\mathcal{L}_{track}
+\eta \mathcal{L}_{balance}
+\mu \mathcal{L}_{budget}.
\]

其中：

- \(\mathcal{L}_{balance}\)：避免专家坍缩到少数专家。
- \(\mathcal{L}_{budget}\)：控制专家激活 token 数或计算预算。
- \(\mathcal{L}_{track}\)：保持主任务监督为核心，避免路由目标喧宾夺主。

训练中可加入模态退化增强，使 router 在可控质量变化下学习合理的模态信任策略。但论文叙事应强调：退化增强用于提升鲁棒性，GRA 本身仍来自 tracking response，而不是外部质量标签。

## 13. 相关工作写法建议

相关工作不应把本文写成对某一篇方法的修补，而应围绕问题轴组织。

### 13.1 RGB-X Tracking

介绍 RGB-T、RGB-D、RGB-E 等多模态跟踪任务的发展，强调辅助模态在低照度、遮挡、几何结构和高速运动场景中的价值。同时指出，多模态跟踪的关键挑战不是简单引入更多信息，而是动态判断不同模态在当前帧、当前区域和当前层级的可靠性。

### 13.2 Parameter-Efficient RGB-X Adaptation

介绍将预训练视觉模型适配到 RGB-X 跟踪的 PEFT 类方法，例如 adapter、prompt、LoRA-like tuning 或轻量跨模态模块。本文与这类方法的关系是互补的：PEFT 关注如何低成本适配模型参数，而 GRATrack 关注如何由 tracking response 控制跨模态交互和专家计算。

### 13.3 Robust Tracking under Missing or Degraded Modalities

介绍模态缺失、模态退化和传感器不稳定条件下的鲁棒跟踪研究。本文的区别是：不只在训练中随机丢弃模态或学习静态鲁棒表示，而是在推理过程中根据 response concentration 与 agreement 动态调节模态信任。

### 13.4 Sparse MoE and Conditional Computation

介绍 MoE 与 sparse routing 在视觉任务中的条件计算优势。本文的区别是：专家路由不仅由 token feature 决定，还由 tracking-specific response 信号偏置，因此路由决策与目标定位可靠性直接相关。

### 13.5 Temporal and Historical Fusion

介绍使用历史模板、记忆库或时序融合增强跟踪稳定性的方向。本文可作为正交补充：时序模块回答“如何利用历史信息”，GRATrack 回答“当前层和当前区域是否应该进行跨模态交互”。

## 14. 审稿防御要点

| 潜在质疑 | 回应思路 | 对应实验 |
|---|---|---|
| Attention agreement 不等于 reliability | 本文不直接使用普通 attention agreement，而是结合 response concentration 与 region-level consistency。只有响应集中且跨模态指向一致时，交互强度才提高。 | 对比 raw attention agreement、仅 concentration、仅 agreement、完整 GRA |
| 两个模态可能一致地看错目标 | \(\rho_l\) 不是绝对正确性保证，而是降低有害融合风险的控制信号。最终正确性仍由 tracking supervision 与 box/head 训练保证。 | 失败案例分析、多峰干扰场景可视化 |
| 这是不是普通 fusion 的复杂版本 | 本文核心是 conditional exchange and routing，而不是增加固定融合深度。融合发生在特定层、特定 token、特定专家上。 | 计算量/FPS/激活 token 比例/专家使用率 |
| 为什么需要 controlled degradation | 标准 benchmark 的自然退化混杂，无法判断模型是否真的学会信任可靠模态。受控退化能验证路由是否随 RGB/X 质量变化而合理调整。 | RGB degraded、X degraded、both degraded、misalignment、modality missing |
| Gini concentration 是否过于简单 | 简单指标具有可解释性和低成本，且与 tracking response 的空间确定性直接相关。可通过与 entropy、max response、learned confidence 对比验证。 | confidence metric ablation |
| MoE 是否会带来额外不稳定 | 使用 TopK sparse routing、load balancing 和 response-aware token selection 限制专家激活范围，避免无约束专家膨胀。 | 专家负载、路由熵、TopK 敏感性实验 |
| 高响应 token 选择是否会错过目标 | token sparsification 可采用保守比例，并保留低响应 token 的残差路径。若响应不确定，可提高 token budget 或降低 sparsity。 | \(K_s\) 敏感性、困难场景分析 |

## 15. 实验设计

### 15.1 标准 Benchmark

建议覆盖 RGB-T、RGB-D、RGB-E 三类设置。

| 模态设置 | 数据集建议 | 评价重点 |
|---|---|---|
| RGB-T | RGBT234、LasHeR、VTUAV、GTOT | 弱光、热干扰、遮挡、昼夜变化 |
| RGB-D | DepthTrack、CDTB、VOT-RGBD 相关协议 | 深度缺失、几何边界、遮挡恢复 |
| RGB-E | VisEvent、COESOT | 高速运动、亮度变化、事件稀疏性 |

具体数据集应以目标投稿社区常用协议为准。主文表格建议每类模态选择 1 到 2 个核心 benchmark，更多结果放 appendix。

### 15.2 主实验表

主实验应报告：

- Success / Precision / Normalized Precision
- PR / SR 或对应 benchmark 官方指标
- FPS 或 latency
- 参数量、FLOPs 或 activated FLOPs
- expert activation ratio
- high-response token ratio

主表建议结构：

| Method | Backbone | RGB-T metric | RGB-D metric | RGB-E metric | FPS | Activated Tokens | Activated Experts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline tracker | same | TBD | TBD | TBD | TBD | 100% | none |
| Dense fusion variant | same | TBD | TBD | TBD | TBD | 100% | all |
| Sparse MoE w/o GRA | same | TBD | TBD | TBD | TBD | TBD | top-k |
| GRATrack | same | TBD | TBD | TBD | TBD | TBD | top-k |

### 15.3 受控退化协议

标准 benchmark 只能评估自然分布下的整体表现，难以隔离模型是否真的学会了可靠模态选择。因此需要 controlled degradation。

RGB 退化：

- Gaussian blur / motion blur
- low-light / over-exposure
- color jitter
- random occlusion
- compression noise

X 模态退化：

- Thermal: thermal washout、hot distractor、low contrast
- Depth: missing holes、quantization noise、edge corruption
- Event: event dropout、background event noise、temporal accumulation mismatch

跨模态退化：

- spatial shift
- scale mismatch
- temporal delay
- partial modality missing
- one modality all-zero / random noise

退化等级建议设置为：

```text
clean -> mild -> moderate -> severe -> missing
```

报告性能退化曲线，而不只报告单点结果。

### 15.4 路由行为验证

为了证明方法不是只在最终指标上偶然有效，需要观察路由是否符合预期。

| 场景 | 预期行为 |
|---|---|
| RGB 清晰，X 退化 | \(c_r > c_x\)，路由偏向 RGB-preserving 或 RGB-to-X |
| X 清晰，RGB 退化 | \(c_x > c_r\)，路由偏向 X-preserving 或 X-to-RGB |
| 两模态一致且集中 | \(\rho_l\) 高，cross-modal exchange 增强 |
| 两模态冲突 | \(a_l\) 低，exchange 被抑制 |
| 响应分散 | \(c_r,c_x\) 低，专家激活减少或偏向保守路径 |
| 目标区域明确 | high-response token 被优先送入专家 |

可报告：

- \(\rho_l\) 随退化强度变化曲线
- \(\lambda_{x\rightarrow r,l}\)、\(\lambda_{r\rightarrow x,l}\) 方向性变化
- 专家使用直方图
- 高响应 token mask 与预测框的重合度
- 路由熵与专家负载均衡

### 15.5 消融实验

| 变体 | 目的 |
|---|---|
| w/o GRA | 验证 response agreement 是否必要 |
| w/o concentration \(c\) | 验证单模态响应集中度作用 |
| w/o agreement \(a\) | 验证跨模态一致性作用 |
| raw attention agreement | 证明普通 attention similarity 不足 |
| fixed exchange gate | 验证动态 \(\rho_l\) 优势 |
| w/o trust bias in MoE | 验证 \(q_l\) 注入 router 的作用 |
| dense MoE for all tokens | 验证 token sparsification 的效率收益 |
| random token sparsification | 验证 response-aware selection 的必要性 |
| different TopK experts | 分析专家数量敏感性 |
| different \(K_s\) | 分析 token budget 与性能/速度权衡 |
| entropy confidence vs Gini concentration | 验证 confidence metric 选择 |
| token-level vs region-level agreement | 验证 region-level consistency 的鲁棒性 |

## 16. 论文结构建议

### Introduction

叙事顺序建议：

1. RGB-X tracking 需要动态利用互补模态。
2. 多模态可靠性在跟踪中是时变、区域相关、任务相关的。
3. 固定融合或更重融合无法回答 when/which/how much。
4. tracking response 提供了目标条件化的可靠性线索。
5. 提出 GRATrack，让 response 控制 exchange、routing 与 token sparsification。
6. 概述贡献和实验设计。

### Related Work

按问题轴组织：

- RGB-X tracking
- parameter-efficient multimodal adaptation
- modality degradation and missing modality robustness
- sparse MoE and conditional computation
- temporal/context fusion for tracking

### Method

建议小节：

1. Problem Formulation
2. Template-search Response Extraction
3. Gini Response Agreement
4. Response-Gated Attention Exchange
5. Trust-Biased Sparse MoE
6. Response-Aware Token Sparsification
7. Training Objective and Complexity Analysis

### Experiments

建议小节：

1. Experimental Setup
2. Comparison with State-of-the-Art
3. Controlled Degradation Evaluation
4. Ablation Study
5. Efficiency Analysis
6. Visualization and Case Study

### Conclusion

强调本文贡献不是更复杂的融合，而是 response-guided conditional interaction and computation。

## 17. 边界表述

论文中不要把 \(\rho_l\) 说成绝对可靠性判断。更稳的表述是：

> GRA is a target-response consistency signal that modulates cross-modal exchange and routing. It reduces the probability of harmful interaction under modality conflict, but does not guarantee absolute correctness under shared distractors.

也就是说，\(\rho_l\) 是降低有害融合概率的控制信号，不是直接证明某个模态“必然正确”的 oracle。
