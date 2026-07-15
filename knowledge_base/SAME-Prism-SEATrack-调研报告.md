# SAME × Prism × SEATrack 调研报告

> 研究对象：*SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning*（arXiv:2602.01990v2，ICML 2026）及其 Prism 实现；目标是判断哪些思想值得迁移到 SEATrack，并形成可证伪、可投顶会的研究问题。
> 调研日期：2026-07-13。论文证据固定到 arXiv `2602.01990v2`，Prism 固定到 `7154be2a72a4f8e694c4361b7c6e05bb51bf5cc4`。本文只给出研究判断和实验门禁，不包含未经批准的模型改造。

## 1. 结论先行

SAME 值得借鉴，但借鉴对象需要拆成两层：基础结构是 **MoE + LoRA**，可用于普通 RGB-X 跟踪；频谱路由、曲率缩放与任务级冻结才属于持续学习机制。用户已明确研究重点是前者，并认为频谱感知路由尤其适合 RGB-X。因此，本报告不再把“RGB-T→RGB-D→RGB-E 且禁止旧数据回放”作为默认主线。

更新后的推荐主线是：**Target-conditioned Eigenspace-Stabilized MoE-LoRA for RGB-X Tracking**。核心不是再放一个 MoE，而是利用成对 RGB/X、template/search、目标/背景的独特几何，构造路由器输入的跟踪特定谱分解。但这是一个待证假设，不是 SAME 论文已经证明的跟踪结论：

1. template/search 分开统计，避免 search token 数量支配协方差；
2. 用目标响应区分 target 与 background，避免保护大量背景方向；
3. 用对齐 RGB/X token 的差分二阶矩刻画模态不一致，而不是用模态标签硬路由；
4. 先在权重冻结的路由实验中检验“对模态变化敏感、对历史目标身份影响小”的子空间是否存在；
5. MoE 专家采用数学正确的 $\sum_i\omega_iB_iA_ix$，不能复制 Prism 的交叉专家实现。

历史帧可以有两种作用：仅更新有界的低秩谱记忆，或进一步在线更新 router/LoRA 参数。前者是 online spectral memory，不应误称为参数持续学习；后者才是真正的测试时持续适配，必须与 CVPR 2025 PURA、经典在线 tracker 和通用 continual TTA 正面对比，并处理伪标签污染、参数回滚和实时性。研究主线应保留两级结构：先以无反传版本验证“历史谱是否真的能改善路由”，再讨论 router-only 在线更新；LoRA expert 在线更新只作为通过安全门禁后的第二阶段。任何阶段都不能事先承诺“不损失性能”，只能通过冻结的 clean noninferiority 和配对 benchmark 门禁判定。

## 2. 论文身份与证据范围

- arXiv：[2602.01990v2](https://arxiv.org/abs/2602.01990)，v2 日期 2026-05-27；页面注明 “Accepted to ICML 2026” 和代码链接。
- ICML 2026 官方 [Downloads](https://icml.cc/Downloads/2026) 与 [poster 页](https://icml.cc/virtual/2026/poster/64407) 包含 SAME；OpenReview id 为 `Nvim88fA66`。截至调研日未找到 PMLR 正式卷页或 DOI，因此不虚构卷页信息。
- 官方代码：[LAMDA-CL/Prism](https://github.com/LAMDA-CL/Prism)。本次审计固定在 commit `7154be2a72a4f8e694c4361b7c6e05bb51bf5cc4`。
- 本地全文材料（全文与图像因许可边界不纳入 Git 存档）：`knowledge_base/papers/arxiv_2602_01990_same/`，含 v2 PDF、官方 LaTeX、图像资源、页面映射和翻译说明。

论文实验只覆盖 LLaVA-v1.5-7B + CLIP-L/14-336，LoRA 仅插入语言模型 FFN，每任务 1 epoch；主表没有多 seed 方差。这足以支持 MCIT 场景中的方法有效性，但不能直接推出视频跟踪、双流 HMoE 或单任务训练有效。

## 3. SAME 到底做了什么

### 3.1 问题定义

任务序列记为 $D_1,\ldots,D_T$。在第 $t$ 个任务训练时不能访问旧数据。MoE-LoRA 的路由器和专家持续更新，产生两类遗忘：

- **路由器漂移**：旧样本随训练进程被分配到不同专家；
- **专家漂移**：即使重新匹配路由器，专家本身也已被新任务覆写，无法恢复旧功能。

SAME 的三个模块分别处理路由更新、专家更新和任务级专家激活。

### 3.2 频谱感知路由

论文累计路由输入的未中心化二阶矩：

$$
C^t=\frac{\alpha_{t-1}C^{t-1}+n_t\hat C^t}{\alpha_t},\qquad
\alpha_t=\alpha_{t-1}+n_t.
$$

对 $C^t$ 分解并保留达到能量阈值 $\delta$ 的最小主子空间 $V_\parallel$。路由器梯度被拆成高能子空间和近似零空间两部分，高能方向再按局部滑动窗口内的谱值进行各向异性缩放。思想上，它试图把“需要适应当前累计分布”和“尽量不影响历史输入”的方向分离。

必须注意：$C^t$ 按公式包含所有已见任务，因而 $V_\parallel$ 更准确地说是**累计分布高能子空间**，不是严格的新任务专属空间；原文对此表述并不完全一致。

### 3.3 曲率感知缩放

对线性专家 $h=W_i x$，历史功能变化近似为：

$$
\Delta_{\mathrm{degrad}}
=\mathbb E\|\Delta W_i x\|_2^2
=\operatorname{tr}(\Delta W_i C^{t-1}\Delta W_i^\top).
$$

由约束优化和黎曼梯度解释，作者用历史几何的逆对普通梯度预条件：

$$
\widetilde{\nabla W_i}=\nabla W_i(C^{t-1})^{-1},
$$

再用阻尼低秩伪逆避免存储/求逆完整矩阵。这里的乘法方向取决于权重的存储约定；不能脱离具体张量形状照抄。

### 3.4 自适应专家激活

作者综合当前任务的路由利用率 $\mathcal U(i)$ 与历史重要性 $\mathcal F^{pre}(i)$，逐层 min-max 归一化后定义：

$$
\operatorname{Score}(i)=\widetilde{\mathcal U}(i)-\widetilde{\mathcal F}^{pre}(i).
$$

分数低于阈值的专家在当前任务训练期间暂时冻结，下一任务或推理时重新激活。它希望同时降低跨任务干扰和训练成本。论文算法中任务计数 $n$ 的重置顺序与 $\mathcal F^{pre}$ 的读取存在按字面执行的边界歧义，代码是否实现作者语义必须独立审计。

## 4. 论文结果：哪些证据成立，哪些没有

| 证据 | SAME | 对照 | 能支持的结论 |
|---|---:|---:|---|
| TriGap 平均分 | 46.53 | MoELoRA 44.45 | 长序列、异质 MCIT 上 +2.08 |
| CoIN 平均分 | 66.82 | HiDe 63.95；MoELoRA 50.58 | 组合方法在该顺序上显著优于 MoELoRA |
| UCIT 平均分 | 67.12 | ModalPrompt 65.52 | 六任务序列上 +1.60 |
| CoIN 累加消融 | 50.58→61.32→65.89→66.82 | Router→Expert→Activation | 三模块呈累加收益，但不是全因子消融 |
| 平均遗忘（论文印刷值） | -4.23 | MoELoRA -19.04；HiDe -6.38 | Table 12 定义、符号和均值可复算冲突，不用作迁移定量证据 |
| 任务顺序最大偏差 | 0.2% | MoELoRA 7.1% | 对已测顺序更稳健 |
| MMMU OOD | 36.35 | MoELoRA 34.83；zero-shot 33.75 | 未见基准有小幅正迁移 |

论文的因果诊断比最终均值更值得关注。Fig.1 中，Task-1 与 Task-3/5/7 路由快照的重叠依次降到 86.2%/82.7%/79.4%；即使重新训练 Task-1 router，旧任务准确率仍从 87.9% 降到 73.2%，归一化路由熵从 92.6% 降到 61.1%，支持“误路由之外还存在专家功能退化”。后续机制图显示：Task-8 时 spectral router 的路由重叠为 81.59%，无该模块为 79.40%；重训 router 后，curvature scaling 的 Task-1 恢复准确率为 78.9%，无 scaling 为 75.7%。这些差值支持机制方向，但也表明漂移并未被消除。

论文还报告 top-k 推理吞吐从 1.03 提升到 1.26 it/s、平均每任务节省 32.1 分钟训练时间和 2.3K MiB/GPU。但这些效率结果依赖其稀疏专家实现；不能外推到 SEATrack 当前的 dense HMoE。

证据边界：单一 7B backbone、固定 LoRA/MoE 家族、无多 seed 误差条、任务边界已知。论文局限部分也明确指出 ambiguous task boundaries 和 input-format variation 仍待解决。

复现时必须区分“可复算错误”与“可复现性歧义”。可复算错误包括：

- TriGap 主文声称训练样本 “over 250k”，而 Appendix Table 5 十项相加为 235,000。
- CoIN VizWiz 在 Table 2 为 54.13、Table 4 为 54.53；后者八项字面均值为 66.87，不是印刷的 66.82。
- 正文称 FloodNet 增益明显，但 Table 1 中 SAME 81.09，低于 MoELoRA 90.41 达 9.32 个百分点。
- Fig. 4 文字称 curvature scaling “consistently improves”，但 T1 为 84.2<85.9，T4 为 80.1<82.1；优势主要出现在后期任务。
- Table 12 的遗忘公式、负值符号和平均分母不一致；MoELoRA 表内七项均值为 -20.41，不是印刷的 -19.04。

可复现性歧义包括：CoIN 基线多为直接引用原论文；UCIT 超参数在 20% 训练子集上选择但独立 validation 不明；“only LoRA trainable”与 Algorithm 1 显式更新 router 冲突；$\epsilon,\lambda$、优化器、weight decay、专家数、top-k、seed/repeats 均未完整报告；正文称 task-level freezing，伪代码却逐 batch 重算；共用计数 $n$ 的更新次序会在字面上覆盖 $F^{pre}$；Eq. (3) 的任务级累积与逐 batch 调用的避免重复计数方式未说明；$W_i=B_iA_i$ 的整体预条件如何落到 A/B 两个因子也未给出。$\mu$ 并非宽范围稳健：附录报告 $\mu=0.09/0.9/9$ 时分别为 62.3/67.4/61.4。

## 5. Prism 实现审计：不能按论文标题相信代码

以下映射固定到 Prism commit `7154be2…`。核心代码位于 `PEFT/tuners/custom/same.py`，生命周期/保存逻辑位于 `method/custom/same/integration.py`，默认参数位于 `config/methods/same.py`。

| 论文概念 | Prism 位置 | 审计判断 |
|---|---|---|
| SAME 配置与线性层 | `SAMEConfig`、`SAMEModel`、`SAMELinear`、`SAMEExpert` | 结构入口清晰；rank 配置是总 rank，实际每专家 rank 需除以专家数 |
| 路由前向 | `_same_forward_with_routing` | 训练时把所有 token/batch 的 logits 求均值，只产生一个全 batch 路由向量；与公式中的逐输入路由不同 |
| 频谱路由梯度 hook | `_spectral_aware_router_hook` | 实现了投影/缩放，但缩放函数与论文 $\sigma_i/\hat\sigma_i$ 不完全一致 |
| 专家曲率 hook | `_make_curvature_hook` | 只注册到 LoRA A，不是完整复合专家 $W_i$；属于近似实现 |
| 协方差更新 | `_update_router_covariance` | **主要不一致**：当前 batch 先中心化，每 20 step 重做随机低秩 sketch 并覆盖；没有按式 (3) 做跨任务计数加权累计 |
| 利用率/历史重要性 | activation metrics 相关函数 | 有运行统计和归一化，但语义受任务重置顺序影响 |
| 专家冻结 | adaptive freezing / masks | mask 路由但 `SAMELinearA/B.forward` 仍遍历并计算所有专家，不能据此声称跳过前向计算 |
| 任务重置 | `reset_for_new_task` | 仓库内未找到调用者；且未重置 `all_frozen`，训练期“下一任务重新激活”可疑 |
| checkpoint | `integration.py` / `specialized_integration.py` | 最终根 checkpoint 可保存 extra state，但 carry state 不完整，中间 `checkpoint-*` 还有 safetensors 优先级风险；不能整体复用 |

已确认的额外风险：

1. **专家组合公式错误。** 代码先算 $a=\sum_i w_iA_ix$，再算 $y=\sum_jw_jB_ja$，实际得到 $\sum_{i,j}w_iw_jB_jA_ix$ 的交叉专家项；只有严格 one-hot 路由才等价于论文 $\sum_iw_iB_iA_ix$。
2. **4-bit 分支静默退化。** 4-bit 创建标准 `Linear4bit` 而非 SAME layer，因此没有多专家、路由、频谱 hook、曲率缩放和激活 mask；8-bit 才有专门 SAME 实现。
3. 随机 range finder 后只做 QR，没有对小矩阵再 eig/SVD，也没有按谱值排序；后续能量累计和滑动窗口却假设主方向已降序。
4. 推理不仅使用 learned router，还把图像/文本 CLIP task prototypes 的 mixture 与其相乘；这是论文 Algorithm 2 未披露的第二套路由。
5. 推理 logits 先除以专家数再 softmax，会人为提高温度、使分布更平；随后才 top-2。
6. 当前任务编号对应专家被强制激活，这是论文公式之外的 task-to-expert 假设。
7. `_get_masked_routing` 的紧急分支按二维索引一个一维向量；正常路径因强制激活通常不会触发，但它是潜在 shape bug。
8. `lora_dropout=0.05` 被创建但 SAME forward 从未调用；`merge_and_unload` 也没有真正把 adapter 合并回 base weight。
9. 曲率 hook 内存在无日志用途的 `.norm().item()`，会在大量 layer/expert backward 上触发 GPU 同步。
10. 对激活 checkpoint 重计算做了 guard，且考虑了 8-bit 权重和 buffer 保存；这些防重复统计、状态持久化模式值得复用。
11. **中间 checkpoint 可能丢失 LoRA/router。** Trainer 先以 `safe_serialization=False` 写含适配器权重的 `adapter_model.bin`，extra-state 路径在 safetensors 不存在时另建一个仅含 `prism.same.*` 的 `adapter_model.safetensors`；加载器优先读 safetensors，因而中断后从 `checkpoint-*` 续训或手动加载可能忽略 bin 中的 LoRA/router。最终根 checkpoint 走 `safe_serialization=True`，不代表中间恢复安全。

Prism 还把 `lora_r` 定义为所有专家的总 rank，而每个专家实际为 `r/E`。CoIN/UCIT/TriGap 配置的 64/48/80 配合 8/6/10 个专家，恰好都是每专家 rank 8。若直接照论文把配置写成 `r=8`，可能截断甚至产生零 rank。

因此，本研究应“复现论文数学语义 + 以 Prism 为实现参考”，不能把 Prism 当前行为当作规范。

最小可复用的不是整套 `SameIntegration`，而是两个解耦概念：一是带显式 task lifecycle 的二阶矩/低秩谱状态管理器；二是放在 backward 与 optimizer step 之间、按实际张量方向变换梯度的稳定器。Prism 的 CLIP prototype task router、整棵 PEFT fork、DeepSpeed/HF wrapper 都不应搬入 SEATrack。

## 6. SEATrack 中真正相关的结构

当前构建路径是：

`lib/models/seatrack/seatrack.py → lib/models/seatrack/vit_ci.py → CEBlock_AP → HMoE`。

六个交互层为 `[1,3,5,7,9,11]`，每层在 attention 后和 FFN 后各放一个 HMoE，共 12 个模块；template/search 会分别调用，因此每个 batch 有 24 次 HMoE forward。

### 6.1 HMoE 不是 SAME 的 MoE-LoRA

在 `lib/models/layers/attn.py` 中：

- attention HMoE：$E=4,S=2$；FFN HMoE：$E=8,S=2$；
- `gate_thi` 形状是 `[384,E×S]`；
- Dispatch 在 token/subtoken 维 softmax，Combine 在 expert 维 softmax；
- 所有专家密集计算，不是 top-k conditional computation；
- `linear1` 自身也是 `768→4→768`、`if_act=False` 的 LoRP；专家 `lora_a` 为 `[E,384,4]`，`lora_b` 为 `[E,4,384]`，还有所有专家共享的 rank-4 `linear2`。

SAME 的 utilization 应映射到 Combine，而不是 Dispatch。template 与 search 必须分开统计，否则 search token 数多 4 倍，会淹没 template 几何。但“直接在 `[B,N×S,384]` 上做 rank-8/16/32 谱”已被代数和数值证据否定：在 eval 下，`linear1` 完整 768 维输出的未中心化秩为 5，每个 384 维 slot 为 5，两个 slot 合并为 10；rank-8 已解释 99.52% 能量，rank-16/32 只对应约 `1e-13` 的数值零空间。

因此更合理的 Stage-0 候选是 **eval-only native latent controller**：令

$$
z=\operatorname{LN}(x)A+b_a\in\mathbb R^4,\quad
H_s=zB_s+b_{b,s},\quad L_s=H_sG,
$$

只用 4 维 latent 历史谱产生 $\Delta z$，并以

$$
\Delta L_s=\Delta z(B_sG)
$$

直接调制 logits；专家 Dispatch 仍聚合原始 $H$。本地 float64 校验中，它与显式计算 $\Delta H_sG$ 的最大误差为 `1.33e-15`。该等价只在 tracker eval、dropout 关闭时成立；它不等价于原计划允许的任意 384 维 $\Delta H$，也不可外推到训练态。

Decision 8 的 B4 主方案对每个语义 family 保存精确的 $4\times4$ 未中心二阶充分统计，不做 rank-1 至 rank-4 的硬截断。令 $S_j$ 为加权外积和、$n_j$ 为有效权重，则写入时更新

$$
S_j\leftarrow\beta S_j+qZ^\top WZ,\qquad
n_j\leftarrow\beta n_j+q\operatorname{tr}(W),\qquad
C_j=S_j/\max(n_j,\epsilon).
$$

对 $C_j=U_j\operatorname{diag}(\lambda)U_j^\top$ 使用连续谱滤波

$$
\Pi_j=U_j\operatorname{diag}\left(
\frac{\lambda_k}{\lambda_k+\tau\bar\lambda}
\right)U_j^\top,\qquad \tau=1.
$$

$\epsilon$ 只防止零迹除法，不能代替谱尺度；否则 $\Pi_j$ 容易退化为近似单位阵。B5 的 $[z,1]$ 仅作为预注册 bias-direction 消融，不能在看到主结果后替换 B4。

### 6.2 梯度乘法方向必须重推

Prism 普通线性路由权重按 `[out,in]` 记号右乘输入空间投影；SEATrack `gate_thi=[in,out]`，所以路由梯度应左乘：

$$
\widetilde G_{gate}=P\,G_{gate}.
$$

同理，`lora_a[e]=[384,4]` 对输入做右侧线性映射，历史输入几何预条件应为：

$$
\widetilde G_{A_e}=C^{-1}G_{A_e},
$$

而不是 Prism 风格的 $G C^{-1}$。`lora_b[e]=[4,384]` 若也处理，需要独立的 $4\times4$ hidden covariance。共享 `linear2` 不是 expert-specific 参数，不应被包装成 SAME 专家保护。

### 6.3 正确的生命周期位置

- 统计：`HMoE.forward`，但 diagnostics 模式必须 output-identical；
- 梯度变换：`ltr_trainer.py` 的 backward 后、clip 和 optimizer step 前；AMP 必须先 `unscale_`；
- task start/end、历史快照和重置：trainer 生命周期，而不是 tracking loss；
- checkpoint：优先注册 buffers；否则显式保存所有 $C/U/F/n/mask$；
- DDP：统计量需要 all-reduce，否则每张卡会得到不同投影和冻结决策。

## 7. 本仓库现有实验给出的硬约束

1. **没有 router collapse 证据。** GRA 长训的 Attn expert load max 约 0.35–0.47，FFN 约 0.20–0.35，不能预设 SAME 修复了塌缩。
2. **动态路由可学习性有负证据。** 四动作 oracle 提升 `+0.016443 IoU`，但离线路由器只有 26.75% action accuracy，实际收益 `-0.000446`。稳定路由不等于可判别路由。
3. **门控日志正确不等于实际调制有效。** GRA 的 raw 指标变化被固定 `RHO_MIN` 淹没，最终 LasHeR 四项都落后基线。
4. **短训/validation 不能替代最终 benchmark。** LiftTrack 三 seed pilot 和长训 validation 都接近基线，但 LasHeR test 出现 PR20 `-0.140417`、NPR20 `-0.144785`、SR/AUC `-0.116474` 的灾难性泛化差距。
5. **效率叙事不成立。** 删除 12 个 HMoE 后推理和训练都明显更快；保留 dense HMoE 再加 SVD/all-reduce 不能宣称天然高效。
6. **冻结 slice 不等于冻结参数。** 当前专家堆在单个 Parameter 中；即使梯度 slice 清零，AdamW momentum 和 decoupled weight decay 仍可能改变它。必须逐 expert 检查参数和 optimizer state。
7. **HMoE 有共享 rank-4 瓶颈。** 现有诊断显示 12 个 HMoE 的中心化输出数值秩均为 4，SAME 只能减少漂移，不能增加表示秩。
8. **路由几何也受 `linear1` rank-4 约束。** 原 Workstream A 的 `[8,16,32]` 候选会把数值零空间当成额外自由度；在改写谱坐标前不应开始实现。

## 8. 竞争格局与创新碰撞

截至 2026-07，单纯“给 RGB-X 跟踪器加 MoE/路由”已非常拥挤：

- [XTrack（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_XTrack_Multimodal_Training_Boosts_RGB-X_Video_Object_Trackers_ICCV_2025_paper.html) 已利用多模态训练和 MoE 路由提升 RGB-X 跟踪；
- [What You Have is What You Track / FlexTrack（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_What_You_Have_is_What_You_Track_Adaptive_and_Robust_ICCV_2025_paper.html) 已研究缺失模态和异构 MoE；
- [OneTrackerV2（ICML 2026）](https://arxiv.org/abs/2605.03716) 已提出 Dual MoE，统一多种 RGB/RGB-X 任务并处理缺失模态；
- [SEATrack（CVPR 2026 Oral）](https://arxiv.org/abs/2604.12502) 本身已把多模态适配和 HMoE 作为核心结构。
- [PURA（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Shao_PURA_Parameter_Update-Recovery_Test-Time_Adaption_for_RGB-T_Tracking_CVPR_2025_paper.html) 已做 RGB-T 测试时参数更新/恢复，并用 SVD 分解更新轨迹；“在线更新 + SVD”本身不是空位。
- [SPMTrack（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Cai_SPMTrack_Spatio-Temporal_Parameter-Efficient_Fine-Tuning_with_Mixture_of_Experts_for_Scalable_CVPR_2025_paper.html) 已使用初始帧、3 个时序参考帧和包含低秩专家的 TMoE；“历史帧 + routed low-rank experts”不是空位。
- [DTPTrack（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html) 已用固定初始模板、3 个有界动态状态和逐帧可靠性压缩历史；“reliability-gated bounded history”也不是空位。
- [GOLA（AAAI 2026）](https://arxiv.org/abs/2512.05359) 已对 RGB-T LoRA 权重做 SVD、冻结重要 rank 并约束冗余 rank 组正交；“LoRA 谱分解/正交”也不是空位。
- [SpecTrack（2026）](https://arxiv.org/abs/2607.05988) 已在 MSI/HSI 跟踪中提出 Spectral Prompt Router + MoE；其 spectral 指波段光谱，而本研究必须明确指输入二阶矩的 eigenspectrum，避免标题和概念混淆。
- [EMoE（2026）](https://arxiv.org/abs/2601.12137) 已让正交基对齐经验输入协方差的主特征空间，并用 token 在各 eigenvector 上的投影能量路由专家；不能声称首次提出 input-covariance/eigenspace router。
- [ERMoE（CVPR 2026）](https://arxiv.org/abs/2511.10971) 已将专家重参数化到正交特征基并按输入—专家 basis 相似度路由；“eigenbasis 稳定路由”也不是单独创新点。

因此，以下标题级主张不足以构成强创新：稳定 MoE 路由、给 SEATrack 加专家正则、在线参数更新、对 LoRA 做 SVD、处理缺失模态、统一多个 RGB-X benchmark、输入协方差特征基路由。较明确但很窄的组合空位是：**在无标签 RGB-X 单目标跟踪中，从可信有界历史在线估计目标特定的跨模态共享/私有协方差子空间，用它同时控制 MoE-LoRA 的路由、记忆准入和安全参数更新**。这与 PURA 的 BN 参数轨迹谱、GOLA 的 LoRA 权重谱、SpecTrack 的波段光谱、EMoE 的当前输入能量路由均不同。

## 9. 三条研究路线

| 路线 | 核心问题 | 新颖性 | 风险 | 当前建议 |
|---|---|---:|---:|---|
| A. 离线区域平衡频谱路由 | 训练期以 template/target/background/RGB-X 差分几何整形 router；推理参数固定 | 中 | 与 SAME/EMoE 接近 | 必做基础基线 |
| B. 在线共享—私有谱记忆 | 可信历史只更新有界低秩谱状态；四个子空间直接调制当前路由，不反传 | 中高 | 与 DTPTrack/多级记忆方法重叠，不能称 continual parameter learning | **安全验证层** |
| C. 可信历史驱动的 continual MoE-LoRA | B 的谱状态同时门控 memory admission 与 gradient admission；先 router-only，后选中 LoRA experts；支持锚定和回滚 | 条件性高 | 与 PURA、BECoTTA、在线 tracker 正面比较；伪标签污染和速度风险最大 | **最终主方法候选** |

推荐组合暂称 **Reliability-Gated Target Spectral Continual MoE-LoRA**。它不把 eigenspectrum 当作普通正则，而是利用 RGB-X 空间对齐与 tracking response 定义四类有界历史几何：共享目标身份 $C_{id}$、跨模态私有差异 $C_{private}$、可信外观动态 $C_{dyn}$ 和困难背景 $C_{bg}$。对 RGB/X 的响应加权目标原型 $p_t^R,p_t^X$，构造

$$
c_t=(p_t^R+p_t^X)/\sqrt 2,\qquad
d_t=(p_t^R-p_t^X)/\sqrt 2,\qquad
\nu_t=c_t-c_{t-1},
$$

分别更新共享、私有和动态谱，并由初始模板对子空间和参数保持锚定。可信度 $q_t$ 必须综合响应集中度、跨模态预测一致性、与初始/可信模板的相似度和增强或前后向一致性；低 $q_t$ 时既不写记忆，也不执行 optimizer step。最值得验证的假设是：历史目标谱能在一个模态退化时保护共享身份方向，同时允许私有/动态方向选择和更新不同 LoRA experts。

### 9.1 Stage-0 谱坐标的三个实施选项

| 选项 | 语义 | 风险 | 当前判断 |
|---|---|---|---|
| A. 384D post-`linear1`，rank=8 | 对现计划改动最小 | 合并 slot 子空间可交叉混合，rank-16/32 无意义，新增路由计算较大 | 仅作回退/对照 |
| B. 4D native latent，eval-only | 在 HMoE 真正的路由信息坐标中维护精确 $4\times4$ 状态，以连续谱滤波融合计算 $\Delta z(B_sG)$ | 可能退化为标量算子；不能宣称与 384D SAME 算子等价；未来训练态须另行设计 dropout 语义 | **Decision 8 主方案 B4** |
| C. 768D pre-`linear1` | 在完整视觉特征上记忆谱，再经 $A$ 影响路由 | 引入新的高容量外部控制，速度、污染和归因风险最高 | 只作后续容量消融 |

对于 $N=512,S=2$ 的理论路由控制路径，B 相比 A 的新增乘加约少 128–154 倍，但这不是端到端 FPS 结果。只有配对 profiler 与完整 OPE 评测才能给出效率或精度结论。

## 10. 三道 go/no-go 门禁

### Gate A：零行为谱诊断

用固定 clean/blur/missing 样本，对每个 `block × attn/ffn × template/search` 记录 covariance spectrum、主子空间角度、Combine utilization、专家输入能量以及投影前后梯度夹角，但不改变输出、不执行参数更新。

通过条件：instrumentation on/off 的预测逐元素一致；统计保留 layer/mode 粒度；存在可复现的跨任务/域几何变化。若不存在漂移，停止 SAME 主线。

### Gate B：单 batch 梯度与冻结语义

检查 $P G_{gate}$、$C^{-1}G_A$ 的 shape、有限性、norm、cosine；验证单卡/双卡 sufficient statistics 一致；执行一次 AdamW step，确认冻结专家参数及 optimizer state 均 bitwise 不变。

### Gate C：历史谱记忆证伪

模型权重完全固定，只在每条序列内维护有界 4D 谱状态。比较：初始模板、普通动态模板、PURA、仅置信度路由、目标条件频谱路由。必须在 clean、RGB 缺失、X 缺失、blur、遮挡恢复和长序列上同时报告准确率、失败恢复、更新时间、FPS 和额外显存。

只有 B 相对“同容量普通记忆/普通置信度门控”产生可重复收益，且没有利用测试 ground-truth，才进入 C。C 的第一版只更新 HMoE router，冻结 LoRA experts、共享投影和 backbone：这既隔离 router drift，也避免更新共享投影后历史谱坐标系立即失效。通过后才允许低频更新被选中的 LoRA experts，并加入参数快照、连续异常回滚和逐序列 reset。

在开始 gate-confirmation 前还必须通过四个预注册 stop gates：

1. 每个活跃 `(seed, benchmark, block, site, family)` 的各向异性指标 $\|\Pi_j-\operatorname{tr}(\Pi_j)I/4\|_F/\|\Pi_j\|_F$，sequence-bootstrap 单侧 95% LCB 至少为 0.10；
2. $K=[B_0G,B_1G]$ 按 $\sigma_i/\sigma_1\ge10^{-3}$ 的数值秩为 4；$C_j^{1/2}K$ 对 identity/private/background 的 `rank_1e-2` 至少为 2，对 dynamic 至少为 1，并覆盖至少 80% admitted active frames；
3. 归一化 family projector 的最小两两 Frobenius 距离单侧 95% LCB 至少为 0.05，且四个 strength-matched leave-one-family-out 对比全部为正；
4. LasHeR/DepthTrack 相对 `confidence_only_scalar_history` 与 `routing_disabled_legacy` 的逐 benchmark 单侧 97.5% `LCB(ΔJ)>-0.3 pp`，校准阶段还要求 `LCB(J_B-J_A)>-0.3 pp`。

任一谱门禁失败，就应降级为 scalar 或 fewer-family history，不能继续声称“四谱路由”贡献；非劣门禁也是停止条件，不是事前性能保证。

## 11. 当前暂不做的事情

- 不修改 tracking loss；SAME 原方法没有新增 actor loss。
- 不同时叠加 GRA、BiLift、ProbAlign、top-k 重构和 SAME。
- 不把仅保存模板/统计量误称为参数持续学习；若无在线参数更新，应明确叫 online spectral memory/routing。
- 不以训练 IoU、短训 validation 或单一 checkpoint 作为主结论。
- 不宣称“专家冻结节省计算”，除非 profiler 证明前向/反向确实跳过，并且参数与 optimizer state 都保持不变。

## 12. 已批准的修订边界

用户已批准 Workstream A 的 Decision 7，以及 **Decision 8-B4**：Stage 0 改为 eval-only 4D native-latent controller，A8 降为强制性能对照，B5/C 保留为预注册消融或诊断上界。隔离工作树 `.worktrees/target-spectral-a` 已建立；在补齐仓库本地、被忽略的 `tools/` 覆盖后，基线 40 项测试全部通过。下一步先冻结 Decision 8 addendum，再按 TDD 修复 evaluator/lifecycle/配置隔离等因果先决条件；当前仍没有模型实现或性能结论。
