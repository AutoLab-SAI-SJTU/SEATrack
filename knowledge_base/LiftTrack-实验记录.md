# LiftTrack 实验记录

> 方法：LiftTrack: Rank-Collapsed Information-Preserving Lifting for Efficient Multimodal Tracking  
> 启动日期：2026-07-11  
> 状态：工程、效率与三种子 matched pilot 门禁通过，正在运行 60-epoch 正式训练  
> 设计冻结文档：`docs/superpowers/specs/2026-07-11-lifttrack-rank-collapsed-bilift-design.md`

## 1. 冻结假设

SEATrack 的跨模态交换有效，但 AMG 的全局标量混合可能通过压缩模态差异制造 attention 一致性，HMoE 则以密集执行所有专家的方式输出最多 rank-4 的低秩残差。候选方法保留可合并的 K/V LoRA，删除 AMG 与 HMoE，仅在 blocks `[5, 9]` 插入两个 rank-8、顺序交替的 BiLift 可逆加性耦合。

预声明机制：

```text
block 5: rgb_1 = rgb + f(x);     x_1 = x + g(rgb_1)
block 9: x_1   = x   + g(rgb);   rgb_1 = rgb + f(x_1)
f/g: up(GELU(down(parameter-free LN(source))))
```

`down` 使用 Xavier 初始化，`up` 为零初始化。每个耦合单元具有精确逆和单位 Jacobian 行列式；不声称整个跟踪网络可逆。第一候选保留原最终求和与冻结 Center Head，不加入动态 router、OT、token pruning、early exit、蒸馏或时序记忆。

## 2. 设计依据

- epoch-50 SEATrack checkpoint 中，AMG 对两模态 attention 差异的乘子在 L1/L3 仅为 `0.00503/-0.05456`，前层接近奇异混合。
- 在 640 个固定样本上，强制 AMG 两权重为 `0.5/0.5` 相对学习值的 IoU 变化为 `-0.000399`，完全关闭交换为 `-0.015658`。交换有用，attention 完全相等不是有效目标。
- 六个选层每次 forward 共执行 24 次 HMoE；所有 12 个 HMoE 模块的中心化输出数值秩均为 4，99% 能量位于 2 至 4 个分量。
- 1,600 样本四动作 oracle 可提升 `+0.016443` IoU，但离线 MLP router 测试动作准确率仅 `26.75%`，收益 `-0.000446`。动态路由不进入第一候选。
- 640 样本结构干预中，中后层交换最有价值，首选插入层预声明为 `[5, 9]`。

## 3. 分析预算

以下为实现前分析值，不作为效率验收结果：

| 项目 | SEATrack | LiftTrack | 备注 |
|---|---:|---:|---|
| 估算总 MACs | 56.466 G | 56.256 G | 输入 token 协议一致 |
| 旧 HMoE MACs | 0.2259 G | 0 | 六层、24 次调用 |
| 两个 BiLift MACs | 0 | 0.0157 G | rank 8 |
| 可训练参数 | 636,324 | 196,608 | Lift 为 LoRA 147,456 + BiLift 49,152 |

模型构造测试已实测 LiftTrack 的 BiLift 参数为 `49,152`，LoRA+BiLift 可训练参数为 `196,608`；CUDA 延迟和显存仍待配对 profile。

## 4. 预声明门禁

工程门禁：

- 所有单元/集成测试通过。
- 零初始化 BiLift 与 LoRA-only block 输出逐元素相同。
- 两种耦合顺序的逆重建误差不超过 `1e-5`。
- LiftTrack 不实例化 HMoE、AMG scaling、GRA 或 RGAE 参数。
- 真实 LasHeR 单样本 forward/backward 的预测、损失、LoRA/BiLift 梯度均有限。

效率门禁，相对同机同协议复现 SEATrack：

- 最坏路径分析 MACs `<= 1.00x`。
- batch-1 mean latency `<= 1.00x`。
- P90 latency `<= 1.02x`。
- peak allocated inference memory `<= 1.00x`。
- 相同 warmup、日志和 batch 下 training step time `<= 1.00x`。

五轮 pilot 门禁：

- SEATrack、LoRA-only、LiftTrack 使用相同 seed、样本数、样本顺序、初始化、优化器和保存协议。
- 预声明 checkpoints 为 epochs 1、2、3、4、5，不允许事后挑轮。
- 每个 checkpoint 上 LiftTrack 相对 SEATrack 的验证 IoU 不低于 `-0.002`。
- epoch 5 的 LiftTrack 验证 IoU 必须高于 LoRA-only。
- seed 0 工程有效后再跑 seeds 1/2；三 seed 均值通过后才允许 60 epoch。
- 默认候选失败时只依次测试 rank 16、layers `[7, 11]`，不得同时调整。

## 5. 环境与可复现状态

```text
workspace: /home/yufan/code/SEATrack-ProbAlign-VRE
branch: gratrack-scge-experiment
HEAD: 33905559007284c50f9e8ea268defc45afa0d0b7
worktree: dirty, 56 porcelain entries at record creation
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
driver: 595.71.05
Python: 3.12.13
PyTorch: 2.12.1+cu130
CUDA runtime: 13.0
cuDNN: 92000
kernel: Linux 7.0.0-27-generic x86_64
```

当前 dirty worktree 包含前序 GRATrack 研究与训练基础设施修改，是本轮实验的权威状态。不得把结果描述为仅由上述 HEAD 干净复现；每次运行保存完整 config 和日志。

预训练权重：

```text
/home/yufan/code/SEATrack-ProbAlign-VRE/pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
size: 354 MiB
```

已有复现 checkpoint：

```text
SEATrack ep50:
/mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0050.pth.tar

SEATrack ep60:
/mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0060.pth.tar

failed GRA V0-b ep60:
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0060.pth.tar
```

已有 LasHeR 复现证据仅作为研究起点，不替代本轮 matched pilot：

| 方法/权重 | PR20 | NPR20 | NPR AUC | SR/AUC |
|---|---:|---:|---:|---:|
| SEATrack clean ep50 | 0.703055 | 0.669015 | 0.604298 | 0.565194 |
| GRA V0-b ep60 | 0.697105 | 0.660070 | 0.597281 | 0.558795 |

## 6. 配置清单

```text
experiments/seatrack/rgbt_lora_only.yaml
experiments/seatrack/rgbt_seatrack_pilot.yaml
experiments/seatrack/rgbt_lora_only_pilot.yaml
experiments/seatrack/rgbt_lifttrack_short.yaml
experiments/seatrack/rgbt_lifttrack_pilot.yaml
experiments/seatrack/rgbt_lifttrack.yaml
```

三组 pilot 均为 5 epochs、train/val 各 60,000 samples、batch 32、每轮验证和保存。完整配置为 60 epochs、每 5 轮验证和保存。short 配置只用于通路验证，任何 short loss/IoU 都不得写入精度结论。

## 7. 工程验证记录

2026-07-11：

```text
command: /home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
result: PASS, 23 tests, 1.762 s
```

已通过：BiLift 零初始化、双顺序逆变换、梯度、诊断脱图；block 输出等价；默认旧模块兼容；配置互斥；选层/顺序传播；builder 参数传播；参数自由 LayerNorm 初始化；诊断聚合；PEFT 参数纯度；六组 YAML 审计。

## 8. 待运行清单

| 阶段 | 状态 | 产物 |
|---|---|---|
| CPU/CUDA synthetic forward-backward | passed | profiler JSON + finite checks |
| LasHeR one-sample short train | passed | checkpoint + split logs |
| SEATrack/LiftTrack paired profile | passed | paired JSON + gate decision |
| seed-0 三组 5-epoch pilot | passed | checkpoints + per-epoch table |
| seeds 1/2 pilot | passed | aggregate mean/std |
| 60-epoch LiftTrack | in progress | run manifest |
| LasHeR/RGBT234 benchmark | blocked by full training | raw outputs + metrics + CI |

## 9. Smoke 与效率门禁结果

### 9.1 CPU/CUDA synthetic forward-backward

CPU inference 产物：

```text
logs/lifttrack/smoke_20260711/cpu_synthetic_profile.json
config: rgbt_lifttrack_short
input: synthetic, template 1x6x64x64, search 1x6x128x128
finite outputs: true
pred_boxes: [1, 1, 4]
score_map: [1, 1, 8, 8]
HMoE/BiLift: 0/2
trainable parameters: 196,608
```

独立 CPU 与 CUDA backward 使用相同 short 配置和 seed 0。两者均得到 20 个可训练张量、196,608 个可训练参数、8 个 BiLift gradient tensors、12 个 LoRA gradient tensors；missing/nonfinite gradient 均为空，14 个 gradient tensors 非零。CUDA synthetic backward 的 peak allocated/reserved 为 `578,987,008/664,797,184` bytes，小于 32 GiB 门槛。

### 9.2 真实 LasHeR 单样本训练

```bash
CUDA_VISIBLE_DEVICES=0 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_lifttrack_short \
  --save_dir /mnt/tipro4t/seatrack_train_runs/lifttrack_short_20260711_1436 \
  --mode single
```

结果：

```text
wall time: 5.36 s
single train step total time: 0.513 s
checkpoint:
/mnt/tipro4t/seatrack_train_runs/lifttrack_short_20260711_1436/checkpoints/rgbt_lifttrack_short/SEATrack_ep0001.pth.tar
checkpoint bytes: 371,830,459
all floating checkpoint tensors finite: true
BiLift/LoRA state keys: 8/12
HMoE, r2dte_scaling, dte2r_scaling, RGAE, GRA state keys: 0
```

训练日志中的三个诊断值均有限：

```text
BiLift/x2r_update_ratio: 0.01087
BiLift/r2x_update_ratio: 0.01630
BiLift/difference_ratio: 1.00056
```

该 run 使用随机初始化和单样本，只证明数据、forward、loss、backward、日志和保存通路，不作为精度结果。

### 9.3 配对 inference profile

产物目录：

```text
logs/lifttrack/profile_20260711/01_seatrack.json
logs/lifttrack/profile_20260711/02_lifttrack.json
logs/lifttrack/profile_20260711/03_lifttrack.json
logs/lifttrack/profile_20260711/04_seatrack.json
logs/lifttrack/profile_20260711/profile_gate_summary.json
```

协议：RTX 5090，batch 1，同一 `smoke_sequence_000` template frame 0/search frame 1，输入 `[1,6,128,128]` 和 `[1,6,256,256]`；加载顺序 `SEATrack, LiftTrack, LiftTrack, SEATrack`；每次加载 warmup 20、测量 100 次，每方法共 200 次。两组均不加载 checkpoint，测量的是相同随机初始化协议下的架构开销；训练权重不会改变 dense kernel 形状，完成正式训练后仍需用最终 checkpoint 复测。

| 指标 | SEATrack | LiftTrack | Lift/SEA |
|---|---:|---:|---:|
| mean latency | 13.0793 ms | 9.9081 ms | 0.7575 |
| FPS from mean | 76.4565 | 100.9279 | 1.3201 |
| per-load P90 | 13.1159/13.0966 ms | 9.9324/9.9503 ms | 0.7598 conservative |
| peak allocated | 454,108,672 B | 448,827,392 B | 0.9884 conservative |
| trainable parameters | 636,324 | 196,608 | 0.3090 |
| HMoE/BiLift modules | 12/0 | 0/2 | - |
| analytical MACs | 56.466 G | 56.256 G | 0.9963 |

推理门禁：mean `PASS`，P90 `PASS`，peak allocated `PASS`，分析 MACs `PASS`。所有 400 次测量对应的最终输出均有限，形状为 `pred_boxes [1,1,4]`、`score_map [1,1,16,16]`。

### 9.4 配对 batch-32 training-step profile

产物目录：

```text
logs/lifttrack/train_profile_20260711/01_seatrack.json
logs/lifttrack/train_profile_20260711/02_lifttrack.json
logs/lifttrack/train_profile_20260711/03_lifttrack.json
logs/lifttrack/train_profile_20260711/04_seatrack.json
logs/lifttrack/train_profile_20260711/train_profile_gate_summary.json
```

协议：与 inference 相同的真实帧对复制为 batch 32；加载顺序相同；每次 warmup 5、测量 20 次，每方法共 40 次。CUDA event 覆盖 `zero_grad + forward + squared-output loss + backward + AdamW step`，无 AMP。

| 指标 | SEATrack | LiftTrack | Lift/SEA |
|---|---:|---:|---:|
| mean step | 342.1797 ms | 278.3232 ms | 0.8134 |
| per-load P90 | 342.7015/342.0637 ms | 278.3781/278.4364 ms | 0.8140 conservative |
| peak allocated | 16,303,964,672 B | 12,003,312,128 B | 0.7362 conservative |

训练效率门禁：mean step `PASS`，P90 `PASS`，peak allocated `PASS`。所有输出与 profile loss 有限。

### 9.5 决策

工程、真实单样本、推理效率和训练效率门禁全部通过。允许进入三方法、seed-0、5 epoch matched pilot；尚不允许启动 60 epoch 或正式 benchmark。

## 10. Matched Pilot 公平性审计

在启动 pilot 前发现并修复两项会破坏“同 seed、同样本顺序”的隐患：

1. 不同架构在构造模型时消耗不同数量的全局 RNG。修复前，同 seed 的 SEATrack/LiftTrack 12 个共同 K/V LoRA tensors 中有 5 个不同。新增 opt-in `MODEL.DETERMINISTIC_LORA_INIT=true`，以 `SHA-256(experiment_seed, stable_module_name)` 派生局部 seed，并按原规则重置 `lora_A` Xavier gain sqrt(2)、`lora_B` zero。三组完整模型 seed 0 的 12 个共同 LoRA tensors 现已逐元素一致。
2. DataLoader iterator 在模型构造后创建，原 shuffle 与 worker base seed 依赖全局 RNG。新增独立 train/val generator、Python/NumPy worker seeding 和 `base_seed + epoch` 重置；train seed 为实验 seed，val seed 为实验 seed加 1,000,000。断点恢复到相同 epoch 会得到相同顺序。

顶层 `tracking/train.py --seed` 已透传至 `run_training.py`、settings、LoRA 初始化和 loader。旧配置默认 `DETERMINISTIC_LORA_INIT=false`，不改变已有实验行为。

严格复现 `build loader -> build method-specific model -> iterate epoch 1` 后，三组首 batch 的全部 tensor SHA-256 相同：

```text
SEATrack:  a83f69279f6060e7a017d56a61a9b6895a60e2e59cc6eac16cfd0c3b8943d3d1
LoRA-only: a83f69279f6060e7a017d56a61a9b6895a60e2e59cc6eac16cfd0c3b8943d3d1
LiftTrack: a83f69279f6060e7a017d56a61a9b6895a60e2e59cc6eac16cfd0c3b8943d3d1
```

审计 batch shapes：template `[1,32,6,128,128]`、search `[1,32,6,256,256]`、template/search anno `[1,32,4]`、valid `[32]`。

最终 pre-pilot 回归：

```text
command: /home/yufan/code/SEATrack/.venv/bin/python -m unittest discover -s tests -v
result: PASS, 40 tests, 4.29 s
scoped git diff --check: PASS
```

## 11. Seed-0 Matched Pilot 进度

### 11.1 已完成的两条基线

两条基线均使用 seed 0、相同 deterministic LoRA 初始化、相同逐 epoch loader seed、每轮 60,000 个 train/val samples、batch 32。每轮最终验证统计如下：

| Epoch | SEATrack val IoU | LoRA-only val IoU | LoRA - SEA | SEATrack val loss | LoRA-only val loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.72768 | 0.72458 | -0.00310 | 1.24982 | 1.26876 |
| 2 | 0.73117 | 0.72817 | -0.00300 | 1.23045 | 1.24838 |
| 3 | 0.73422 | 0.73049 | -0.00373 | 1.21209 | 1.22898 |
| 4 | 0.73497 | 0.73168 | -0.00329 | 1.20479 | 1.21878 |
| 5 | 0.73903 | 0.73412 | -0.00491 | 1.18864 | 1.21406 |

运行目录：

```text
SEATrack:
/mnt/tipro4t/seatrack_train_runs/lifttrack_pilot_seatrack_seed0_20260711

LoRA-only clean restart:
/mnt/tipro4t/seatrack_train_runs/lifttrack_pilot_lora_seed0_cleanrestart_20260711
```

两组各有 epochs 1--5 共五个 checkpoint。逐个加载检查后，全部浮点模型 tensors 有限，checkpoint 内记录 epoch 与文件名一致。

### 11.2 断点公平性修复

首次 LoRA-only 运行在 epoch 3 中途因主机重启中断。旧 checkpoint 仅恢复模型与优化器，未保存 Python、NumPy、Torch 与 CUDA RNG；恢复后的 epoch-3 前 50 batch IoU 与原运行不一致，因此该恢复结果被废弃，不进入表格。

训练器现将四类 RNG 状态保存在 checkpoint 的 `rng_state` 字段，并在模型、优化器和统计字段恢复后重建 RNG。纯 RNG round-trip 与 checkpoint reconstruction 回归均通过。为消除旧 checkpoint 的不可恢复偏差，LoRA-only 从 seed 0 全新重跑；其 epoch 1/2 指标逐位复现首次运行，随后正常完成 epochs 3--5。

### 11.3 LiftTrack seed-0 结果

LiftTrack 使用预声明的 rank 8、layers `[5, 9]`，运行目录为：

```text
/mnt/tipro4t/seatrack_train_runs/lifttrack_pilot_lifttrack_seed0_20260711_190200
```

| Epoch | SEATrack IoU | LoRA-only IoU | LiftTrack IoU | Lift - SEA | Lift - LoRA |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.72768 | 0.72458 | 0.73011 | +0.00243 | +0.00553 |
| 2 | 0.73117 | 0.72817 | 0.73309 | +0.00192 | +0.00492 |
| 3 | 0.73422 | 0.73049 | 0.73529 | +0.00107 | +0.00480 |
| 4 | 0.73497 | 0.73168 | 0.73700 | +0.00203 | +0.00532 |
| 5 | 0.73903 | 0.73412 | 0.73976 | +0.00073 | +0.00564 |

LiftTrack 五个 checkpoint 均可加载，全部浮点模型 tensors 有限，epoch 编号与文件名一致，且 `rng_state` 均包含 Python、NumPy、Torch 与 CUDA 状态。epoch-5 val loss 为 `1.18281`，低于 SEATrack 的 `1.18864` 和 LoRA-only 的 `1.21406`。

### 11.4 Seed-0 门禁决策

LiftTrack 在每个预声明 checkpoint 上都不低于 SEATrack `-0.002`，实际差值范围为 `+0.00073` 至 `+0.00243`；epoch 5 高于 LoRA-only `+0.00564`。seed-0 精度门禁 `PASS`，保留首选 rank-8、layers `[5, 9]`，不触发 rank 16 或 layers `[7, 11]` 备选。

该运行在 RTX 5090 硬件 `power.limit=550 W` 下完成。训练期间显存约 `12,259 MiB`，温度主要为 `69--71 C`；驱动瞬时 `power.draw` 可围绕限制窗口小幅波动，但 `power.limit` 始终保持 `550.00 W`。下一步对 seeds 1/2 分别运行三方法 matched pilot，并以三 seed 的逐轮均值和标准差作最终 pilot 决策。

## 12. 三种子 Matched Pilot 结果

统计产物：

```text
logs/lifttrack/pilot_20260712/pilot_summary.json
```

三个方法在 seeds 0/1/2 上均完成五个预声明 checkpoint。下表为 val IoU 的跨种子均值和样本标准差：

| Epoch | SEATrack | LoRA-only | LiftTrack | Lift - SEA | Lift - LoRA |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.725250 ± 0.002324 | 0.722330 ± 0.002368 | 0.728167 ± 0.001694 | +0.002917 ± 0.000895 | +0.005837 ± 0.001180 |
| 2 | 0.730897 ± 0.000500 | 0.726947 ± 0.001138 | 0.732777 ± 0.000356 | +0.001880 ± 0.000671 | +0.005830 ± 0.000810 |
| 3 | 0.734607 ± 0.000687 | 0.730490 ± 0.000590 | 0.735753 ± 0.000661 | +0.001147 ± 0.000100 | +0.005263 ± 0.000406 |
| 4 | 0.736380 ± 0.001252 | 0.732993 ± 0.001231 | 0.738240 ± 0.001129 | +0.001860 ± 0.000642 | +0.005247 ± 0.000136 |
| 5 | 0.739320 ± 0.000930 | 0.734163 ± 0.000287 | 0.740027 ± 0.000260 | +0.000707 ± 0.000775 | +0.005863 ± 0.000449 |

15 个 LiftTrack 对 SEATrack 的 seed×epoch 配对中，14 个为正，1 个为 `-0.00008`；所有配对均高于预声明下限 `-0.002`。LiftTrack 对 LoRA-only 的 15 个配对全部为正，最小增益 `+0.00480`。epoch 5 的三个 seed 均高于对应 LoRA-only，差值为 `+0.00564/+0.00557/+0.00638`。

新增的 seeds 1/2 三方法 30 个 checkpoint 与 seed-0 LiftTrack 5 个 checkpoint 均逐个加载审计：模型浮点 tensors 全部有限，epoch 编号正确，`rng_state` 完整包含 Python、NumPy、Torch 和 CUDA。seed-0 两条基线的 10 个 checkpoint 已在 11.1 节单独完成有限性审计。

三种子门禁 `PASS`。保留首选 rank-8、layers `[5, 9]`，不运行 rank 16 或 layers `[7, 11]` 备选；允许启动 seed-0 60-epoch 正式训练。当前 pilot 只证明同预算验证 IoU 和效率，不替代正式 LasHeR/RGBT234 benchmark。
