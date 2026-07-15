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

## 13. Seed-0 60-Epoch 正式训练

正式运行目录：

```text
/mnt/tipro4t/seatrack_train_runs/lifttrack_full_seed0_20260712_020400
```

运行使用 commit `16d188e5c0e6c91b325cb6bb195c7c6505bb48d1`、seed 0、rank 8、layers `[5, 9]`、batch 32、每轮 60,000 个 train/val samples、每 5 轮验证与保存，并将 GPU 功率上限保持为 550 W。可复现配置和预注册 benchmark 门槛记录在运行目录的 `run_manifest.yaml`。

训练曾在 epoch 1 batch 950 后人工暂停，随后以 `SIGCONT` 原进程恢复。暂停时段污染 epoch 1 的累计时间与 FPS，但未重启进程、未重载 checkpoint，也未改变优化器状态或样本序列。恢复后完整跨入 epoch 2；epoch 2--4 的整轮训练时间分别为 `0:07:34.557077`、`0:07:34.174875` 和 `0:07:34.314869`。

### 13.1 Epoch-5 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train IoU | 0.72849 | 0.73253 | +0.00404 |
| validation IoU | 0.74061 | 0.74033 | -0.00028 |
| train epoch time | 0:11:02.696569 | 0:07:34.392275 | -31.4% |
| validation epoch time | 0:04:58.384676 | 0:03:35.027691 | -27.9% |

正式 epoch-5 validation 比同协议 clean baseline 低 `0.00028`，仍高于 pilot 预声明下限 `-0.002`，因此该节点不触发停止；这一结果不能单独证明正式精度提升，继续训练并按所有 validation 节点选择最佳权重。

Checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/lifttrack_full_seed0_20260712_020400/checkpoints/rgbt_lifttrack/SEATrack_ep0005.pth.tar
```

文件大小为 372,666,631 bytes，SHA-256 为 `6cd7dd0f30c5465e68983df974d286b4d76003d97a9e60719e9a909ef1b64d0b`。加载审计确认 checkpoint epoch 为 5，250 个浮点模型 tensor 全部有限，`rng_state` 完整包含 Python、NumPy、Torch 和 CUDA 状态。相同 epoch 的 clean baseline checkpoint 为 378,185,557 bytes，LiftTrack 小 5,518,926 bytes（1.46%）。审计后训练自动进入 epoch 6。

### 13.2 训练后数值可逆性

从 epoch-5 checkpoint 分别加载 layer 5（forward coupling）与 layer 9（reverse coupling）的已训练 BiLift 权重，在 CPU float32 上对固定随机输入执行 `forward -> inverse`。layer 5 的 RGB/X 最大绝对恢复误差分别为 `4.7684e-7/2.3842e-7`，相对 L2 误差为 `3.8314e-8/2.0658e-8`；layer 9 的最大绝对误差为 `2.3842e-7/5.3644e-7`，相对 L2 误差为 `2.7619e-8/5.5297e-8`。因此两个方向的已训练耦合层均保持 float32 数值精度下的可逆性。

### 13.3 Epoch-6--8 Train 进度

| Epoch | 方法 | train loss | train IoU | epoch time |
|---:|---|---:|---:|---:|
| 6 | Clean baseline | 1.20700 | 0.73372 | 0:11:03.061632 |
| 6 | LiftTrack | 1.19891 | 0.73484 | 0:07:34.492453 |
| 7 | Clean baseline | 1.21548 | 0.73189 | 0:11:02.697799 |
| 7 | LiftTrack | 1.17840 | 0.73829 | 0:07:34.637275 |
| 8 | Clean baseline | 1.18995 | 0.73647 | 0:11:02.426803 |
| 8 | LiftTrack | 1.17833 | 0.73845 | 0:07:34.523496 |

LiftTrack 在 epoch 6 的 loss/IoU 相对 baseline 变化为 `-0.00809/+0.00112`，epoch 7 为 `-0.03708/+0.00640`，epoch 8 为 `-0.01162/+0.00198`。三轮训练时间分别缩短 `31.45%/31.44%/31.38%`。这些数据继续支持训练优化和效率优势，但属于 train split 证据，不替代 epoch-10 validation 或最终 LasHeR benchmark。训练已自动进入 epoch 9。

### 13.4 BiLift 交换强度趋势与表述边界

epoch 1--7 整轮最终诊断中，`x2r_update_ratio` 从 `0.10731` 增长到 `0.20792`，`r2x_update_ratio` 从 `0.13013` 增长到 `0.31589`，`difference_ratio` 从 `1.01863` 增长到 `1.07895`。同期损失持续下降、IoU 上升，并且输出和 checkpoint 审计未发现非有限值，因此当前证据支持 BiLift 学到了非平凡的双向补充更新，不支持数值发散判断。

`r2x_update_ratio > x2r_update_ratio` 只表示 RGB 对 X 分支的相对更新更大，不能单独证明 RGB 更可靠或交换具有质量自适应性。`difference_ratio > 1` 表示交换后两分支差异范数略有扩大，因此不得将当前机制表述为“直接缩小模态差异”。当前可验证的方法表述是：可逆、低秩、非平凡的双向补充交换。“可靠性自适应”必须等待受控模态退化实验证据。

epoch 8 的最终 `x2r/r2x/difference_ratio` 为 `0.20697/0.31927/1.08087`，相比 epoch 7 的 `0.20792/0.31589/1.07895` 基本稳定，没有继续单调放大。

### 13.5 Epoch-10 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.16808 | 1.16786 | -0.00022 |
| train IoU | 0.74039 | 0.73959 | -0.00080 |
| validation loss | 1.15312 | 1.17550 | +0.02238 |
| validation IoU | 0.74558 | 0.74102 | -0.00456 |
| train epoch time | 0:11:02.963484 | 0:07:34.365747 | -31.47% |
| validation epoch time | 0:05:00.181873 | 0:03:35.181965 | -28.32% |

LiftTrack epoch-10 validation IoU 比自身 epoch 5 的 `0.74033` 提高 `0.00069`，因此当前最佳 validation checkpoint 更新为 epoch 10。但它比同协议 clean baseline epoch 10 低 `0.00456`，已超过 pilot 阶段使用的 `-0.002` 参考容差，是明确的正式精度风险。这一中期节点不改变预声明的选权规则：继续训练，仅在所有 validation 节点中选最高 IoU，不使用 LasHeR test 调参或挑选 epoch。

epoch-10 checkpoint 为：

```text
/mnt/tipro4t/seatrack_train_runs/lifttrack_full_seed0_20260712_020400/checkpoints/rgbt_lifttrack/SEATrack_ep0010.pth.tar
```

文件大小为 372,751,495 bytes，比同轮 baseline checkpoint 小 5,518,798 bytes（1.46%）；SHA-256 为 `069c81fc628e0ad7cfa240d2947a4b7a7aa84f96d8e6ac42ebbf0a07087df893`。加载审计确认 epoch 为 10，250 个浮点模型 tensor 全部有限，`rng_state` 完整包含 Python、NumPy、Torch 和 CUDA 状态。训练已自动进入 epoch 11。

epoch-10 validation 总损失差 `+0.02238` 不是单一项异常：按训练损失权重还原后，GIoU、L1 和 location 对 LiftTrack-baseline 差额的贡献分别约为 `+0.00962/+0.00525/+0.00754`，三项均变差。训练端总 loss 虽然仅差 `-0.00022`，但 LiftTrack 的原始 GIoU/L1 分别差 `+0.00103/+0.00024`，只是被 location loss 的 `-0.00348` 抵消。因此当前风险更接近中期定位能力的广泛差距，不应归因于某一个 loss head。若后续 validation 不能追回，优先审查低参数容量与交换层位置，而不是只调整某一损失权重。

epoch 11 的整轮 train loss/IoU 为 `1.14426/0.74503`，同轮 baseline 为 `1.18252/0.73751`，变化为 `-0.03826/+0.00752`；训练时间 `0:07:34.388659` 对 `0:11:02.790188`，缩短 `31.44%`。这说明 epoch 10 的训练端轻微落后未形成持续性崩坏，但 train 回升不能证明 validation 已恢复，需等待 epoch 15 validation 验证。训练已进入 epoch 12。

epoch 12 的 train loss/IoU 为 `1.14971/0.74387`，baseline 为 `1.16309/0.74160`，变化 `-0.01338/+0.00227`，因此 epoch 11 的回升至少延续了一轮。前 12 轮配对汇总中，LiftTrack 的 train loss `12/12` 全部低于 baseline，train IoU `11/12` 高于 baseline；平均 loss 差为 `-0.02007`，平均 IoU 差为 `+0.00356`。这证明训练优化优势是整体趋势，但 epoch-10 validation 风险表明它尚未稳定转化为泛化优势。

epoch 13 的 train loss/IoU 为 `1.14177/0.74505`，baseline 为 `1.15015/0.74411`，变化 `-0.00838/+0.00094`。此时 `x2r/r2x/difference_ratio` 为 `0.23191/0.34644/1.09585`，相比 epoch 12 的 `0.22105/0.34577/1.09559`，只有 X-to-RGB 更新幅度明显增加，而分支差异比基本稳定。当前没有“交换持续放大且性能同步下降”的证据。

### 13.6 Epoch-15 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.14654 | 1.12445 | -0.02209 |
| train IoU | 0.74414 | 0.74807 | +0.00393 |
| validation loss | 1.15917 | 1.17395 | +0.01478 |
| validation IoU | 0.74422 | 0.74235 | -0.00187 |
| train epoch time | 0:11:03.119805 | 0:07:34.537527 | -31.45% |
| validation epoch time | 0:04:59.807106 | 0:03:35.036634 | -28.28% |

epoch-15 validation IoU 比 LiftTrack epoch 10 的 `0.74102` 提高 `0.00133`，当前最佳 checkpoint 更新为 epoch 15。它与同协议 baseline epoch 15 的差距从 epoch 10 的 `-0.00456` 收窄至 `-0.00187`，重新进入 `-0.002` pilot 参考容差内，但仍未形成 validation 精度领先。该结果支持继续按预声明协议训练，不支持提前宣称方法有效。

同轮比较不等于公平选权比较。截至 epoch 15，LiftTrack 在已有 validation 节点中的最优值为 `0.74235@15`，baseline 在相同训练进度范围内的最优值为 `0.74558@10`，最优对最优的差距仍为 `-0.00323`，未进入 `-0.002` 参考容差。已完成的 baseline 60-epoch 运行最优值为 `0.75075@50`，相对 LiftTrack 当前最优值高 `0.00840`；因 LiftTrack 尚未训练到同等阶段，该值只作为最终待追赶目标，不用于当前提前判负。

epoch-15 checkpoint 大小为 372,836,423 bytes，比同轮 baseline 小 5,518,606 bytes（1.46%）；SHA-256 为 `32a51bd33bc7e126043f17a1c371fe11a7441637191cb6491e6693bf59f175dc`。加载审计确认 epoch 为 15，250 个浮点模型 tensor 全部有限，Python、NumPy、Torch 和 CUDA RNG 状态完整。训练已进入 epoch 16。

epoch 16 的 train loss/IoU 为 `1.12151/0.74853`，baseline 为 `1.14332/0.74489`，变化 `-0.02181/+0.00364`。`x2r/r2x/difference_ratio` 从 epoch-15 validation 的 `0.24869/0.38246/1.12290` 回落至训练整轮的 `0.24114/0.35974/1.10217`，而 train IoU 仍保持优势；但因 train/validation 样本分布不同，这只足以否定“交换幅度越大则性能越高”的简单跨 split 单调解释，不能用来推断因果。

### 13.7 Epoch-20 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.13229 | 1.11425 | -0.01804 |
| train IoU | 0.74699 | 0.75023 | +0.00324 |
| validation loss | 1.15065 | 1.15402 | +0.00337 |
| validation IoU | 0.74711 | 0.74550 | -0.00161 |
| train epoch time | 0:11:02.560625 | 0:07:34.413203 | -31.42% |
| validation epoch time | 0:04:58.991592 | 0:03:34.965157 | -28.10% |

epoch-20 validation IoU 比 LiftTrack epoch 15 的 `0.74235` 提高 `0.00315`，当前最佳 checkpoint 更新为 epoch 20。截至相同训练进度，baseline 的最佳 validation 也为 `0.74711@20`，因此公平的最佳对最佳差距为 `-0.00161`，首次进入 `-0.002` 参考容差。相对 baseline 完整训练最优 `0.75075@50` 仍差 `-0.00525`，但 LiftTrack 尚未到同等训练阶段。

这一结果表明方法已达到预先设定的中期 validation 精度近似门槛，同时保持大幅效率优势；但它仍未证明精度领先，也不替代最终最佳 validation 选权和 LasHeR benchmark。

epoch-20 checkpoint 大小为 372,921,287 bytes，比同轮 baseline 小 5,518,414 bytes（1.46%）；SHA-256 为 `3c8d24af53cfdef339d9dd0c156f3fe8b9efa63a59e24637e2ce1933afb937aa`。加载审计确认 epoch 为 20，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练已进入 epoch 21。

### 13.8 Epoch-25 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.09999 | 1.08918 | -0.01081 |
| train IoU | 0.75249 | 0.75456 | +0.00207 |
| validation loss | 1.13711 | 1.17122 | +0.03411 |
| validation IoU | 0.74844 | 0.74349 | -0.00495 |
| train epoch time | 0:11:02.810370 | 0:07:33.044398 | -31.65% |
| validation epoch time | 0:04:58.984417 | 0:03:34.328471 | -28.31% |

epoch-25 训练指标仍优于同轮 baseline，但 validation IoU 为 `0.74349`，比自身当前最优 `0.74550@20` 低 `0.00201`，比 baseline epoch 25 低 `0.00495`。因此不更新最优权重，继续保留 epoch 20。截至 epoch 25，baseline 在相同训练范围内的最优值为 `0.74844@25`，LiftTrack 最优对 baseline 最优的公平差距为 `-0.00294`，已离开 `-0.002` 参考容差。

该结果表明更低的训练 loss 和更高的训练 IoU 没有稳定转化为 validation 收益，存在 epoch 20 后的泛化回退。按预声明协议继续训练并观察后续 validation 节点，不因单个下降节点提前停止，也不使用 LasHeR test 挑选 epoch。

epoch-25 checkpoint 大小为 373,006,151 bytes，比同轮 baseline 小 5,518,286 bytes（1.46%）；SHA-256 为 `f90c8b0eb66cabb5c7bcc790f1dffadcef55d20217d6a7283ee2ee792d0182c3`。加载审计确认 epoch 为 25，250 个浮点模型 tensor 全部有限，Python、NumPy、Torch 和 CUDA RNG 状态完整。训练已进入 epoch 26。

### 13.9 Epoch-30 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.10443 | 1.07157 | -0.03286 |
| train IoU | 0.75241 | 0.75788 | +0.00547 |
| validation loss | 1.14220 | 1.15281 | +0.01061 |
| validation IoU | 0.74779 | 0.74584 | -0.00195 |
| train epoch time | 0:11:02.908388 | 0:07:32.194737 | -31.79% |
| validation epoch time | 0:04:58.269475 | 0:03:34.246984 | -28.17% |

epoch-30 validation IoU 为 `0.74584`，比原最优 `0.74550@20` 高 `0.00034`，因此当前最优 checkpoint 更新为 epoch 30。它与同轮 baseline 的差距为 `-0.00195`，重新进入 `-0.002` 参考容差；但截至相同训练进度，baseline 最优仍为 `0.74844@25`，因此最优对最优差距为 `-0.00260`，尚未进入参考容差。相对 baseline 完整训练最优 `0.75075@50` 仍差 `-0.00491`。

epoch 25 的回落在 epoch 30 得到恢复，说明单节点下降不足以判定持续过拟合。当前证据支持“效率优势稳定、validation 精度接近 baseline”，仍不支持“精度领先”。

epoch-30 checkpoint 大小为 373,091,015 bytes，比同轮 baseline 小 5,518,094 bytes（1.46%）；SHA-256 为 `705721999817638c471fb0d0f8b27c0d4c99f39a95a9ce0f771c5a38d8b6252c`。加载审计确认 epoch 为 30，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 31。

### 13.10 Epoch-35 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.07788 | 1.06940 | -0.00848 |
| train IoU | 0.75654 | 0.75811 | +0.00157 |
| validation loss | 1.16053 | 1.16021 | -0.00032 |
| validation IoU | 0.74628 | 0.74530 | -0.00098 |
| train epoch time | 0:11:02.675717 | 0:07:32.266907 | -31.75% |
| validation epoch time | 0:04:58.973487 | 0:03:33.859201 | -28.47% |

epoch-35 validation IoU 为 `0.74530`，比当前最优 `0.74584@30` 低 `0.00054`，因此不更新最优 checkpoint。它与同轮 baseline 的差距仅 `-0.00098`，但截至 epoch 35 的最优对最优差距仍为 `0.74584@30 - 0.74844@25 = -0.00260`。因此可以表述为同轮精度接近，不能表述为相同训练范围内已追平 baseline 最优。

epoch-35 checkpoint 大小为 373,175,879 bytes，比同轮 baseline 小 5,517,966 bytes（1.46%）；SHA-256 为 `820aaa5fa6269cc826afb670b38e116e3eba574a817b025532f85af2d6d336c3`。加载审计确认 epoch 为 35，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 36。

### 13.11 Epoch-40 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.07770 | 1.06799 | -0.00971 |
| train IoU | 0.75707 | 0.75898 | +0.00191 |
| validation loss | 1.14836 | 1.16580 | +0.01744 |
| validation IoU | 0.74759 | 0.74420 | -0.00339 |
| train epoch time | 0:11:02.887419 | 0:07:32.020542 | -31.81% |
| validation epoch time | 0:04:58.574111 | 0:03:34.163280 | -28.27% |

epoch-40 validation IoU 为 `0.74420`，比当前最优 `0.74584@30` 低 `0.00164`，不更新最优 checkpoint。同轮 baseline 为 `0.74759`，差距 `-0.00339`；截至 epoch 40 的最优对最优差距仍为 `-0.00260`。训练 loss/IoU 继续优于 baseline，但未转化为该节点的 validation 收益，是需要在 epoch 45 以及 epoch 48 学习率衰减后继续核验的泛化风险。

epoch-40 validation 的 `x2r/r2x/difference_ratio` 为 `0.28439/0.43540/1.17020`，交换诊断比 epoch 35 增大，同时 validation IoU 下降。这仅是相关变化，既不足以证明交换幅度导致回落，也不足以支持模态可靠性自适应的因果表述。

epoch-40 checkpoint 大小为 373,260,743 bytes，比同轮 baseline 小 5,517,838 bytes（1.46%）；SHA-256 为 `ccbf66a39ca5e34bb3459eaf8ad410737c4e622ff2cb4f3da64faba1677189ee`。加载审计确认 epoch 为 40，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 41。

### 13.12 Epoch-45 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.08029 | 1.04851 | -0.03178 |
| train IoU | 0.75643 | 0.76177 | +0.00534 |
| validation loss | 1.14928 | 1.17264 | +0.02336 |
| validation IoU | 0.74724 | 0.74429 | -0.00295 |
| train epoch time | 0:11:02.442130 | 0:07:31.987384 | -31.77% |
| validation epoch time | 0:04:58.707707 | 0:03:34.023010 | -28.35% |

epoch-45 validation IoU 为 `0.74429`，比当前最优 `0.74584@30` 低 `0.00155`，不更新最优 checkpoint。同轮 baseline 为 `0.74724`，差距 `-0.00295`；截至 epoch 45 的最优对最优差距仍为 `-0.00260`。这是 epoch 48 学习率衰减前的最后一个 validation 节点，将与 epoch 50/55 直接对照，判断低学习率是否缓解训练与 validation 的分离。

epoch-45 validation 的 `x2r/r2x/difference_ratio` 为 `0.32200/0.43202/1.18190`，交换诊断仍高于早期节点，但不能据此推断因果。当前最稳妥的结论是：效率优势已稳定验证，训练拟合优于 baseline，但 validation 精度只是接近而非领先。

epoch-45 checkpoint 大小为 373,345,607 bytes，比同轮 baseline 小 5,517,646 bytes（1.46%）；SHA-256 为 `55962216462c5e73d56e4b3b94603e7dd529053c484c8b454a76112a1936f3fd`。加载审计确认 epoch 为 45，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 46。

### 13.13 Epoch-50 Validation/Checkpoint 与学习率衰减核验

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.04869 | 1.03277 | -0.01592 |
| train IoU | 0.76271 | 0.76489 | +0.00218 |
| validation loss | 1.13603 | 1.16454 | +0.02851 |
| validation IoU | 0.75075 | 0.74499 | -0.00576 |
| train epoch time | 0:11:02.918278 | 0:07:32.005104 | -31.81% |
| validation epoch time | 0:04:59.035045 | 0:03:34.238582 | -28.36% |

epoch-50 checkpoint 中 optimizer param group 的实际学习率为 `4e-5`，相比 epoch-45 checkpoint 的 `4e-4` 降低 10 倍，证明 epoch 48 的 StepLR 衰减已正确执行。epoch-50 validation IoU 为 `0.74499`，比衰减前 epoch 45 的 `0.74429` 回升 `0.00070`，但仍比当前最优 `0.74584@30` 低 `0.00085`，因此不更新最优 checkpoint。

同轮 baseline 在 epoch 50 刷新为 `0.75075`，LiftTrack 同轮差距为 `-0.00576`；截至相同训练进度，最优对最优差距为 `0.74584@30 - 0.75075@50 = -0.00491`。因此低学习率目前只带来小幅 validation 恢复，尚未缓解与 baseline 的正式精度差距。效率优势仍然稳定，但“有效结果”的精度部分仍需 epoch 55/60 和最终 benchmark 验证。

epoch-50 checkpoint 大小为 373,430,471 bytes，比同轮 baseline 小 5,517,518 bytes（1.46%）；SHA-256 为 `ed7536ba88680c76580afee3ca47f72b19633a0b12c5cb474e9768d758850d02`。加载审计确认 epoch 为 50，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 51。

### 13.14 Epoch-55 Validation/Checkpoint 节点

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.04113 | 1.03005 | -0.01108 |
| train IoU | 0.76340 | 0.76596 | +0.00256 |
| validation loss | 1.13489 | 1.16121 | +0.02632 |
| validation IoU | 0.75064 | 0.74539 | -0.00525 |
| train epoch time | 0:11:02.362793 | 0:07:32.042883 | -31.76% |
| validation epoch time | 0:04:59.176642 | 0:03:34.634278 | -28.26% |

epoch-55 validation IoU 为 `0.74539`，比 epoch 50 的 `0.74499` 再回升 `0.00040`，但仍比当前最优 `0.74584@30` 低 `0.00045`，因此不更新最优 checkpoint。同轮 baseline 为 `0.75064`，差距 `-0.00525`；截至 epoch 55，baseline 最优仍为 `0.75075@50`，最优对最优差距仍为 `-0.00491`。

checkpoint 中 optimizer LR 仍为 `4e-5`。低学习率阶段从 epoch 45 的 `0.74429` 逐步恢复到 epoch 55 的 `0.74539`，但恢复幅度尚不足以超过 epoch 30，也未缩小与 baseline 最优的差距。只剩 epoch 60 最终 validation 节点；完成后才能按预声明规则冻结唯一 benchmark checkpoint。

epoch-55 checkpoint 大小为 373,515,335 bytes，比同轮 baseline 小 5,517,326 bytes（1.46%）；SHA-256 为 `53003b10fa5a6fca5f27c7fd9125ab873a605217318c0bf04ebe89dda951e075`。加载审计确认 epoch 为 55，250 个浮点模型 tensor 全部有限，四类 RNG 状态完整。训练继续进入 epoch 56。

### 13.15 Epoch-60 最终 Validation 与权重冻结

| 指标 | Clean baseline | LiftTrack | Lift - baseline |
|---|---:|---:|---:|
| train loss | 1.03658 | 1.02548 | -0.01110 |
| train IoU | 0.76432 | 0.76615 | +0.00183 |
| validation loss | 1.13977 | 1.16406 | +0.02429 |
| validation IoU | 0.74817 | 0.74488 | -0.00329 |
| train epoch time | 0:11:02.651659 | 0:07:32.235632 | -31.76% |
| validation epoch time | 0:04:58.833791 | 0:03:34.352561 | -28.27% |

epoch-60 validation IoU 为 `0.74488`，比 epoch 55 低 `0.00051`，比全程最优 `0.74584@30` 低 `0.00096`，因此不更新最优 checkpoint。按预声明的 LasHeR validation 选权规则，60 轮训练结束后唯一冻结用于 benchmark 的权重为 `SEATrack_ep0030.pth.tar`，其 validation IoU 为 `0.74584`。测试集结果不参与 epoch 选择。

LiftTrack 最优 validation IoU 比 clean baseline 最优 `0.75075@50` 低 `0.00491`。最终证据支持：LiftTrack 在同一训练协议下将训练耗时稳定降低约 31.8%、validation 耗时降低约 28.3%，训练拟合指标略优；但 validation 精度没有追平 baseline，不能宣称精度提升。下一步仅对冻结的 epoch-30 权重执行一次 LasHeR benchmark，检验效率收益是否伴随可接受的测试集精度。

epoch-60 checkpoint 大小为 373,600,263 bytes；SHA-256 为 `8448492dd14fc2458f23c71458899e26fa978f55eb6316cb941e79e506b37da7`。加载审计确认 epoch 为 60，250 个浮点模型 tensor 全部有限，Python、NumPy、Torch 和 CUDA RNG 状态完整，optimizer LR 为 `4e-5`。训练进程于 2026-07-12 12:19:03 正常完成并退出。

### 13.16 冻结 Epoch-30 LasHeR Benchmark

唯一冻结的 epoch-30 权重按与 clean baseline 相同的 LasHeR 245 序列、4 worker 和指标实现完成正式 benchmark。跟踪进程退出码为 0，总耗时 1749.34 秒；raw result 为 245/245，`missing=0`，未发现 Traceback、RuntimeError、OOM、checkpoint missing/unexpected key 或结果写入失败。

| 指标 | Clean baseline ep50 | LiftTrack ep30 | Lift - baseline |
|---|---:|---:|---:|
| PR20 | 0.703055 | 0.562638 | -0.140417 |
| NPR20 | 0.669015 | 0.524230 | -0.144785 |
| NPR AUC | 0.604298 | 0.476457 | -0.127841 |
| SR/AUC | 0.565194 | 0.448720 | -0.116474 |

LiftTrack 在四项指标上均显著低于 clean baseline，既未通过“至少一项不低于 baseline”的最低门禁，也未达到论文强度目标。该下降远大于 validation 最优差距 `-0.00491`，说明当前 BiLift 训练目标在 LasHeR validation 上表现接近，但没有迁移到 test benchmark；当前最稳妥结论是测试集泛化失败，而不是精度接近。

checkpoint 由测试 tracker 使用 `strict=True` 完整加载，配置启用了 BiLift layers `[5, 9]`、rank 8；结果数、注释匹配和异常扫描均通过。因此没有证据将大幅下降归因于漏载权重或不完整评测。当前 LiftTrack 形式应拒绝，不应继续 RGBT234 或多 seed 扩展，也不应基于 test 结果改选 epoch。后续若继续研究，应先设计独立的受控模态退化诊断与更强的 validation/test 泛化约束；这些属于新实验，不在本轮启动。

按用户要求，benchmark 完成后未启动后续 GPU 工作；检查时相关训练/评测进程均已退出，GPU compute process 为空，显存 `0 MiB`、利用率 `0%`。
