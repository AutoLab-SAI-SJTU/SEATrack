---
title: "GRATrack 阶段 1-2 实验记录"
type: "experiment-log"
topic: "多模态目标跟踪"
created: 2026-07-09
status: "running"
source_plan: "knowledge_base/GRATrack-实验实施方案.md"
---

# GRATrack 阶段 1-2 实验记录

## 1. 当前目标

执行 `knowledge_base/GRATrack-实验实施方案.md` 中的阶段 1 和阶段 2：

1. **阶段 1：诊断日志接入，不改变模型行为**  
   目标是在训练 status/TensorBoard 中记录 `GRA/*`、`Gate/*`、`Router/*` 等诊断量，并确认 baseline 行为不被 RGAE 改写。

2. **阶段 2：V0，GRA + RGAE**  
   目标是开启 per-head RGAE，用 template-search response agreement 控制双向 attention exchange，并验证训练可以稳定运行。

## 2. 环境与入口

当前默认 shell 的 `/usr/bin/python 3.14.4` 缺少训练依赖，因此使用第 0 阶段历史日志中实际跑通的环境：

```text
Python: /home/yufan/code/SEATrack/.venv/bin/python
torch: 2.12.1+cu130
CUDA: available
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
```

阶段配置：

```text
experiments/seatrack/rgbt_gratrack_stage1.yaml
experiments/seatrack/rgbt_gratrack_v0b.yaml
```

## 3. 已完成验证

### 2026-07-09：代码与配置静态检查

结果：

```text
python -m py_compile:
  lib/models/layers/attn.py
  lib/models/layers/attn_blocks.py
  lib/models/seatrack/vit_ci.py
  lib/models/seatrack/seatrack.py
  lib/train/actors/seatrack.py
  lib/config/seatrack/config.py
  lib/train/base_functions.py
  tracking/train.py
  lib/train/run_training.py

Status: PASS
```

配置加载：

```text
rgbt_gratrack_stage1: ENABLED=False, DIAGNOSTICS=True, LAYERS=[1,3,5,7,9,11], RHO_MIN=0.1
rgbt_gratrack_v0b:    ENABLED=True,  DIAGNOSTICS=True, LAYERS=[1,3,5,7,9,11], RHO_MIN=0.1

Status: PASS
```

### 2026-07-09：CEBlock_AP 随机前向

使用真实 ViT-B hidden dim `C=768` 进行最小前向验证。

结果：

```text
stage1 ok keys=17 rho=0.10000000149011612
v0b    ok keys=19 rho=0.10000023990869522

Status: PASS
```

结论：

- 阶段 1 诊断模式能返回 `GRA/*` 和 `Router/*` 标量。
- V0-b 模式能额外返回 `Gate/x2r_mean` 和 `Gate/r2x_mean`。
- 使用 `C=32` 的测试会失败，因为当前 HMoE 内部 LoRP 固定按 ViT-B 的 768 维初始化；这属于测试设置不匹配，不是阶段 1/2 接入错误。

### 2026-07-09：完整 backbone 小尺寸随机前向

使用 `vit_base_patch16_224_ce`，输入：

```text
template: [1, 6, 64, 64]
search:   [1, 6, 128, 128]
GRA layers: [1, 3]
```

结果：

```text
stage1 ok feat=(1, 64, 768) stats=17 rho=0.10000000149011612
v0b    ok feat=(1, 64, 768) stats=19 rho=0.10000000149011612

Status: PASS
```

结论：

- `vit_ci -> CEBlock_AP -> aux_dict['gratrack_stats'] -> actor status` 的核心数据链路已具备可运行基础。
- `recover_tokens` 在 `CAT_MODE='direct'` 下最终输出 search token，当前小尺寸 search grid 为 \(8 \times 8 = 64\)，因此输出 `[1,64,768]` 是预期行为。

### 2026-07-09：阶段 1 短训练通路验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_stage1_short \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_stage1_short_20260709 \
  --mode single
```

结果：

```text
Status: PASS
Batch: 1 / 1
Loss/total: 16.10210
Loss/giou: 1.30670
Loss/l1: 0.30520
Loss/location: 11.96270
IoU: 0.00000
GRA/rho_mean: 0.10000
GRA/rho_raw_mean: 0.00000
GRA/agreement_mean: 0.00000
GRA/c_rgb_mean: 0.00050
GRA/c_x_mean: 0.00013
```

日志与产物：

```text
Train log:
/mnt/tipro4t/seatrack_train_runs/gratrack_stage1_short_20260709/logs/seatrack-rgbt_gratrack_stage1_short.train.log

Checkpoint:
/mnt/tipro4t/seatrack_train_runs/gratrack_stage1_short_20260709/checkpoints/rgbt_gratrack_stage1_short/SEATrack_ep0001.pth.tar

Model size:
total=92.970537M, trainable=0.636324M
```

结论：

- `GRA/*` 已进入 trainer status/train log。
- `AttnMoE/*/Router/*` 与 `FfnMoE/*/Router/*` 已进入 trainer status/train log。
- 阶段 1 配置下没有 `Gate/x2r_mean`、`Gate/r2x_mean`，符合 `MODEL.GRA.ENABLED=False` 的诊断模式预期。
- 真实 dataloader + actor + trainer 链路已跑通 1 个 batch，未出现 NaN、shape error 或显存异常。

### 2026-07-09：阶段 2 V0-b 短训练通路验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_v0b_short \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_short_20260709 \
  --mode single
```

结果：

```text
Status: PASS
Batch: 1 / 1
Loss/total: 15.83346
Loss/giou: 1.30645
Loss/l1: 0.30618
Loss/location: 11.68965
IoU: 0.00000
GRA/rho_mean: 0.10000
GRA/rho_raw_mean: 0.00000
GRA/agreement_mean: 0.00000
GRA/c_rgb_mean: 0.00051
GRA/c_x_mean: 0.00013
Gate/x2r_mean: 0.07311
Gate/r2x_mean: 0.07311
```

日志与产物：

```text
Train log:
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_short_20260709/logs/seatrack-rgbt_gratrack_v0b_short.train.log

Checkpoint:
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_short_20260709/checkpoints/rgbt_gratrack_v0b_short/SEATrack_ep0001.pth.tar

Model size:
total=92.970585M, trainable=0.636372M
```

结论：

- 开启 `MODEL.GRA.ENABLED=True` 后，`Gate/x2r_mean` 与 `Gate/r2x_mean` 已进入 trainer status/train log。
- V0-b 相比阶段 1 新增的 per-head RGAE 参数生效，trainable 参数从 `0.636324M` 增至 `0.636372M`。
- 真实 dataloader + actor + trainer 链路已跑通 1 个 batch，未出现 NaN、shape error 或显存异常。

### 2026-07-09：阶段 1 正式训练启动与早期监控

后台会话：

```text
tmux session: gratrack_stage1_20260709
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_stage1 \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_stage1_20260709 \
  --mode single
```

日志：

```text
Console log:
logs/gratrack_runs/gratrack_stage1_20260709.console.log

Train log:
/mnt/tipro4t/seatrack_train_runs/gratrack_stage1_20260709/logs/seatrack-rgbt_gratrack_stage1.train.log
```

启动状态：

```text
GPU memory: 16720 / 32607 MiB
GPU util: 98-100%
Epoch 1 batches: 1875
```

早期监控：

```text
[train: 1, 50 / 1875]
Loss/total: 1.65311
IoU: 0.65877
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00167
GRA/c_rgb_mean: 0.00764
GRA/c_x_mean: 0.00789

[train: 1, 100 / 1875]
Loss/total: 1.61504
IoU: 0.66293
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00164
GRA/c_rgb_mean: 0.00754
GRA/c_x_mean: 0.00767

[train: 1, 500 / 1875]
Loss/total: 1.49989
IoU: 0.68197
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00171
GRA/c_rgb_mean: 0.00790
GRA/c_x_mean: 0.00727

[train: 1, 550 / 1875]
Loss/total: 1.49122
IoU: 0.68353
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00172
GRA/c_rgb_mean: 0.00796
GRA/c_x_mean: 0.00729
```

结论：

- 阶段 1 正式训练已成功启动。
- `GRA/*` 与 `Router/*` 在正式配置下正常写入。
- 没有 `Gate/*` 指标，符合阶段 1 只诊断、不启用 RGAE 的目标。
- 训练稳定观察到 `550 / 1875` batch，loss 从 `1.65311` 降到 `1.49122`，IoU 从 `0.65877` 升到 `0.68353`。
- 该 run 已手动停止以释放唯一 GPU 给阶段 2 V0-b；阶段 1 当前结论是“诊断接入通过”，不是 60 epoch full baseline-equivalence 训练结果。
- 若论文最终需要 instrumentation-only full 对照，应单独补跑 `rgbt_gratrack_stage1` 完整训练和评测。

### 2026-07-09：阶段 2 V0-b 正式训练启动与早期监控

后台会话：

```text
tmux session: gratrack_v0b_20260709
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_v0b \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_20260709 \
  --mode single
```

日志：

```text
Console log:
logs/gratrack_runs/gratrack_v0b_20260709.console.log

Train log:
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_20260709/logs/seatrack-rgbt_gratrack_v0b.train.log
```

启动状态：

```text
GPU memory: 18180 / 32607 MiB
GPU util: 96-100%
Epoch 1 batches: 1875
```

早期监控：

```text
[train: 1, 50 / 1875]
Loss/total: 1.60055
IoU: 0.66775
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00157
GRA/c_rgb_mean: 0.00754
GRA/c_x_mean: 0.00775
Gate/x2r_mean: 0.07312
Gate/r2x_mean: 0.07313

[train: 1, 100 / 1875]
Loss/total: 1.57601
IoU: 0.66917
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00153
GRA/c_rgb_mean: 0.00744
GRA/c_x_mean: 0.00762
Gate/x2r_mean: 0.07312
Gate/r2x_mean: 0.07314

[train: 1, 500 / 1875]
Loss/total: 1.46456
IoU: 0.68895
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00158
GRA/c_rgb_mean: 0.00777
GRA/c_x_mean: 0.00760
Gate/x2r_mean: 0.07320
Gate/r2x_mean: 0.07320

[train: 1, 1000 / 1875]
Loss/total: 1.42929
IoU: 0.69494
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00169
GRA/c_rgb_mean: 0.00823
GRA/c_x_mean: 0.00801
Gate/x2r_mean: 0.07330
Gate/r2x_mean: 0.07328

[train: 1, 1875 / 1875]
Loss/total: 1.39271
IoU: 0.70113
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00187
GRA/c_rgb_mean: 0.00890
GRA/c_x_mean: 0.00857
Gate/x2r_mean: 0.07348
Gate/r2x_mean: 0.07345
Epoch 1 summary: epoch_time=0:11:38.270137

[train: 2, 100 / 1875]
Loss/total: 1.29145
IoU: 0.72239
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00220
GRA/c_rgb_mean: 0.01022
GRA/c_x_mean: 0.00951
Gate/x2r_mean: 0.07389
Gate/r2x_mean: 0.07382
```

结论：

- 阶段 2 V0-b 正式训练已成功启动。
- `GRA/*`、`Gate/*` 与 `Router/*` 在正式配置下正常写入。
- `Gate/x2r_mean` 和 `Gate/r2x_mean` 保持在约 `0.073`，说明 per-head RGAE 分支已生效。
- 训练稳定观察到 `500 / 1875` batch，loss 从 `1.60055` 降到 `1.46456`，IoU 从 `0.66775` 升到 `0.68895`。
- `1000 / 1875` batch 时继续稳定，loss `1.42929`，IoU `0.69494`，`Gate/*` 未塌缩或消失。
- 第 1 个 epoch 已完整跑完，epoch 末 loss `1.39271`，IoU `0.70113`，用时约 `11m38s`。
- 第 2 个 epoch 前 `100 / 1875` batch 继续下降，loss `1.29145`，IoU `0.72239`，`Gate/*` 小幅升至约 `0.0739`。
- 显存稳定约 `18180 / 32607 MiB`，吞吐约 `85-86 FPS`，相对阶段 1 显存增加约 `1.46 GiB`。
- 早期 loss、IoU、显存和吞吐未显示异常；下一步让 V0-b 保持后台运行，并继续观察 checkpoint 生成和后续 epoch 稳定性。

### 2026-07-09：阶段 2 V0-b 保存策略修正与重启

问题：

```text
原正式配置:
SAVE_EPOCH_INTERVAL=60
SAVE_LAST_N_EPOCH=1
```

影响：

- V0-b 需要约 `11m38s/epoch`。
- 若只在 epoch 60 保存，训练约 11 小时内没有可恢复 checkpoint。
- 当前 run 已证明 V0-b 可稳定进入 epoch 2，但不适合作为整晚长跑继续占用 GPU。

处理：

```text
experiments/seatrack/rgbt_gratrack_stage1.yaml:
  SAVE_EPOCH_INTERVAL=5
  SAVE_LAST_N_EPOCH=5

experiments/seatrack/rgbt_gratrack_v0b.yaml:
  SAVE_EPOCH_INTERVAL=5
  SAVE_LAST_N_EPOCH=5
```

旧 run：

```text
tmux session: gratrack_v0b_20260709
save_dir: /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_20260709
status: stopped intentionally after epoch 2 reached 350 / 1875
reason: restart with recoverable checkpoint cadence
```

新 run：

```text
tmux session: gratrack_v0b_ckpt5_20260709
save_dir: /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709
console log: logs/gratrack_runs/gratrack_v0b_ckpt5_20260709.console.log
train log: /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/logs/seatrack-rgbt_gratrack_v0b.train.log
status: running
```

新 run 配置确认：

```text
MODEL.GRA.ENABLED=True
MODEL.GRA.DIAGNOSTICS=True
TRAIN.SAVE_EPOCH_INTERVAL=5
TRAIN.SAVE_LAST_N_EPOCH=5
```

新 run 早期监控：

```text
[train: 1, 100 / 1875]
Loss/total: 1.57601
IoU: 0.66917
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00153
GRA/c_rgb_mean: 0.00744
GRA/c_x_mean: 0.00762
Gate/x2r_mean: 0.07312
Gate/r2x_mean: 0.07314

[train: 1, 200 / 1875]
Loss/total: 1.52360
IoU: 0.67701
GRA/rho_mean: 0.10001
GRA/agreement_mean: 0.00151
GRA/c_rgb_mean: 0.00744
GRA/c_x_mean: 0.00755
Gate/x2r_mean: 0.07313
Gate/r2x_mean: 0.07315

[train: 1, 1875 / 1875]
Loss/total: 1.39271
IoU: 0.70113
GRA/rho_mean: 0.10002
GRA/agreement_mean: 0.00187
GRA/c_rgb_mean: 0.00890
GRA/c_x_mean: 0.00857
Gate/x2r_mean: 0.07348
Gate/r2x_mean: 0.07345
Epoch 1 summary: epoch_time=0:11:38.720733

[train: 2, 150 / 1875]
Loss/total: 1.30325
IoU: 0.71752
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00221
GRA/c_rgb_mean: 0.01018
GRA/c_x_mean: 0.00952
Gate/x2r_mean: 0.07390
Gate/r2x_mean: 0.07383
```

结论：

- 旧 V0-b run 不是因训练异常停止；停止原因是保存频率不满足长跑恢复要求。
- 新 V0-b run 使用同一方法配置，但保存策略已改为每 5 epoch 落 checkpoint，更适合持续训练。
- 新 V0-b run 前 200 batch 与旧 run 指标对齐，说明保存策略修改没有改变模型路径或训练稳定性。
- 新 V0-b run 已完整通过 epoch 1 并进入 epoch 2。epoch 1 末 loss `1.39271`，IoU `0.70113`，`Gate/*` 仍稳定；system log 未出现 crash。
- 当前尚未到 checkpoint 保存点；第一枚 checkpoint 预计在 epoch 5 后生成。

### 2026-07-09：阶段 2 V0-b ckpt5 长跑继续监控

采样结果：

```text
[train: 3, 1875 / 1875]
Loss/total: 1.27310
IoU: 0.72102
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00225
GRA/c_rgb_mean: 0.01092
GRA/c_x_mean: 0.01077
Gate/x2r_mean: 0.07498
Gate/r2x_mean: 0.07492
Epoch 3 summary: epoch_time=0:11:37.864512

[train: 4, 300 / 1875]
Loss/total: 1.25194
IoU: 0.72567
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00227
GRA/c_rgb_mean: 0.01119
GRA/c_x_mean: 0.01093
Gate/x2r_mean: 0.07536
Gate/r2x_mean: 0.07532
GPU memory: 18180 / 32607 MiB
```

结论：

- ckpt5 run 已稳定完成 epoch 3，并进入 epoch 4。
- `Gate/*` 从 epoch 1 的约 `0.0735` 平滑升至 epoch 4 前段的约 `0.0753`，未出现突变、消失或发散。
- `GRA/agreement_mean` 从约 `0.0019` 升至约 `0.0023`，与训练推进方向一致，未观察到 `rho` 塌缩到 0/1。
- 显存仍稳定在约 `18.18 GiB`；下一关键门槛仍是 epoch 5 checkpoint。

### 2026-07-09：阶段 2 V0-b ckpt5 首个 checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0005.pth.tar
size: 361 MiB / 378194921 bytes
saved_at: 2026-07-09 04:22:53
```

epoch 5 validation：

```text
[val: 5, 1875 / 1875]
Loss/total: 1.18201
IoU: 0.73914
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00237
GRA/c_rgb_mean: 0.01140
GRA/c_x_mean: 0.01123
Gate/x2r_mean: 0.07676
Gate/r2x_mean: 0.07660
Epoch 5 val summary: epoch_time=0:05:05.534113
```

checkpoint 后继续训练：

```text
[train: 6, 700 / 1875]
Loss/total: 1.22750
IoU: 0.72963
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00219
GRA/c_rgb_mean: 0.01138
GRA/c_x_mean: 0.01118
Gate/x2r_mean: 0.07686
Gate/r2x_mean: 0.07668
GPU memory: 18180 / 32607 MiB
```

结论：

- 保存策略修正有效，epoch 5 checkpoint 已经落盘。
- epoch 5 validation loss/IoU 正常，`GRA/*`、`Gate/*`、`Router/*` 在 validation 阶段也正常写入。
- checkpoint 后训练自动进入 epoch 6，没有因保存或 validation 中断。
- 下一关键门槛是 epoch 10 checkpoint，同时继续观察 `Gate/*` 是否平滑变化、router entropy/load 是否出现专家塌缩。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 7 完成

epoch 7 结束：

```text
[train: 7, 1875 / 1875]
Loss/total: 1.22301
IoU: 0.73051
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00223
GRA/c_rgb_mean: 0.01189
GRA/c_x_mean: 0.01148
Gate/x2r_mean: 0.07757
Gate/r2x_mean: 0.07743
AttnMoE/template/Router/expert_load_max: 0.45263
AttnMoE/search/Router/expert_load_max: 0.46236
FfnMoE/template/Router/expert_load_max: 0.30601
FfnMoE/search/Router/expert_load_max: 0.31360
Epoch 7 train summary: epoch_time=0:11:37.789793
```

epoch 8 起步：

```text
[train: 8, 100 / 1875]
Loss/total: 1.21197
IoU: 0.73405
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00236
GRA/c_rgb_mean: 0.01233
GRA/c_x_mean: 0.01191
Gate/x2r_mean: 0.07786
Gate/r2x_mean: 0.07779
GPU memory: 18180 / 32607 MiB
```

结论：

- ckpt5 run 已稳定完成 epoch 7，并进入 epoch 8。
- `Gate/*` 从 epoch 5 validation 的约 `0.0767` 平滑升至 epoch 8 起步的约 `0.0778`，没有突跳、归零或饱和。
- epoch 6 后段一度偏高的 AttnMoE/search load 在 epoch 7 末回到约 `0.46`，FfnMoE load max 约 `0.31`，当前未观察到专家塌缩。
- 严格异常扫描仍未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；显存保持约 `18.18 GiB`。
- 下一关键门槛仍是 epoch 10 validation 与 `SEATrack_ep0010.pth.tar` 保存。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 10 checkpoint

epoch 10 训练结束：

```text
[train: 10, 1875 / 1875]
Loss/total: 1.17794
IoU: 0.73834
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00236
GRA/c_rgb_mean: 0.01247
GRA/c_x_mean: 0.01185
Gate/x2r_mean: 0.07913
Gate/r2x_mean: 0.07921
AttnMoE/template/Router/expert_load_max: 0.42712
AttnMoE/search/Router/expert_load_max: 0.43870
FfnMoE/template/Router/expert_load_max: 0.30364
FfnMoE/search/Router/expert_load_max: 0.31533
Epoch 10 train summary: epoch_time=0:11:37.978349
```

epoch 10 validation：

```text
[val: 10, 1875 / 1875]
Loss/total: 1.16361
IoU: 0.74343
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00259
GRA/c_rgb_mean: 0.01225
GRA/c_x_mean: 0.01161
Gate/x2r_mean: 0.07937
Gate/r2x_mean: 0.07947
AttnMoE/template/Router/expert_load_max: 0.46171
AttnMoE/search/Router/expert_load_max: 0.46972
FfnMoE/template/Router/expert_load_max: 0.34555
FfnMoE/search/Router/expert_load_max: 0.33086
Epoch 10 val summary: epoch_time=0:05:05.724120
```

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0010.pth.tar
size: 361 MiB / 378280681 bytes
saved_at: 2026-07-09 05:26:10
```

checkpoint 后继续训练：

```text
[train: 11, 50 / 1875]
Loss/total: 1.17091
IoU: 0.74012
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00243
GRA/c_rgb_mean: 0.01245
GRA/c_x_mean: 0.01191
Gate/x2r_mean: 0.07938
Gate/r2x_mean: 0.07947
GPU memory: 18180 / 32607 MiB
```

结论：

- epoch 10 checkpoint 已成功落盘，说明 `SAVE_EPOCH_INTERVAL: 5` 的可恢复保存策略连续两次有效。
- epoch 10 validation 相比 epoch 5 validation 继续改善：`Loss/total` 从 `1.18201` 降至 `1.16361`，`IoU` 从 `0.73914` 升至 `0.74343`。
- `Gate/*` 从 epoch 5 validation 的约 `0.0767` 平滑升至 epoch 10 validation 的约 `0.0794`，两方向仍基本同步。
- `GRA/rho_mean` 仍稳定在约 `0.10004`，未出现门控饱和；Router load max 在 validation 中最高约 `0.47`，未观察到专家塌缩。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；checkpoint 后训练自动进入 epoch 11。
- 下一关键门槛是 epoch 15 checkpoint；若需要缩短监督窗口，也可以先在 epoch 10 checkpoint 上安排阶段性评测。

### 2026-07-09：训练框架纯度与 OT 模块排查

排查范围：

```text
current process:
tracking/train.py --script seatrack --config rgbt_gratrack_v0b

config:
experiments/seatrack/rgbt_gratrack_v0b.yaml

checked paths:
lib/
tracking/
experiments/
```

排查结论：

- 当前正在跑的是阶段 2 V0-b，即 `MODEL.GRA.ENABLED=True`、`RGAE_ENABLED=True` 的 GRATrack 实验变体，不是干净 baseline。
- 当前运行配置与代码路径没有混入旧的 OT / Optimal Transport / Sinkhorn / Wasserstein / ProbAlign / transport loss 模块。
- `lib/train/actors/seatrack.py` 实际训练 loss 仍是 `giou + l1 + focal/location`，日志中的 `Loss/total` 也只由这三项组成。
- `lib/models/seatrack/seatrack.py` 的模型构建入口只从 `cfg.MODEL.GRA` 读取并传入 GRA/RGAE 参数，没有 OT 模块入口。
- repo 内按 `optimal transport`、`sinkhorn`、`wasserstein`、`prob_align`、`ot_loss`、`transport_loss` 等关键词检查，`lib/tracking/experiments` 未发现有效调用。
- actor 中有一段历史 `ce_loss` 代码是注释状态，不参与当前 loss；`train_script.py` 构造的 `ce` objective 当前未被执行到。

baseline 约束：

- 后续若要跑“干净 baseline”，不能使用当前 `rgbt_gratrack_v0b` run。
- baseline 配置应显式设置 `MODEL.GRA.ENABLED=False` 且 `MODEL.GRA.DIAGNOSTICS=False`，并保持 loss 只包含 `giou/l1/location`。
- 阶段 1 的 diagnostics-only 配置可用于行为等价诊断；正式 baseline 结果最好单独启动 clean baseline run，避免和 V0-b 的实验权重混淆。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 15 checkpoint

epoch 15 训练结束：

```text
[train: 15, 1875 / 1875]
Loss/total: 1.15489
IoU: 0.74241
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00229
GRA/c_rgb_mean: 0.01261
GRA/c_x_mean: 0.01232
Gate/x2r_mean: 0.08121
Gate/r2x_mean: 0.08138
AttnMoE/template/Router/expert_load_max: 0.38381
AttnMoE/search/Router/expert_load_max: 0.40295
FfnMoE/template/Router/expert_load_max: 0.27063
FfnMoE/search/Router/expert_load_max: 0.25714
Epoch 15 train summary: epoch_time=0:11:37.851684
```

epoch 15 validation：

```text
[val: 15, 1875 / 1875]
Loss/total: 1.17136
IoU: 0.74162
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00234
GRA/c_rgb_mean: 0.01198
GRA/c_x_mean: 0.01186
Gate/x2r_mean: 0.08141
Gate/r2x_mean: 0.08158
AttnMoE/template/Router/expert_load_max: 0.39392
AttnMoE/search/Router/expert_load_max: 0.35460
FfnMoE/template/Router/expert_load_max: 0.26478
FfnMoE/search/Router/expert_load_max: 0.22639
Epoch 15 val summary: epoch_time=0:05:05.688460
```

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0015.pth.tar
size: 361 MiB / 378366377 bytes
saved_at: 2026-07-09 06:29:26
```

checkpoint 后继续训练：

```text
[train: 16, 200 / 1875]
Loss/total: 1.14911
IoU: 0.74351
GRA/rho_mean: 0.10003
GRA/agreement_mean: 0.00223
GRA/c_rgb_mean: 0.01240
GRA/c_x_mean: 0.01200
Gate/x2r_mean: 0.08143
Gate/r2x_mean: 0.08160
GPU memory: 18180 / 32607 MiB
```

结论：

- epoch 15 checkpoint 已成功落盘，保存策略连续三次通过：`ep0005`、`ep0010`、`ep0015`。
- epoch 15 validation 相比 epoch 10 validation 有回落：`Loss/total` 从 `1.16361` 升至 `1.17136`，`IoU` 从 `0.74343` 降至 `0.74162`；但仍优于 epoch 5 validation 的 `Loss/total=1.18201`、`IoU=0.73914`。
- `Gate/*` 从 epoch 10 validation 的约 `0.0794` 平滑升至 epoch 15 validation 的约 `0.0815`，两方向仍同步。
- `GRA/rho_mean` 仍稳定在约 `0.10003-0.10004`，未出现门控饱和。
- Router load 在 epoch 15 validation 中更均衡，最高 load max 约 `0.39`，未观察到专家塌缩。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；checkpoint 后训练自动进入 epoch 16。
- 下一关键门槛是 epoch 20 checkpoint；从阶段性指标看，后续应同时保留 `ep0010` 和 `ep0015` 做测试集评测比较，不能只按训练 epoch 单调假设选择最后权重。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 20 checkpoint

epoch 20 训练结束：

```text
[train: 20, 1875 / 1875]
Loss/total: 1.14458
IoU: 0.74450
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00224
GRA/c_rgb_mean: 0.01278
GRA/c_x_mean: 0.01280
Gate/x2r_mean: 0.08264
Gate/r2x_mean: 0.08328
AttnMoE/template/Router/expert_load_max: 0.39699
AttnMoE/search/Router/expert_load_max: 0.41276
FfnMoE/template/Router/expert_load_max: 0.24591
FfnMoE/search/Router/expert_load_max: 0.23496
Epoch 20 train summary: epoch_time=0:11:37.715155
```

epoch 20 validation：

```text
[val: 20, 1875 / 1875]
Loss/total: 1.16846
IoU: 0.74316
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00247
GRA/c_rgb_mean: 0.01249
GRA/c_x_mean: 0.01230
Gate/x2r_mean: 0.08278
Gate/r2x_mean: 0.08346
AttnMoE/template/Router/expert_load_max: 0.45064
AttnMoE/search/Router/expert_load_max: 0.46685
FfnMoE/template/Router/expert_load_max: 0.25198
FfnMoE/search/Router/expert_load_max: 0.22450
Epoch 20 val summary: epoch_time=0:05:05.879145
```

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0020.pth.tar
size: 361 MiB / 378452137 bytes
saved_at: 2026-07-09 07:32:41
```

checkpoint 后继续训练：

```text
[train: 21, 100 / 1875]
Loss/total: 1.12834
IoU: 0.74967
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00230
GRA/c_rgb_mean: 0.01292
GRA/c_x_mean: 0.01282
Gate/x2r_mean: 0.08279
Gate/r2x_mean: 0.08347
```

结论：

- epoch 20 checkpoint 已成功落盘，保存策略连续四次通过：`ep0005`、`ep0010`、`ep0015`、`ep0020`。
- epoch 20 validation 相比 epoch 15 validation 恢复：`Loss/total` 从 `1.17136` 降至 `1.16846`，`IoU` 从 `0.74162` 升至 `0.74316`。
- 但 epoch 20 validation 仍未超过 epoch 10 validation：`Loss/total=1.16846` 高于 `1.16361`，`IoU=0.74316` 略低于 `0.74343`。当前阶段不能按最后 epoch 直接选权重，至少应保留 `ep0010`、`ep0015`、`ep0020` 进入后续测试集评测。
- `Gate/*` 继续平滑升至约 `0.083`，两方向同步，没有突变或饱和。
- `GRA/rho_mean` 仍稳定在约 `0.10004`；`GRA/agreement_mean` 在 validation 为 `0.00247`，处于此前波动范围。
- validation 的 AttnMoE search load max 升至 `0.46685`，高于 epoch 15，但仍未出现单专家占满式塌缩；后续 epoch 25 需继续观察 router load。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；checkpoint 后训练自动进入 epoch 21。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 25-40 checkpoint sweep

checkpoint：

| epoch | 文件 | size_bytes | saved_at |
| --- | --- | ---: | --- |
| 25 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0025.pth.tar` | 378537897 | 2026-07-09 08:35:55 |
| 30 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0030.pth.tar` | 378623593 | 2026-07-09 09:39:10 |
| 35 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0035.pth.tar` | 378709353 | 2026-07-09 10:42:24 |
| 40 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0040.pth.tar` | 378795049 | 2026-07-09 11:45:38 |

训练末端指标：

| epoch | Loss/total | IoU | GRA/rho_mean | Gate/x2r | Gate/r2x | Attn load max T/S | FFN load max T/S |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 25 | 1.11957 | 0.74888 | 0.10004 | 0.08392 | 0.08480 | 0.39766 / 0.39255 | 0.24062 / 0.22942 |
| 30 | 1.11798 | 0.74969 | 0.10004 | 0.08501 | 0.08609 | 0.37901 / 0.38411 | 0.23357 / 0.22682 |
| 35 | 1.09477 | 0.75331 | 0.10004 | 0.08592 | 0.08709 | 0.38184 / 0.38548 | 0.23456 / 0.21852 |
| 40 | 1.10204 | 0.75244 | 0.10004 | 0.08671 | 0.08802 | 0.41855 / 0.41678 | 0.22605 / 0.22367 |

validation 末端指标：

| epoch | Loss/total | IoU | GRA/rho_mean | GRA/agreement | Gate/x2r | Gate/r2x | Attn load max T/S | FFN load max T/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 25 | 1.15136 | 0.74626 | 0.10004 | 0.00245 | 0.08406 | 0.08494 | 0.39242 / 0.41122 | 0.21372 / 0.21941 |
| 30 | 1.15354 | 0.74542 | 0.10004 | 0.00262 | 0.08507 | 0.08618 | 0.41248 / 0.41403 | 0.26268 / 0.24859 |
| 35 | 1.16500 | 0.74418 | 0.10004 | 0.00244 | 0.08601 | 0.08721 | 0.42254 / 0.41508 | 0.26676 / 0.23552 |
| 40 | 1.15673 | 0.74554 | 0.10005 | 0.00256 | 0.08682 | 0.08809 | 0.44000 / 0.42979 | 0.21568 / 0.22085 |

当前继续训练状态：

```text
[train: 45, 1050 / 1875]
Loss/total: 1.10224
IoU: 0.75160
GRA/rho_mean: 0.10004
Gate/x2r_mean: 0.08745
Gate/r2x_mean: 0.08876
GPU memory/util/temp: 18180 / 32607 MiB, 95%, 61 C
```

checkpoint 保留语义复核：

- 配置仍是 `SAVE_EPOCH_INTERVAL=5`、`SAVE_LAST_N_EPOCH=5`。
- 代码实际判断是 `epoch > (max_epochs - save_last_n_epoch) or epoch % save_epoch_interval == 0`，即每 5 个 epoch 保存一次，并在训练最后 5 个 epoch 额外保存；没有删除旧 checkpoint 的清理逻辑。
- 当前目录实测保留 `ep0005` 至 `ep0040` 共 8 个 checkpoint，checkpoint 目录约 `2.9G`，`/mnt/tipro4t` 仍有约 `1.1T` 可用，暂不需要清理。

结论：

- checkpoint 保存策略连续八次通过：`ep0005`、`ep0010`、`ep0015`、`ep0020`、`ep0025`、`ep0030`、`ep0035`、`ep0040`。
- 在已经记录的 validation 点中，epoch 25 当前最好：`Loss/total=1.15136`、`IoU=0.74626`；它超过了此前 epoch 10 的 `Loss/total=1.16361`、`IoU=0.74343`。
- epoch 30 与 epoch 40 接近但未超过 epoch 25；epoch 35 validation 明显回落。后续测试集评测优先候选建议为 `ep0025`，同时保留 `ep0030`、`ep0040` 和早期强点 `ep0010` 做对照。
- 训练集指标从 epoch 25 到 epoch 40 整体下降 loss、提升 IoU，但 validation 并非单调提升，说明后续选权应以验证/测试结果为准，不能默认最后 epoch 最好。
- `Gate/*` 从 epoch 25 validation 的约 `0.084/0.085` 平滑升至 epoch 40 validation 的约 `0.087/0.088`，两方向同步，没有突变。
- `GRA/rho_mean` 稳定在 `0.10004-0.10005`；`GRA/agreement_mean` 在 `0.00244-0.00262` 区间波动，未出现异常漂移。
- Router load 仍未出现单专家塌缩；epoch 40 validation 的 Attn load max 达到 `0.44000/0.42979`，比 epoch 25 高，需要继续观察 epoch 45/50 是否继续升高。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；训练已进入 epoch 45。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 45-50 checkpoint sweep

checkpoint：

| epoch | 文件 | size_bytes | saved_at |
| --- | --- | ---: | --- |
| 45 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0045.pth.tar` | 378880809 | 2026-07-09 12:48:53 |
| 50 | `/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0050.pth.tar` | 378966569 | 2026-07-09 13:52:07 |

训练末端指标：

| epoch | Loss/total | IoU | GRA/rho_mean | Gate/x2r | Gate/r2x | Attn load max T/S | FFN load max T/S |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 45 | 1.10147 | 0.75233 | 0.10004 | 0.08750 | 0.08878 | 0.38446 / 0.39029 | 0.23929 / 0.23618 |
| 50 | 1.06853 | 0.75912 | 0.10004 | 0.08801 | 0.08929 | 0.35455 / 0.37587 | 0.20361 / 0.19601 |

validation 末端指标：

| epoch | Loss/total | IoU | GRA/rho_mean | GRA/agreement | Gate/x2r | Gate/r2x | Attn load max T/S | FFN load max T/S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 45 | 1.16287 | 0.74505 | 0.10005 | 0.00257 | 0.08759 | 0.08884 | 0.35911 / 0.37023 | 0.21399 / 0.20364 |
| 50 | 1.14237 | 0.74875 | 0.10004 | 0.00251 | 0.08802 | 0.08930 | 0.35427 / 0.37822 | 0.20775 / 0.20127 |

checkpoint 后继续训练：

```text
[train: 51, 150 / 1875]
Loss/total: 1.07809
IoU: 0.75735
GRA/rho_mean: 0.10004
Gate/x2r_mean: 0.08801
Gate/r2x_mean: 0.08930
GPU memory/util/temp: 18180 / 32607 MiB, 100%, 60 C
```

结论：

- epoch 45 validation 回落，没有超过 epoch 25/40。
- epoch 50 validation 刷新当前最佳：`Loss/total=1.14237`、`IoU=0.74875`，相比此前最佳 epoch 25 的 `Loss/total=1.15136`、`IoU=0.74626` 有明确提升。
- epoch 50 的 train loss/IoU 也显著优于此前记录点：`Loss/total=1.06853`、`IoU=0.75912`，但仍需要后续测试集验证是否转化为真实跟踪性能。
- epoch 50 validation 的 Router load max 降至 `0.35427/0.37822`，比 epoch 40 的 `0.44000/0.42979` 更均衡，专家塌缩风险降低。
- `Gate/*` 继续平滑升至约 `0.088/0.089`，`GRA/rho_mean` 仍稳定在约 `0.10004-0.10005`。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback；checkpoint 后训练自动进入 epoch 51。
- 当前测试集评测优先候选更新为 `ep0050`；建议同时保留 `ep0025`、`ep0040`、`ep0045` 作为对照。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 55 checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0055.pth.tar
size: 362 MiB / 379052265 bytes
saved_at: 2026-07-09 14:55:23
```

epoch 55 训练结束：

```text
[train: 55, 1875 / 1875]
Loss/total: 1.05689
IoU: 0.76019
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00228
GRA/c_rgb_mean: 0.01307
GRA/c_x_mean: 0.01316
Gate/x2r_mean: 0.08806
Gate/r2x_mean: 0.08936
AttnMoE/template/Router/expert_load_max: 0.37580
AttnMoE/search/Router/expert_load_max: 0.38206
FfnMoE/template/Router/expert_load_max: 0.20888
FfnMoE/search/Router/expert_load_max: 0.19691
Epoch 55 train summary: epoch_time=0:11:38.064355
```

epoch 55 validation：

```text
[val: 55, 1875 / 1875]
Loss/total: 1.14350
IoU: 0.74863
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00249
GRA/c_rgb_mean: 0.01247
GRA/c_x_mean: 0.01263
Gate/x2r_mean: 0.08807
Gate/r2x_mean: 0.08937
AttnMoE/template/Router/expert_load_max: 0.38299
AttnMoE/search/Router/expert_load_max: 0.38940
FfnMoE/template/Router/expert_load_max: 0.20489
FfnMoE/search/Router/expert_load_max: 0.19771
Epoch 55 val summary: epoch_time=0:05:05.685783
```

checkpoint 后继续训练：

```text
[train: 56, 800 / 1875]
Loss/total: 1.05029
IoU: 0.76202
GRA/rho_mean: 0.10004
Gate/x2r_mean: 0.08807
Gate/r2x_mean: 0.08937
GPU memory/util/temp: 18180 / 32607 MiB, 96%, 61 C
```

结论：

- epoch 55 checkpoint 成功落盘，训练自动进入 epoch 56。
- epoch 55 validation 接近但未超过 epoch 50：`Loss/total=1.14350` 高于 `1.14237`，`IoU=0.74863` 低于 `0.74875`。当前最佳候选仍是 `ep0050`。
- epoch 55 明显优于 epoch 25 的 `Loss/total=1.15136`、`IoU=0.74626`，因此可作为后续评测的强对照权重。
- `Gate/*` 与 `GRA/rho_mean` 继续平稳，未出现门控饱和。
- Router load max 为 `0.38299/0.38940`，比 epoch 50 稍高但仍无专家塌缩信号。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。
- 从 epoch 56 开始进入 `SAVE_LAST_N_EPOCH=5` 触发区间，预计 epoch 56、57、58、59、60 每轮都会保存 checkpoint。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 56 train-only checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0056.pth.tar
size: 362 MiB / 379069353 bytes
saved_at: 2026-07-09 15:07:02
```

epoch 56 训练结束：

```text
[train: 56, 1875 / 1875]
Loss/total: 1.05361
IoU: 0.76089
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00228
GRA/c_rgb_mean: 0.01302
GRA/c_x_mean: 0.01308
Gate/x2r_mean: 0.08807
Gate/r2x_mean: 0.08937
AttnMoE/template/Router/expert_load_max: 0.38193
AttnMoE/search/Router/expert_load_max: 0.38796
FfnMoE/template/Router/expert_load_max: 0.20599
FfnMoE/search/Router/expert_load_max: 0.19474
Epoch 56 train summary: epoch_time=0:11:37.887524
```

checkpoint 后继续训练：

```text
[train: 57, 150 / 1875]
Loss/total: 1.05888
IoU: 0.76159
GRA/rho_mean: 0.10004
Gate/x2r_mean: 0.08807
Gate/r2x_mean: 0.08938
```

结论：

- epoch 56 checkpoint 成功落盘，证明 `SAVE_LAST_N_EPOCH=5` 已开始生效。
- epoch 56 没有 validation，这是预期行为；当前配置 `VAL_EPOCH_INTERVAL=5`，因此最后阶段只有 epoch 60 会再次触发 validation。
- epoch 56 train-only 指标正常，`Gate/*`、`GRA/rho_mean` 与 Router load 都延续稳定状态。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 57 train-only checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0057.pth.tar
size: 362 MiB / 379086505 bytes
saved_at: 2026-07-09 15:18:40
```

epoch 57 训练结束：

```text
[train: 57, 1875 / 1875]
Loss/total: 1.06553
IoU: 0.75920
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00228
GRA/c_rgb_mean: 0.01297
GRA/c_x_mean: 0.01305
Gate/x2r_mean: 0.08808
Gate/r2x_mean: 0.08939
AttnMoE/template/Router/expert_load_max: 0.37271
AttnMoE/search/Router/expert_load_max: 0.38242
FfnMoE/template/Router/expert_load_max: 0.20782
FfnMoE/search/Router/expert_load_max: 0.19609
Epoch 57 train summary: epoch_time=0:11:37.660212
```

结论：

- epoch 57 checkpoint 成功落盘，训练自动进入 epoch 58。
- epoch 57 没有 validation，符合 `VAL_EPOCH_INTERVAL=5` 预期。
- train-only 指标较 epoch 56 略回落，但仍处于稳定区间；`Gate/*`、`GRA/rho_mean` 和 Router load 未出现异常。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 58 train-only checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0058.pth.tar
size: 362 MiB / 379103593 bytes
saved_at: 2026-07-09 15:30:18
```

epoch 58 训练结束：

```text
[train: 58, 1875 / 1875]
Loss/total: 1.05796
IoU: 0.76077
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00229
GRA/c_rgb_mean: 0.01301
GRA/c_x_mean: 0.01303
Gate/x2r_mean: 0.08809
Gate/r2x_mean: 0.08939
AttnMoE/template/Router/expert_load_max: 0.38104
AttnMoE/search/Router/expert_load_max: 0.38591
FfnMoE/template/Router/expert_load_max: 0.20338
FfnMoE/search/Router/expert_load_max: 0.19964
Epoch 58 train summary: epoch_time=0:11:38.290860
```

结论：

- epoch 58 checkpoint 成功落盘，训练自动进入 epoch 59。
- epoch 58 没有 validation，符合 `VAL_EPOCH_INTERVAL=5` 预期。
- train-only 指标重新回到 epoch 56 附近，`Gate/*` 与 `GRA/rho_mean` 平稳。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 59 train-only checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0059.pth.tar
size: 362 MiB / 379120681 bytes
saved_at: 2026-07-09 15:42:12
```

epoch 59 训练结束：

```text
[train: 59, 1875 / 1875]
Loss/total: 1.05378
Loss/giou: 0.25994
Loss/l1: 0.02421
Loss/location: 0.41288
IoU: 0.76146
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00227
GRA/c_rgb_mean: 0.01294
GRA/c_x_mean: 0.01290
Gate/x2r_mean: 0.08809
Gate/r2x_mean: 0.08941
AttnMoE/template/Router/expert_load_max: 0.38187
AttnMoE/search/Router/expert_load_max: 0.38725
FfnMoE/template/Router/expert_load_max: 0.20561
FfnMoE/search/Router/expert_load_max: 0.20485
Epoch 59 train summary: epoch_time=0:11:53.713767
```

结论：

- epoch 59 checkpoint 成功落盘，训练自动进入 epoch 60。
- epoch 59 没有 validation，符合 `VAL_EPOCH_INTERVAL=5` 预期；最终 validation 将在 epoch 60 结束后触发。
- train-only 指标继续稳定，`IoU` 达到 0.76146，`Gate/*` 与 `GRA/rho_mean` 未出现跳变。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。

### 2026-07-09：阶段 2 V0-b ckpt5 epoch 60 最终 validation 与 checkpoint

checkpoint：

```text
/mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0060.pth.tar
size: 362 MiB / 379138025 bytes
saved_at: 2026-07-09 15:59:05
```

epoch 60 训练结束：

```text
[train: 60, 1875 / 1875]
Loss/total: 1.05857
Loss/giou: 0.26138
Loss/l1: 0.02440
Loss/location: 0.41381
IoU: 0.76040
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00229
GRA/c_rgb_mean: 0.01300
GRA/c_x_mean: 0.01298
Gate/x2r_mean: 0.08810
Gate/r2x_mean: 0.08942
AttnMoE/template/Router/expert_load_max: 0.37857
AttnMoE/search/Router/expert_load_max: 0.38536
FfnMoE/template/Router/expert_load_max: 0.21008
FfnMoE/search/Router/expert_load_max: 0.20909
Epoch 60 train summary: epoch_time=0:11:46.653207
```

epoch 60 validation 结束：

```text
[val: 60, 1875 / 1875]
Loss/total: 1.15113
Loss/giou: 0.28286
Loss/l1: 0.02823
Loss/location: 0.44429
IoU: 0.74563
GRA/rho_mean: 0.10004
GRA/agreement_mean: 0.00249
GRA/c_rgb_mean: 0.01238
GRA/c_x_mean: 0.01245
Gate/x2r_mean: 0.08811
Gate/r2x_mean: 0.08943
AttnMoE/template/Router/expert_load_max: 0.38601
AttnMoE/search/Router/expert_load_max: 0.38760
FfnMoE/template/Router/expert_load_max: 0.20819
FfnMoE/search/Router/expert_load_max: 0.20798
Epoch 60 val summary: epoch_time=0:05:05.602545
```

结论：

- 阶段 2 V0-b ckpt5 正式长训练完整结束，system log 记录 `Finished training!`。
- epoch 60 checkpoint 成功落盘；checkpoint 目录共保留 16 个 checkpoint，包含 epoch 5/10/.../55 以及最终逐轮保存的 epoch 56/57/58/59/60。
- epoch 60 validation IoU 为 0.74563，低于 epoch 50 的 0.74875，也低于 epoch 55 的 0.74863；当前最佳 validation 候选仍是 `SEATrack_ep0050.pth.tar`，`SEATrack_ep0055.pth.tar` 可作为强对照候选。
- epoch 60 train-only 指标与 epoch 56-59 尾段一致，`Gate/*`、`GRA/rho_mean` 和 Router load 未出现崩溃或跳变。
- 严格异常扫描未发现 `NaN`、`RuntimeError`、OOM 或 Traceback。

### 2026-07-09：当前训练框架纯净性复核

复核范围：

```text
lib/
tracking/
experiments/
当前 run 训练日志、配置日志、系统日志、console log
当前训练进程命令
```

复核结论：

- 当前 `rgbt_gratrack_v0b` 训练没有混入旧 OT / Optimal Transport / Sinkhorn / ProbAlign 失败模块。
- 当前进程命令为 `tracking/train.py --script seatrack --config rgbt_gratrack_v0b`，配置文件为 `experiments/seatrack/rgbt_gratrack_v0b.yaml`。
- 代码检索未发现 active `transport_loss`、`ot_loss`、`sinkhorn`、`wasserstein` 或旧 ProbAlign 训练路径。
- Actor loss 仍是干净的 `giou + l1 + focal/location`，没有 OT loss 项。
- 当前日志指标只有 `Loss/giou`、`Loss/l1`、`Loss/location`、`GRA/*`、`Gate/*` 与 Router 统计，没有 `OT/*` 指标。
- 需要注意：当前 run 是阶段 2 V0-b 变体，`MODEL.GRA.ENABLED=True`、`RGAE_ENABLED=True`，不是 clean baseline。clean baseline 必须单独使用 `MODEL.GRA.ENABLED=False` 的配置启动，日志中不应出现 `GRA/*`、`Gate/*`、Router 诊断指标。

### 2026-07-09：阶段 2 V0-b ep0050 LasHeR benchmark

评测对象：

```text
variant: gratrack_v0b_ep0050_lasher_eval_20260709_160623
yaml: rgbt_gratrack_v0b
checkpoint: /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0050.pth.tar
raw results: /home/yufan/code/SEATrack-ProbAlign-VRE/RGBT_workspace/gratrack_v0b_ep0050_lasher_eval_20260709_160623/LasHeR
metrics: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/gratrack_v0b_ep0050_lasher_eval_20260709_160623/lasher_metrics.json
manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/gratrack_v0b_ep0050_lasher_eval_20260709_160623/run_manifest.json
```

评测结果：

```text
sequences: 245
missing: 0
precision_20: 0.6971046370309925
normalized_precision_20: 0.6600701219552605
normalized_precision_auc: 0.5972806050623618
success_auc: 0.5587946142971875
```

结论：

- LasHeR 245 个测试序列全部完成，raw result 文件数为 245，`missing=0`。
- 评测过程未发现 `Traceback`、`RuntimeError`、OOM 或缺文件错误。
- 当前 benchmark 使用的是阶段 2 V0-b 的最佳 validation 候选 `ep0050`，不是 clean baseline。
- 本地 `datasets/RGBT234` 当前没有可用序列，RGBT234 benchmark 暂不能在本机直接补跑；需要先补齐数据集路径或数据内容。

### 2026-07-09：RGB-T clean baseline 独立长训练启动

启动目的：

- 为 GRATrack V0-b 提供干净 baseline，对齐同骨干、同数据、同训练超参。
- 明确排除旧 OT / Sinkhorn / ProbAlign 失败模块。
- 明确关闭 GRA/RGAE/诊断，避免 baseline 日志被 `GRA/*`、`Gate/*` 或 Router 诊断污染。

运行信息：

```text
run_id: rgbt_clean_baseline_ckpt5_20260709
tmux: rgbt_clean_baseline_ckpt5_20260709
config: experiments/seatrack/rgbt_clean_baseline.yaml
save_dir: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709
manifest: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/run_manifest.yaml
console: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_runs/rgbt_clean_baseline_ckpt5_20260709.console.log
```

配置锁定：

```text
MODEL.GRA.ENABLED: False
MODEL.GRA.DIAGNOSTICS: False
MODEL.GRA.RGAE_ENABLED: False
TRAIN.EPOCH: 60
TRAIN.VAL_EPOCH_INTERVAL: 5
TRAIN.SAVE_EPOCH_INTERVAL: 5
TRAIN.SAVE_LAST_N_EPOCH: 5
```

启动后首轮观察：

```text
[train: 1, 350 / 1875]
Loss/total: 1.52283
Loss/giou: 0.36683
Loss/l1: 0.03991
Loss/location: 0.58961
IoU: 0.67810
```

epoch 1 结束：

```text
[train: 1, 1875 / 1875]
Loss/total: 1.41850
Loss/giou: 0.34370
Loss/l1: 0.03640
Loss/location: 0.54912
IoU: 0.69593
Epoch 1 train summary: epoch_time=0:11:03.594303
```

epoch 5 首个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0005.pth.tar
checkpoint_size: 378185557 bytes
```

```text
[train: 5, 1875 / 1875]
Loss/total: 1.23245
Loss/giou: 0.30116
Loss/l1: 0.02990
Loss/location: 0.48065
IoU: 0.72849
Epoch 5 train summary: epoch_time=0:11:02.696569
```

```text
[val: 5, 1875 / 1875]
Loss/total: 1.17456
Loss/giou: 0.28836
Loss/l1: 0.02881
Loss/location: 0.45378
IoU: 0.74061
Epoch 5 val summary: epoch_time=0:04:58.384676
```

结论：

- clean baseline 已进入 epoch 1 正常训练，GPU 占用约 16.4 GiB，训练吞吐约 90 FPS。
- epoch 1 已完整结束并自动进入 epoch 2；epoch 1 不保存 checkpoint 符合 `SAVE_EPOCH_INTERVAL=5` 预期。
- epoch 5 已完成首个 validation/checkpoint 节点，并自动进入 epoch 6；`SEATrack_ep0005.pth.tar` 已落盘。
- 日志只包含 `Loss/total`、`Loss/giou`、`Loss/l1`、`Loss/location` 和 `IoU`；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn` 或 `ProbAlign` 指标。
- Actor loss 仍是 `giou + l1 + focal/location`，没有 OT loss 项。
- 该 run 是后续与 `rgbt_gratrack_v0b` 对齐比较的 baseline，不应与阶段 1 diagnostic run 混用。

epoch 10 第二个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0010.pth.tar
checkpoint_size: 378270293 bytes
```

```text
[train: 10, 1875 / 1875]
Loss/total: 1.16808
Loss/giou: 0.28589
Loss/l1: 0.02777
Loss/location: 0.45746
IoU: 0.74039
Epoch 10 train summary: epoch_time=0:11:02.963484
```

```text
[val: 10, 1875 / 1875]
Loss/total: 1.15312
Loss/giou: 0.28271
Loss/l1: 0.02811
Loss/location: 0.44713
IoU: 0.74558
Epoch 10 val summary: epoch_time=0:05:00.181873
```

epoch 10 结论：

- clean baseline 已完成第二个 validation/checkpoint 节点，并自动进入 epoch 11。
- `SEATrack_ep0010.pth.tar` 已落盘，当前 manifest 的 latest/best validation checkpoint 均更新为 ep0010。
- epoch 10 validation IoU=0.74558，高于 epoch 5 validation IoU=0.74061，baseline 早期收敛正常。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 15 第三个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0015.pth.tar
checkpoint_size: 378355029 bytes
```

```text
[train: 15, 1875 / 1875]
Loss/total: 1.14654
Loss/giou: 0.28122
Loss/l1: 0.02724
Loss/location: 0.44791
IoU: 0.74414
Epoch 15 train summary: epoch_time=0:11:03.119805
```

```text
[val: 15, 1875 / 1875]
Loss/total: 1.15917
Loss/giou: 0.28421
Loss/l1: 0.02834
Loss/location: 0.44907
IoU: 0.74422
Epoch 15 val summary: epoch_time=0:04:59.807106
```

epoch 15 结论：

- clean baseline 已完成第三个 validation/checkpoint 节点，并自动进入 epoch 16。
- `SEATrack_ep0015.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0015。
- epoch 15 validation IoU=0.74422，低于 epoch 10 validation IoU=0.74558，因此当前 best validation checkpoint 仍保留 `SEATrack_ep0010.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 20 第四个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0020.pth.tar
checkpoint_size: 378439701 bytes
```

```text
[train: 20, 1875 / 1875]
Loss/total: 1.13229
Loss/giou: 0.27770
Loss/l1: 0.02679
Loss/location: 0.44295
IoU: 0.74699
Epoch 20 train summary: epoch_time=0:11:02.560625
```

```text
[val: 20, 1875 / 1875]
Loss/total: 1.15065
Loss/giou: 0.28106
Loss/l1: 0.02796
Loss/location: 0.44870
IoU: 0.74711
Epoch 20 val summary: epoch_time=0:04:58.991592
```

epoch 20 结论：

- clean baseline 已完成第四个 validation/checkpoint 节点，并自动进入 epoch 21。
- `SEATrack_ep0020.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0020。
- epoch 20 validation IoU=0.74711，高于 epoch 10 validation IoU=0.74558，因此当前 best validation checkpoint 更新为 `SEATrack_ep0020.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 25 第五个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0025.pth.tar
checkpoint_size: 378524437 bytes
```

```text
[train: 25, 1875 / 1875]
Loss/total: 1.09999
Loss/giou: 0.27037
Loss/l1: 0.02551
Loss/location: 0.43171
IoU: 0.75249
Epoch 25 train summary: epoch_time=0:11:02.810370
```

```text
[val: 25, 1875 / 1875]
Loss/total: 1.13711
Loss/giou: 0.27950
Loss/l1: 0.02782
Loss/location: 0.43904
IoU: 0.74844
Epoch 25 val summary: epoch_time=0:04:58.984417
```

epoch 25 结论：

- clean baseline 已完成第五个 validation/checkpoint 节点，并自动进入 epoch 26。
- `SEATrack_ep0025.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0025。
- epoch 25 validation IoU=0.74844，高于 epoch 20 validation IoU=0.74711，因此当前 best validation checkpoint 更新为 `SEATrack_ep0025.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 30 第六个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0030.pth.tar
checkpoint_size: 378609109 bytes
```

```text
[train: 30, 1875 / 1875]
Loss/total: 1.10443
Loss/giou: 0.27124
Loss/l1: 0.02575
Loss/location: 0.43323
IoU: 0.75241
Epoch 30 train summary: epoch_time=0:11:02.908388
```

```text
[val: 30, 1875 / 1875]
Loss/total: 1.14220
Loss/giou: 0.28016
Loss/l1: 0.02791
Loss/location: 0.44232
IoU: 0.74779
Epoch 30 val summary: epoch_time=0:04:58.269475
```

epoch 30 结论：

- clean baseline 已完成第六个 validation/checkpoint 节点，并自动进入 epoch 31。
- `SEATrack_ep0030.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0030。
- epoch 30 validation IoU=0.74779，低于 epoch 25 validation IoU=0.74844，因此当前 best validation checkpoint 仍保留 `SEATrack_ep0025.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 35 第七个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0035.pth.tar
checkpoint_size: 378693845 bytes
```

```text
[train: 35, 1875 / 1875]
Loss/total: 1.07788
Loss/giou: 0.26569
Loss/l1: 0.02483
Loss/location: 0.42233
IoU: 0.75654
Epoch 35 train summary: epoch_time=0:11:02.675717
```

```text
[val: 35, 1875 / 1875]
Loss/total: 1.16053
Loss/giou: 0.28254
Loss/l1: 0.02826
Loss/location: 0.45414
IoU: 0.74628
Epoch 35 val summary: epoch_time=0:04:58.973487
```

epoch 35 结论：

- clean baseline 已完成第七个 validation/checkpoint 节点，并自动进入 epoch 36。
- `SEATrack_ep0035.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0035。
- epoch 35 validation IoU=0.74628，低于 epoch 25 validation IoU=0.74844，因此当前 best validation checkpoint 仍保留 `SEATrack_ep0025.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 40 第八个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0040.pth.tar
checkpoint_size: 378778581 bytes
```

```text
[train: 40, 1875 / 1875]
Loss/total: 1.07770
Loss/giou: 0.26477
Loss/l1: 0.02467
Loss/location: 0.42482
IoU: 0.75707
Epoch 40 train summary: epoch_time=0:11:02.887419
```

```text
[val: 40, 1875 / 1875]
Loss/total: 1.14836
Loss/giou: 0.28044
Loss/l1: 0.02795
Loss/location: 0.44773
IoU: 0.74759
Epoch 40 val summary: epoch_time=0:04:58.574111
```

epoch 40 结论：

- clean baseline 已完成第八个 validation/checkpoint 节点，并自动进入 epoch 41。
- `SEATrack_ep0040.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0040。
- epoch 40 validation IoU=0.74759，低于 epoch 25 validation IoU=0.74844，因此当前 best validation checkpoint 仍保留 `SEATrack_ep0025.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 45 第九个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0045.pth.tar
checkpoint_size: 378863253 bytes
```

```text
[train: 45, 1875 / 1875]
Loss/total: 1.08029
Loss/giou: 0.26566
Loss/l1: 0.02486
Loss/location: 0.42466
IoU: 0.75643
Epoch 45 train summary: epoch_time=0:11:02.442130
```

```text
[val: 45, 1875 / 1875]
Loss/total: 1.14928
Loss/giou: 0.28129
Loss/l1: 0.02805
Loss/location: 0.44645
IoU: 0.74724
Epoch 45 val summary: epoch_time=0:04:58.707707
```

epoch 45 结论：

- clean baseline 已完成第九个 validation/checkpoint 节点，并自动进入 epoch 46。
- `SEATrack_ep0045.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0045。
- epoch 45 validation IoU=0.74724，低于 epoch 25 validation IoU=0.74844，也低于 epoch 40 validation IoU=0.74759，因此当前 best validation checkpoint 仍保留 `SEATrack_ep0025.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 50 第十个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0050.pth.tar
checkpoint_size: 378947989 bytes
saved_at: 2026-07-10 03:11:44 +0800
```

```text
[train: 50, 1875 / 1875]
Loss/total: 1.04869
Loss/giou: 0.25768
Loss/l1: 0.02378
Loss/location: 0.41443
IoU: 0.76271
Epoch 50 train summary: epoch_time=0:11:02.918278
```

```text
[val: 50, 1875 / 1875]
Loss/total: 1.13603
Loss/giou: 0.27687
Loss/l1: 0.02748
Loss/location: 0.44490
IoU: 0.75075
Epoch 50 val summary: epoch_time=0:04:59.035045
```

epoch 50 结论：

- clean baseline 已完成第十个 validation/checkpoint 节点，并自动进入 epoch 51。
- `SEATrack_ep0050.pth.tar` 已落盘，manifest 的 latest checkpoint 更新为 ep0050。
- epoch 50 validation IoU=0.75075，高于此前 clean baseline 最佳 epoch 25 的 IoU=0.74844，因此当前 best validation checkpoint 更新为 `SEATrack_ep0050.pth.tar`。
- 纯净性扫描在排除仓库路径名中的 `ProbAlign` 字符串后无命中；未出现 `GRA/*`、`Gate/*`、`OT/*`、`Sinkhorn`、`ProbAlign` 指标或异常关键字。

epoch 55 第十一个 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0055.pth.tar
checkpoint_size: 379032661 bytes
saved_at: 2026-07-10 04:11:57 +0800
```

```text
[train: 55, 1875 / 1875]
Loss/total: 1.04113
Loss/giou: 0.25693
Loss/l1: 0.02360
Loss/location: 0.40929
IoU: 0.76340
Epoch 55 train summary: epoch_time=0:11:02.362793
```

```text
[val: 55, 1875 / 1875]
Loss/total: 1.13489
Loss/giou: 0.27720
Loss/l1: 0.02755
Loss/location: 0.44271
IoU: 0.75064
Epoch 55 val summary: epoch_time=0:04:59.176642
```

epoch 55 结论：

- clean baseline 已完成第十一个 validation/checkpoint 节点，并自动进入 epoch 56。
- epoch 55 validation IoU=0.75064，比 epoch 50 的 0.75075 低 0.00011，因此 best validation checkpoint 仍为 `SEATrack_ep0050.pth.tar`。
- 日志未出现异常或被禁用模块的指标。

epoch 60 最终 validation/checkpoint 节点：

```text
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0060.pth.tar
checkpoint_size: 379117397 bytes
saved_at: 2026-07-10 05:12:11 +0800
```

```text
[train: 60, 1875 / 1875]
Loss/total: 1.03658
Loss/giou: 0.25608
Loss/l1: 0.02359
Loss/location: 0.40645
IoU: 0.76432
Epoch 60 train summary: epoch_time=0:11:02.651659
```

```text
[val: 60, 1875 / 1875]
Loss/total: 1.13977
Loss/giou: 0.27965
Loss/l1: 0.02776
Loss/location: 0.44166
IoU: 0.74817
Epoch 60 val summary: epoch_time=0:04:58.833791
```

epoch 60 结论：

- clean baseline 60 epoch 正式长训练已结束；system log 记录 `Finished training!`，console 退出码为 0，训练 tmux 已退出。
- epoch 56、57、58、59、60 的最终逐轮 checkpoint 均已保存；latest checkpoint 为 `SEATrack_ep0060.pth.tar`。
- epoch 60 validation IoU=0.74817，低于 epoch 50 的 0.75075；best validation checkpoint 保持 `SEATrack_ep0050.pth.tar`。
- 严格异常与纯净性扫描未发现 GRA、Gate、OT、Sinkhorn、ProbAlign 指标，也未发现 Traceback、RuntimeError、OOM 或其他错误签名。

### 2026-07-10：RGB-T clean baseline ep0050 LasHeR benchmark 启动

正式评测信息：

```text
variant: rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916
yaml: rgbt_clean_baseline
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0050.pth.tar
dataset: LasHeR
threads: 4
tmux: rgbt_clean_baseline_ep0050_eval
run manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916/run_manifest.yaml
```

目的与口径：

- 该评测是阶段 2 的公平 benchmark 门禁，与 V0-b ep0050 使用同一评测脚本、同一 LasHeR 245 序列列表、同一指标实现和 4 个跟踪线程。
- 只替换模型配置和 checkpoint；clean baseline 使用 `rgbt_clean_baseline`，V0-b 使用 `rgbt_gratrack_v0b`。
- validation IoU 不参与最终 benchmark 优劣判定；完成后直接比较 PR20、NPR20、NPR_AUC 和 SR/AUC。
- 首次启动时因手工写入的 variant 时间戳与实际时间不一致，在完成 5/245 序列后立即停止；对应日志和 raw results 已统一加 `aborted_` 前缀，正式评测不复用这些结果。

103/245 序列临时同子集预警（非门禁结果，更新此前 83 序列快照）：

| 方法 | PR20 | NPR20 | NPR_AUC | SR/AUC |
|---|---:|---:|---:|---:|
| clean baseline | 0.73121 | 0.69035 | 0.62246 | 0.58149 |
| V0-b | 0.71729 | 0.67214 | 0.60805 | 0.56921 |
| baseline - V0-b | +0.01393 | +0.01821 | +0.01441 | +0.01228 |

说明：该子集由当前优先完成的短序列构成，存在完成顺序偏差，只用于提前发现 V0-b 可能低于 baseline 的风险；不能替代 245/245 全量结果，也不用于最终阶段 2 门禁判定。

全量 245/245 正式结果：

```text
completed_at: 2026-07-10 17:03:50 +0800
tracking_seconds: 3269.7575
raw results: /home/yufan/code/SEATrack-ProbAlign-VRE/RGBT_workspace/rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916/LasHeR
metrics: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916/lasher_metrics.json
json manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916/run_manifest.json
yaml manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_clean_baseline_ep0050_lasher_eval_20260710_160916/run_manifest.yaml
```

| 方法 | PR20 | NPR20 | NPR_AUC | SR/AUC | missing |
|---|---:|---:|---:|---:|---:|
| clean baseline ep0050 | 0.703055 | 0.669015 | 0.604298 | 0.565194 | 0/245 |
| V0-b ep0050 | 0.697105 | 0.660070 | 0.597281 | 0.558795 | 0/245 |
| baseline - V0-b | +0.005950 | +0.008945 | +0.007017 | +0.006400 | - |

阶段 2 benchmark 门禁结论：

- clean baseline 与 V0-b 使用同一 LasHeR 245 序列、同一评测脚本、同一指标实现和相同 4 线程协议；结果文件均完整，`missing=0`。
- V0-b 在 PR20、NPR20、NPR_AUC 和 SR/AUC 四项指标上均低于 clean baseline，因此“V0-b 在至少一个主 benchmark 上不低于 baseline”的当前 RGB-T 门禁失败。
- 该结论与 raw rho 尺度诊断一致：旧 V0-b 的实际动态调制被 `RHO_MIN` 淹没，新增计算没有形成有效 response-conditioned exchange。
- 决策为 `no-go to stage 3`：不直接启动 V1；先完成中心化余弦 agreement + Gini 幅值置信度的 V0-c 设计审批、实现、短训练和同协议复验。

### 2026-07-10：阶段 2 V0-b 配对效率门禁

测量前锁定的门限：

- 同设备、batch=1、同一真实 LasHeR template/search，V0-b 平均前向延迟增幅不超过 10%。
- PyTorch peak allocated memory 增幅不超过 15%。
- 所有输出有限，无 OOM。

协议与产物：

```text
device: NVIDIA GeForce RTX 5090
sample: LasHeR/10runone, template frame 0, search frame 10
timing scope: complete SEATrack network forward including backbone and box head
load order: baseline -> V0-b -> V0-b -> baseline
warmup: 20 iterations per model load
measurement: 2 trials x 50 iterations per load, 200 iterations per variant
manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_efficiency/stage2_v0b_vs_baseline_20260710_170806/run_manifest.yaml
results: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_efficiency/stage2_v0b_vs_baseline_20260710_170806/efficiency_results.json
```

| 指标 | Clean baseline | V0-b | V0-b 相对变化 |
|---|---:|---:|---:|
| mean latency | 12.5748 ms | 13.3435 ms | +6.11% |
| median latency | 12.5715 ms | 13.3034 ms | +5.82% |
| p90 latency | 12.5967 ms | 13.3267 ms | +5.80% |
| FPS（由 mean latency 计算） | 79.52 | 74.94 | -5.76% |
| peak allocated memory | 454111232 B | 454213632 B | +0.023% |
| parameters | 93154857 | 93155001 | +144 |

交叉证据：

- LasHeR 全量 4-worker raw tracking 总耗时：baseline 3269.76 s，V0-b 3407.07 s，V0-b 增加 4.20%，与单样本配对前向的 6.11% 同方向。
- 正式训练日志中的 V0-b 吞吐约比 baseline 低 5.3%，也与该测量一致。
- 两个变体共 400 次测量输出均为有限值，GPU 测量后无残留进程。

结论：V0-b 通过预先锁定的延迟和推理显存效率门禁。阶段 2 的 no-go 原因是四项 LasHeR 精度均低于 baseline，且 legacy rho 动态范围失效；不是计算或显存开销超过门限。

### 2026-07-10：阶段 1 diagnostics-only LasHeR 全量等价性评测启动

```text
variant: rgbt_stage1_diagnostics_ep0050_lasher_eval_20260710_171107
yaml: rgbt_gratrack_stage1
checkpoint: /mnt/tipro4t/seatrack_train_runs/rgbt_clean_baseline_ckpt5_20260709/checkpoints/rgbt_clean_baseline/SEATrack_ep0050.pth.tar
dataset: LasHeR
threads: 4
tmux: rgbt_stage1_diagnostics_ep0050_eval
manifest: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_stage1_diagnostics_ep0050_lasher_eval_20260710_171107/run_manifest.yaml
```

目的与门禁：

- 使用与 clean baseline 完全相同的 ep0050 checkpoint，只把配置切换为 `rgbt_gratrack_stage1`；该配置启用 diagnostics，但关闭 GRA/RGAE 行为改变。
- 全量完成后要求序列集合精确一致、245 个 raw box 数组逐元素完全一致、四项汇总指标差值为 0、`missing=0`。
- 若全部满足，可直接证明 instrumentation-only 不改变 LasHeR benchmark 行为；若不满足，则定位首个不一致序列和帧，不能用“基本一致”模糊通过。

全量完成结果：

```text
completed_at: 2026-07-10 18:10:16 +0800
tracking_seconds: 3544.7377
raw results: /home/yufan/code/SEATrack-ProbAlign-VRE/RGBT_workspace/rgbt_stage1_diagnostics_ep0050_lasher_eval_20260710_171107/LasHeR
metrics: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_stage1_diagnostics_ep0050_lasher_eval_20260710_171107/lasher_metrics.json
equivalence report: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_eval/rgbt_stage1_diagnostics_ep0050_lasher_eval_20260710_171107/equivalence_report.json
```

| 检查项 | 结果 |
|---|---:|
| sequence set exact | true |
| sequences compared | 245 |
| prediction rows compared | 220703 |
| shape mismatch | 0 |
| numeric non-exact files | 0 |
| byte non-exact files | 0 |
| max absolute difference | 0.0 |
| metric deltas | 全部 0.0 |
| missing | 0/245 |

阶段 1 benchmark 结论：

- diagnostics-only 与 clean baseline 的 PR20、NPR20、NPR_AUC、SR/AUC 均完全相同，分别为 0.703055、0.669015、0.604298、0.565194。
- 245 个 raw result 文件不仅数值逐元素相等，文本字节也全部相同，证明 instrumentation-only 没有改变预测轨迹。
- stage1 全量 tracking 总耗时 3544.74 s，baseline 为 3269.76 s，diagnostics 带来 8.41% 时间开销；正式训练短跑的显存采样约从 16.4 GiB 增至 16.72 GiB，未出现 OOM。
- 阶段 1 的“与 baseline 指标基本一致”和“不改变模型行为”门禁可标记通过。严格日志规范仍有一个独立缺口：Gate/Sparse 在禁用时没有显式写出 null/default 占位。

### 2026-07-10：阶段 0-2 严格完成度审计

独立审计按实施方案逐条核对当前工作树、日志、checkpoint、评测结果和 manifest，结论如下：

| 阶段 | 严格状态 | 主要依据或缺口 |
|---|---|---|
| 阶段 0 | 已完成 | RGB-T、RGB-D、RGB-E 入口、训练闭环、指标、FPS、显存和日志路径均有现存证据 |
| 阶段 1 | 核心验收完成，日志规范部分完成 | diagnostics-only 与 clean baseline 的 245 个 raw results、220703 行框和四项指标完全一致；仅禁用分支的 Gate/Sparse 默认占位未写出 |
| 阶段 2 | 未完成严格验收 | 缺同协议 clean baseline benchmark、退化样本解释性和预定义效率门限；`rho_raw_mean` 长期接近 0 |

阶段 2 的关键风险：

- V0-b 日志中的应用后 `GRA/rho_mean` 约为 0.10004，但 `GRA/rho_raw_mean` 长期约为 0.00002-0.00005；前者主要由 `RHO_MIN=0.1` 托底，不能直接证明学习得到的门控未塌缩。
- 现有 V0-b LasHeR benchmark 已完成 245/245 序列，但在 clean baseline 同协议评测完成前，不能用 validation IoU 或历史 baseline 数值替代公平比较。
- 现有配对训练日志显示 V0-b 相比 clean baseline 约为吞吐 -5.3%、step time +6.8%，显存采样约增加 1.78 GiB；开销已量化，但仍需先定义论文可接受门限。

后续门禁顺序：

1. 完成当前 clean baseline LasHeR 同协议评测并直接比较四项 benchmark 指标。
2. 追查 raw rho 近零的公式尺度与实现原因，必要时按方案尝试 `rho.detach()`、后层启用或调整响应统计，而不是仅依赖 `rho_min`。
3. 补最小受控退化样本导出，验证 `agreement`、`rho_raw` 和应用后 rho 的可解释变化。
4. 定义 FPS、step time 和显存的接受阈值，形成明确 go/no-go 判定后再进入阶段 3。

阶段 1 同 checkpoint 前向等价性补充证据：

```text
artifact: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_diagnostics/stage1_forward_equivalence_20260710_162100.json
checkpoint: clean baseline ep0050
sample: LasHeR/10runone frame index 10
baseline config: rgbt_clean_baseline
diagnostics config: rgbt_gratrack_stage1
```

| 输出 | shape | exact equal | max abs diff |
|---|---|---|---:|
| pred_boxes | [1,1,4] | true | 0.0 |
| score_map | [1,1,16,16] | true | 0.0 |
| size_map | [1,2,16,16] | true | 0.0 |
| offset_map | [1,2,16,16] | true | 0.0 |

结论：在相同权重和输入下，diagnostics-only 路径对模型输出逐元素无影响，只新增 `GRA/*` 与 Router 统计。这显著加强了阶段 1 的“不改变模型行为”证据，但不能替代完整 benchmark；Gate/Sparse 禁用分支默认占位仍未闭环，因此阶段 1 严格状态暂保持“部分完成”。

### 2026-07-10：V0-b raw rho 尺度与最小受控退化探针

探针对象与产物：

```text
config: rgbt_gratrack_v0b
checkpoint: /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_ckpt5_20260709/checkpoints/rgbt_gratrack_v0b/SEATrack_ep0050.pth.tar
dataset: LasHeR
sequences: 10runone, baggirl, bikeboywithumbrella
frame policy: each sequence frame index 10 (zero-based)
artifact: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_diagnostics/v0b_ep0050_gra_probe_20260710_161501/gra_probe.json
```

先对真实帧读取六个 GRA 层的逐层统计，排除跨层平均掩盖问题：

- layer 1：`rho_raw=0`，应用后 `rho=0.10000000`。
- layer 3/5/7/9：`rho_raw` 约为 `1.58e-6` 至 `1.51e-5`。
- layer 11：`rho_raw=9.10e-5`，应用后 `rho=0.10008188`。
- 即使最高层，动态分量相对 `RHO_MIN=0.1` 也很小。

三序列、六个 GRA 层的均值：

| Search 条件 | agreement | c_rgb | c_x | rho_raw | 应用后 rho |
|---|---:|---:|---:|---:|---:|
| clean | 0.002209 | 0.011581 | 0.011145 | 3.234e-5 | 0.10002911 |
| RGB blur 31x31 | 0.001843 | 0.009287 | 0.010903 | 2.390e-5 | 0.10002151 |
| X blur 31x31 | 0.001788 | 0.011377 | 0.009725 | 2.421e-5 | 0.10002179 |
| RGB missing | 0.000086 | 0.001280 | 0.011446 | 4.411e-7 | 0.10000040 |
| X missing | 0.000157 | 0.011132 | 0.001982 | 1.090e-6 | 0.10000098 |
| X shift 16 px | 0.001914 | 0.011619 | 0.010464 | 2.684e-5 | 0.10002416 |

分析结论：

- GRA 统计具有正确方向性：RGB/X 强模糊使 `rho_raw` 分别下降 26.09%/25.14%，RGB/X 缺失使其分别下降 98.64%/96.63%，X 平移 16 px 使其下降 16.99%。
- 但 clean 与 RGB missing 的应用后 rho 只相差 `2.87e-5`，`Gate/x2r_mean` 和 `Gate/r2x_mean` 也只发生同量级变化，实际调制几乎由 `RHO_MIN` 和学习到的方向 scale 决定。
- 实现与设计文档公式一致。以 epoch 60 的 `agreement=0.00249`、`c_rgb=0.01238`、`c_x=0.01245` 代入，公式给出 `rho_raw=3.091e-5`、应用后 `rho=0.10002782`，与日志一致。
- 因此根因是当前乘法公式对 ViT 的弥散 attention 产生严重尺度压缩，不是切片错误，也不是简单的跨层平均问题。当前 V0-b 更接近“固定下限 gate + 学习方向 scale”，不足以支撑动态 response-gated 的核心机制主张。
- 在进入阶段 3 前必须先校准或重定义 raw gate，并做同样的退化探针与短训练复验；单纯把 `RHO_MIN` 提高到 0.2 只会进一步增强固定门控，不能解决动态性问题。

候选 gate 方案探针：

1. **固定半饱和校准**：保留旧 `rho_raw`，使用 `rho_cal = rho_raw / (rho_raw + tau)`，取 `tau=1e-3`。在现有三序列探针上，clean 应用后 rho 为 0.12822，RGB/X missing 为 0.10040/0.10098。优点是改动最小；缺点是 `tau` 依赖旧 raw 的数据尺度，方法解释性较弱。
2. **中心化余弦 agreement + Gini 幅值置信度**：先计算 `cosine(p_rgb-u, p_x-u)`，再乘 `(c_rgb*c_x)^(1/4)`。由于当前 `c` 是归一化平方 L2 concentration，四次方根乘积等价于两模态 concentration 幅值的几何平均，具有明确尺度解释。
3. **可学习 per-layer temperature**：让每层学习校准尺度。适应性最高，但容易让模型把温度学成固定 shortcut，弱化机制可解释性，不适合作为第一修复版本。

方案 2 的只读运行时探针产物：

```text
artifact: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_diagnostics/v0b_candidate_gate_probe_20260710_161844/candidate_gate_probe.json
formula: rho_candidate = centered_cosine * (c_rgb * c_x)^(1/4)
```

| Search 条件 | centered cosine | confidence strength | candidate raw | 应用后 rho |
|---|---:|---:|---:|---:|
| clean | 0.51221 | 0.10417 | 0.05477 | 0.14930 |
| RGB blur 31x31 | 0.50864 | 0.09765 | 0.05049 | 0.14544 |
| X blur 31x31 | 0.46990 | 0.09980 | 0.04837 | 0.14353 |
| RGB missing | 0.12818 | 0.05712 | 0.00692 | 0.10622 |
| X missing | 0.13310 | 0.06097 | 0.00952 | 0.10857 |
| X shift 16 px | 0.48170 | 0.10268 | 0.05031 | 0.14528 |

当前推荐方案 2：它在不引入可学习 shortcut 或数据集特定阈值的情况下，把 clean gate 恢复到约 0.15，并在单模态缺失时回落到约 0.11。正式实现前仍需用户批准设计；第一轮短训练建议 `DETACH_RHO=True`，避免归一化与四次方根在近均匀响应处放大梯度，待稳定性确认后再单独比较可反传版本。

20 序列扩展探针：

```text
artifact: /home/yufan/code/SEATrack-ProbAlign-VRE/logs/gratrack_diagnostics/v0b_candidate_gate_probe20_20260710_162347/candidate_gate_probe20.json
sampling: sorted LasHeR test sequence names 上等间隔抽取 20 个序列
conditions: clean, RGB blur 31x31, X blur 31x31, RGB missing, X missing
```

| 条件 | candidate rho mean | p10 | median | p90 |
|---|---:|---:|---:|---:|
| clean | 0.14621 | 0.12938 | 0.14659 | 0.16010 |
| RGB blur 31x31 | 0.13882 | 0.12114 | 0.13955 | 0.15470 |
| X blur 31x31 | 0.13437 | 0.11988 | 0.13502 | 0.14942 |
| RGB missing | 0.11049 | 0.10679 | 0.11002 | 0.11500 |
| X missing | 0.10949 | 0.10481 | 0.10997 | 0.11476 |

配对门禁结果：

- `clean > RGB missing`：20/20。
- `clean > X missing`：20/20。
- clean rho 落入预设保守动态区间 `[0.12, 0.20]`：20/20。
- RGB missing rho 不高于 0.12：20/20。
- X missing rho 不高于 0.12：20/20。

该扩展结果支持把中心化余弦方案作为下一版 V0 的首选设计；旧 legacy gate 在相同 20 序列上的 clean rho 均值仍仅为 0.10002995，RGB/X missing 为 0.10000093/0.10000133，进一步证明旧动态范围不足。

## 4. 运行计划

先跑短训练通路验证，再启动长训练：

```text
1. stage1 short run: done
2. v0b short run: done
3. stage1 formal-config diagnostic run: passed, stopped at 550/1875 to free GPU
4. v0b formal run: completed as ckpt5 run with recoverable checkpoint cadence
```

短训练用于确认：

- `Loss/*` 和 `IoU` 正常输出。
- `GRA/*`、`Gate/*`、`Router/*` 能进入 train log。
- 没有 NaN、shape error 或显存异常。

正式训练使用：

```bash
CUDA_VISIBLE_DEVICES=0 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_stage1 \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_stage1_20260709 \
  --mode single
```

```bash
CUDA_VISIBLE_DEVICES=0 /home/yufan/code/SEATrack/.venv/bin/python tracking/train.py \
  --script seatrack \
  --config rgbt_gratrack_v0b \
  --save_dir /mnt/tipro4t/seatrack_train_runs/gratrack_v0b_20260709 \
  --mode single
```

## 5. 当前结论

阶段 0 已有完整训练、评测和多模态入口证据，可标记为完成。阶段 1 的 diagnostics-only 真实训练通路与 LasHeR 245 序列全量 benchmark 均已跑通；245 个 raw result 文件、220703 行预测框和四项指标与 clean baseline 完全一致，因此“不改变模型行为”和 benchmark 等价性核心门禁通过。阶段 1 仅剩禁用 Gate/Sparse 时缺少 null/default 占位这一项日志规范缺口。阶段 2 V0-b 已完整训练 60 epoch，并完成 LasHeR 245/245 正式评测，且确认没有混入旧 OT / Sinkhorn / ProbAlign 模块；但同协议 clean baseline 在四项 LasHeR 指标上全部更高，baseline 相对 V0-b 分别领先 PR20=0.005950、NPR20=0.008945、NPR_AUC=0.007017、SR/AUC=0.006400。同时，旧 `rho_raw` 虽对 blur/missing 有正确方向变化，但应用后 rho 几乎固定在 0.1。配对效率测量显示 V0-b 延迟增加 6.11%、FPS 下降 5.76%、peak allocated memory 增加 0.023%，通过预设效率门限。故原阶段 2 V0-b 的工程执行已完成、方法验收因精度与机制有效性失败，当前决策为不直接进入阶段 3；下一步是完成推荐的 centered-cosine GRA gate 设计审批并运行 V0-c 修复实验。
