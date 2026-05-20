# SEATrack 复现指南

> **论文**: SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker (CVPR 2026 Oral)
> **代码**: https://github.com/jbs99/SEATrack
> **论文**: https://arxiv.org/abs/2604.12502

---

## 1. 环境配置

### 1.1 创建 Conda 环境

```bash
conda env create -f environment.yaml
conda activate seatrack
```

环境关键依赖：
- Python 3.8
- PyTorch 2.2.2 + CUDA 12.1
- torchvision 0.17.2
- timm 0.5.4
- opencv-python 4.11
- vot-toolkit 0.5.3 (RGB-D 评估需要)
- tensorboard 2.14

> **注意**: `environment.yaml` 中使用了清华镜像源（Linux），Windows 用户需修改或删除 channels 中的 tsinghua 源。

### 1.2 Windows 兼容性说明

该项目原生面向 Linux 开发（`environment.yaml` 中包含 `ld_impl_linux-64`、`_libgcc_mutex` 等 Linux 专有包）。在 Windows 上复现建议：
- 使用 **WSL2** 创建 Linux 环境后按上述步骤配置；或
- 手动安装核心依赖（跳过 Linux 专有包）：
  ```bash
  conda create -n seatrack python=3.8
  conda activate seatrack
  pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
  pip install timm==0.5.4 opencv-python tensorboard vot-toolkit matplotlib pandas
  ```

---

## 2. 数据集准备

### 2.1 下载数据集

| 任务 | 数据集 | 下载链接 |
|------|--------|----------|
| RGB-T | LasHeR | [GitHub](https://github.com/BUGPLEASEOUT/LasHeR) |
| RGB-T | RGBT234 | [百度网盘](https://pan.baidu.com/share/init?surl=weaiBh0_yH2BQni5eTxHgg) 提取码: qvsq |
| RGB-D | DepthTrack | [GitHub](https://github.com/xiaozai/DeT) |
| RGB-D | VOT22-RGBD | [VOT Challenge](https://www.votchallenge.net/vot2022/dataset.html) |
| RGB-E | VisEvent | [GitHub](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) |

### 2.2 数据集目录结构

将所有数据集放置于统一的 `<DATA_PATH>` 目录下（例如 `/data/`），结构如下：

```
<DATA_PATH>/
├── DepthTrack/trainingset/
│   ├── adapter02_indoor/
│   ├── bag03_indoor/
│   ├── bag04_indoor/
│   └── ...
├── LasHeR/
│   ├── trainingset/
│   │   ├── 1boygo/
│   │   ├── 1handsth/
│   │   └── ...
│   └── testingset/
│       ├── 1boygo/
│       └── ...
├── RGBT234/
│   ├── (序列文件夹)/
│   └── ...
├── VisEvent/
│   ├── trainingset/
│   │   ├── 00142_tank_outdoor2/
│   │   └── ...
│   └── testingset/
│       ├── testlist.txt
│       └── ...
└── VOT22-RGBD/ (可选)
    └── sequences/
```

---

## 3. 路径配置

### 方法一：自动生成（推荐）

```bash
cd <SEATrack 项目根目录>
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir <DATA_PATH> \
  --save_dir ./output
```

此脚本会自动生成：
- `lib/train/admin/local.py` — 训练用路径
- `lib/test/evaluation/local.py` — 测试用路径

### 方法二：手动修改

直接编辑下面两个文件中的路径：
- [lib/train/admin/local.py](lib/train/admin/local.py) — 训练数据路径
- [lib/test/evaluation/local.py](lib/test/evaluation/local.py) — 测试/保存路径

需修改的关键路径变量：
- `workspace_dir` — 项目根目录
- `lasher_dir` — LasHeR 训练集路径
- `depthtrack_dir` — DepthTrack 训练集路径
- `visevent_dir` — VisEvent 训练集路径
- `save_dir` — 模型保存目录
- `network_path` — 测试时加载模型的路径

---

## 4. 下载预训练权重

### 4.1 下载 OSTrack 预训练 backbone

从 [OSTrack Google Drive](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing) 下载以下两个预训练权重，放置于：

```
./pretrained/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar
./pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar
```

> 说明：`vitb_256_mae_32x4_ep300` 用于 RGB-D 任务，`vitb_256_mae_ce_32x4_ep300` 用于 RGB-T 和 RGB-E 任务。

### 4.2 下载 SEATrack 已训练模型（可选）

从以下任一源下载已训练好的 SEATrack 模型直接测试：
- [Google Drive](https://drive.google.com/drive/folders/1dDKtK11pX8rmP1pYvgpdLndNjX2hqjYQ?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1QNFkLc0AXvQ8l7LkUYYcCg?pwd=r4s7)
- [Hugging Face](https://huggingface.co/jbs99/SEATrack)

---

## 5. 训练

训练脚本为 [train.sh](train.sh)，支持三种模态分别训练。

### 5.1 RGB-T 训练（LasHeR）

```bash
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script seatrack --config rgbt --save_dir ./models --mode multiple
```

配置详情见 [experiments/seatrack/rgbt.yaml](experiments/seatrack/rgbt.yaml)：
- Backbone: ViT-Base + CE（Candidate Elimination）
- Epoch: 60，Batch Size: 32
- 学习率: 4e-4（第 48 epoch 降至 1/10）
- PEFT (LoRA): 启用，仅训练 0.6M 参数
- 数据集: LasHeR（单数据集训练）

### 5.2 RGB-D 训练（DepthTrack）

```bash
python tracking/train.py --script seatrack --config rgbd --save_dir ./models --mode multiple
```

> 默认使用单 GPU。配置见 [experiments/seatrack/rgbd.yaml](experiments/seatrack/rgbd.yaml)，共 25 epoch。

### 5.3 RGB-E 训练（VisEvent）

```bash
python tracking/train.py --script seatrack --config rgbe --save_dir ./models --mode multiple
```

> 配置见 [experiments/seatrack/rgbe.yaml](experiments/seatrack/rgbe.yaml)，共 45 epoch，学习率 6e-5。

### 5.4 训练参数说明

| 参数 | 含义 |
|------|------|
| `--script` | 训练脚本名，固定为 `seatrack` |
| `--config` | YAML 配置名：`rgbt` / `rgbd` / `rgbe` |
| `--save_dir` | 模型和日志保存根目录 |
| `--mode` | `single`（单卡）/ `multiple`（多卡 DataParallel）/ `multi_node`（多机 DDP） |
| `--nproc_per_node` | 单机 GPU 数（默认使用全部可用 GPU） |

训练日志保存在 `./models/logs/`，模型保存在 `./models/checkpoints/<config>/`。

---

## 6. 测试 / 评估

### 6.1 检查点配置

测试前需确认 [lib/test/parameter/seatrack.py](lib/test/parameter/seatrack.py) 中的 checkpoint 路径正确：

```python
# RGB-T 模型路径
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbt/SEATrack_ep0060.pth.tar")
# RGB-D 模型路径
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbd/SEATrack_ep0025.pth.tar")
# RGB-E 模型路径
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbe/SEATrack_ep0045.pth.tar")
```

### 6.2 RGB-T 测试（LasHeR & RGBT234）

修改 [RGBT_workspace/test_rgbt_mgpus.py](RGBT_workspace/test_rgbt_mgpus.py) 中的数据集路径：

```python
# LasHeR
elif dataset_name == 'LasHeR':
    seq_home = '/data/lasher/testingset'   # 修改为实际路径

# RGBT234
elif dataset_name == 'RGBT234':
    seq_home = '/data/rgbt234'             # 修改为实际路径
```

运行测试：

```bash
# 测试 LasHeR
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name LasHeR --yaml_name rgbt

# 测试 RGBT234
CUDA_VISIBLE_DEVICES=0,2,3 python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name RGBT234 --yaml_name rgbt
```

> 跟踪结果保存至 `./RGBT_workspace/<variants>/<dataset_name>/`。

#### 使用官方工具评估

- **LasHeR**: 使用 [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) 计算 PR/SR
- **RGBT234**: 使用 [RGB-T Toolkit](https://github.com/xuboyue1999/RGBT-Tracking/tree/main) 计算 MPR/MSR

### 6.3 RGB-E 测试（VisEvent）

修改 [RGBE_workspace/test_rgbe_mgpus.py](RGBE_workspace/test_rgbe_mgpus.py) 中的路径：

```python
if dataset_name == 'VisEvent':
    seq_home = '/data/visevent/testingset'  # 修改为实际路径
```

运行测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 python ./RGBE_workspace/test_rgbe_mgpus.py \
  --script_name seatrack --yaml_name rgbe
```

> 使用 [VisEvent Benchmark](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) 进行评估。

### 6.4 RGB-D 测试（DepthTrack & VOT22-RGBD）

RGB-D 评估使用 VOT 官方工具包。

#### 步骤一：准备序列软链接

将数据集序列放入对应 workspace：

```
./Depthtrack_workspace/sequences/   # DepthTrack 测试序列
./VOT22RGBD_workspace/sequences/    # VOT22-RGBD 测试序列
```

对应的 `list.txt` 已提供（[DepthTrack](Depthtrack_workspace/list.txt)、[VOT22](VOT22RGBD_workspace/list.txt)）。

#### 步骤二：修改 tracker 路径

编辑 [Depthtrack_workspace/trackers.ini](Depthtrack_workspace/trackers.ini) 和 [VOT22RGBD_workspace/trackers.ini](VOT22RGBD_workspace/trackers.ini)：

```ini
[rgbd]
label = rgbd
protocol = traxpython
command = seatrack_baseline
paths = <SEATrack项目路径>/lib/test/vot   # 修改为实际项目路径
```

#### 步骤三：运行 VOT 评估

```bash
# DepthTrack 评估
cd Depthtrack_workspace
vot evaluate --workspace ./ rgbd
vot analysis --nocache --name rgbd
cd ..

# VOT22-RGBD 评估
cd VOT22RGBD_workspace
vot evaluate --workspace ./ rgbd
vot analysis --nocache --name rgbd
cd ..
```

或直接运行整合脚本：

```bash
bash eval_rgbd.sh
```

> 注意：VOT 评估依赖 `vot-toolkit`（已在 environment.yaml 中包含）及 TraX 协议，确保 MATLAB 或 Python 版本的 VOT toolkit 正确安装。

---

## 7. 关键文件速查

| 用途 | 文件路径 |
|------|----------|
| 环境配置 | [environment.yaml](environment.yaml) |
| 默认配置 | [lib/config/seatrack/config.py](lib/config/seatrack/config.py) |
| 训练入口 | [tracking/train.py](tracking/train.py) |
| 训练主循环 | [lib/train/run_training.py](lib/train/run_training.py) |
| 训练逻辑 | [lib/train/train_script.py](lib/train/train_script.py) |
| RGB-T 实验配置 | [experiments/seatrack/rgbt.yaml](experiments/seatrack/rgbt.yaml) |
| RGB-D 实验配置 | [experiments/seatrack/rgbd.yaml](experiments/seatrack/rgbd.yaml) |
| RGB-E 实验配置 | [experiments/seatrack/rgbe.yaml](experiments/seatrack/rgbe.yaml) |
| 模型定义 | [lib/models/seatrack/seatrack.py](lib/models/seatrack/seatrack.py) |
| 测试参数 | [lib/test/parameter/seatrack.py](lib/test/parameter/seatrack.py) |
| 路径自动生成 | [tracking/create_default_local_file.py](tracking/create_default_local_file.py) |
| 训练用路径 | [lib/train/admin/local.py](lib/train/admin/local.py) |
| 测试用路径 | [lib/test/evaluation/local.py](lib/test/evaluation/local.py) |
| RGB-T 测试脚本 | [RGBT_workspace/test_rgbt_mgpus.py](RGBT_workspace/test_rgbt_mgpus.py) |
| RGB-E 测试脚本 | [RGBE_workspace/test_rgbe_mgpus.py](RGBE_workspace/test_rgbe_mgpus.py) |
| VOT tracker wrapper | [lib/test/vot/seatrack_baseline.py](lib/test/vot/seatrack_baseline.py) |

---

## 8. 常见问题

**Q: 报错 `YOU HAVE NOT SETUP YOUR local.py!!!`**

A: 需先运行 `python tracking/create_default_local_file.py` 或手动创建 `local.py` 文件。

**Q: 训练时显存不足 (OOM)**

A: RGB-T 默认 batch_size=32，可在对应 YAML 中降低 `TRAIN.BATCH_SIZE`。LoRA 模式下可训练参数仅 0.6M，但仍需存储 ViT backbone 的中间激活。

**Q: VOT 评估时 `TraX support not found`**

A: 需安装 `vot-trax`：`pip install vot-trax`（已在 environment.yaml 中包含）。

**Q: RGB-E 训练学习率不同**

A: VisEvent 数据集特性不同，YAML 中 `LR: 0.00006`（远低于其他两个任务的 0.0004）。

**Q: 项目中无 `models/` 目录**

A: 训练时会自动创建。如从零开始，只需确保 `pretrained/` 下有 OSTrack 预训练权重。
