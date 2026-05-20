# SEATrack 服务器部署与复现指南

> 适用于 Linux 服务器（Ubuntu 20.04/22.04），8×RTX 4090，无 GUI 环境。
> 本指南从零开始，包含代码拉取、数据集下载、测试、训练全流程。

---

## 0. 前置准备

```bash
# 确认 GPU 可用
nvidia-smi

# 确认磁盘空间（VisEvent 需 216GB，总计建议预留 500GB+）
df -h
```

---

## 1. 环境配置

### 1.1 安装 Miniconda（如未安装）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 1.2 克隆代码

```bash
git clone git@github.com:<你的用户名>/SEATrack.git
cd SEATrack
```

### 1.3 创建环境

```bash
# 修改 environment.yaml 中的镜像源（清华源可能在服务器上更快）
# 如果报 channel 错误，删除 channels 中的 tsinghua 行，只保留 pytorch/nvidia/defaults

conda env create -f environment.yaml
conda activate seatrack
```

> **注意**：如果 conda 创建环境很慢，可以按以下 Mamba 加速方式：
> ```bash
> conda install mamba -c conda-forge
> mamba env create -f environment.yaml
> ```

### 1.4 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# 应输出: True 8
```

---

## 2. 数据集下载

### 2.1 下载方式总览

| 数据集 | 大小 | 百度网盘 | Google Drive | 其他 |
|--------|------|----------|-------------|------|
| **LasHeR** | ~50GB | ✅ 提取码 `mmic` | ❌ | TeraBox 提取码 `yfi0` |
| **RGBT234** | ~20GB | ✅ 提取码 `qvsq` | ❌ | GitCode 镜像 |
| **DepthTrack** | ~30GB | ❌ | ✅ [GitHub](https://github.com/xiaozai/DeT) | - |
| **VisEvent** | **216GB** | ✅ 提取码 `AHUE` | ✅ | **Dropbox** ✅ 推荐 |
| **VOT22-RGBD** | ~5GB | ❌ | ❌ | [VOT 官网](https://www.votchallenge.net/vot2022/dataset.html) |

### 2.2 服务器下载方案

#### 方案 A：百度网盘 → 服务器（推荐国内服务器）

服务器上下载百度网盘文件有三种方式：

**A1. bypy（百度网盘 Python 客户端，推荐）**
```bash
pip install bypy
bypy info   # 首次运行会提示授权，在本地浏览器打开授权链接，粘贴授权码
bypy list    # 列出的文件会显示在 "我的应用数据/bypy" 目录下

# 先把数据集文件保存到百度网盘的 "我的应用数据/bypy/" 目录下
# 然后服务器上下载
bypy downfile <远程文件路径> <本地保存路径>
```

**A2. BaiduPCS-Go（更稳定，但需手动安装）**
```bash
# 下载 BaiduPCS-Go 二进制文件
wget https://github.com/qjfoidnh/BaiduPCS-Go/releases/latest
# 详细使用见: https://github.com/qjfoidnh/BaiduPCS-Go
```

**A3. 本地下载 → scp 上传（最稳妥）**
```bash
# 在你的 Windows 上把数据集下载到本地
# 然后用 scp 传到服务器
scp -r /path/to/dataset user@server_ip:/data/
```

#### 方案 B：Google Drive → 服务器

```bash
# 安装 gdown
pip install gdown

# 下载文件
gdown "https://drive.google.com/uc?id=<文件ID>"

# 下载整个文件夹
gdown "https://drive.google.com/drive/folders/<文件夹ID>" -O ./output_dir --folder
```

#### 方案 C：Dropbox → 服务器（VisEvent 推荐！）

**VisEvent Dropbox 直链（无需会员，速度稳定）：**
```
https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn&st=rzf95buf&dl=0
```

##### 方法 C1：tmux + wget 断点续传（推荐，216GB 不怕断）

```bash
# 1. 创建 tmux 会话专门用于下载
tmux new-session -d -s download_vis
tmux send-keys -t download_vis \
  "mkdir -p /data/VisEvent && cd /data/VisEvent && \
   wget -c -O VisEvent.zip \
   'https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn&st=rzf95buf&dl=1'" Enter

# 2. 查看下载进度
tmux attach -t download_vis
# 按 Ctrl+B 再按 D 分离（下载继续在后台跑）

# 3. 如果断网/断开 SSH，重新登录后：
tmux attach -t download_vis
# 检查进度，wget -c 会自动从断点续传

# 4. 下载完成后的操作：
# tmux attach -t download_vis   → 确认下载完成
# Ctrl+C 退出 tmux 会话，然后：
sudo apt install -y p7zip-full
cd /data/VisEvent
7z x VisEvent.zip
# 解压后删除 zip 释放空间
rm VisEvent.zip

# 5. 清理 tmux 会话
tmux kill-session -t download_vis
```

> **关键参数说明：**
> - `-c` (continue)：断点续传，断网重连后继续下载，不会重新开始
> - `-O VisEvent.zip` (大写 O)：指定输出文件名
> - `dl=1`：将 Dropbox 的预览页改为直接下载链接
> - **注意**：如果下载中断且 wget 报错无法续传（某些 CDN 不支持 Range 请求），删除未完成的 `.zip` 文件重新运行即可

##### 方法 C2：本地浏览器下载 → scp（最稳但慢）

```bash
# Windows 上浏览器打开 Dropbox 链接下载
# 然后 scp 上传到服务器
scp -r /path/to/VisEvent.zip user@server_ip:/data/VisEvent/
```

### 2.3 各数据集具体下载步骤

#### LasHeR（RGB-T 训练+测试）

```bash
mkdir -p /data/LasHeR

# 百度网盘链接: https://pan.baidu.com/s/??? (提取码 mmic)
# TeraBox 链接 (提取码 yfi0): 适合海外服务器

# 下载后解压到指定结构
# LasHeR 结构:
#   trainingset/  (训练序列)
#   testingset/   (测试序列)
```

#### RGBT234（RGB-T 额外测试）

```bash
mkdir -p /data/RGBT234

# 百度网盘: https://pan.baidu.com/share/init?surl=weaiBh0_yH2BQni5eTxHgg (提取码 qvsq)

# 下载后直接解压即可，每个子文件夹是一个序列
```

#### DepthTrack（RGB-D 训练+测试）

```bash
mkdir -p /data/DepthTrack

# GitHub: https://github.com/xiaozai/DeT
# README 中找到 Google Drive 链接，分为 training set (100 seqs + 52 seqs) 和 test set
# 用 gdown 下载

# 结构:
#   trainingset/
#   testset/
```

#### VisEvent（RGB-E 训练+测试，216GB）

```bash
mkdir -p /data/VisEvent

# Dropbox (推荐，速度最快):
# https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn&st=rzf95buf&dl=0
# 把 dl=0 改为 dl=1 即可直接下载

# 百度网盘 (备选):
# https://pan.baidu.com/s/1VhdORXT4OvG8TUESfDZHfw (提取码 AHUE)

# 下载后解压
sudo apt install p7zip-full
7z x VisEvent.zip

# 结构:
#   trainingset/
#   testingset/
#       testlist.txt
```

#### VOT22-RGBD（RGB-D 额外测试，可选）

```bash
# 官网注册下载: https://www.votchallenge.net/vot2022/dataset.html
# 需要注册 VOT 账号，下载 VOT2022 RGBD 子集
```

### 2.4 最终数据集目录结构

```
/data/
├── LasHeR/
│   ├── trainingset/
│   │   ├── 1boygo/
│   │   └── ...
│   └── testingset/
├── RGBT234/
│   ├── afterrain/
│   └── ...
├── DepthTrack/
│   ├── trainingset/
│   │   ├── adapter02_indoor/
│   │   └── ...
│   └── testset/
├── VisEvent/
│   ├── trainingset/
│   │   ├── 00142_tank_outdoor2/
│   │   └── ...
│   └── testingset/
│       ├── testlist.txt
│       └── ...
└── (VOT22-RGBD/)
    └── sequences/
```

---

## 3. 下载预训练模型

### 3.1 OSTrack Backbone（训练必需）

```bash
mkdir -p pretrained/vitb_256_mae_32x4_ep300
mkdir -p pretrained/vitb_256_mae_ce_32x4_ep300

# Google Drive: https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy
# 用 gdown 下载 OSTrack_ep0300.pth.tar 到对应目录

# vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar  → 用于 RGB-D
# vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar → 用于 RGB-T 和 RGB-E
```

### 3.2 SEATrack 已训练模型（测试用）

```bash
# 方法1: Hugging Face (推荐，海外服务器速度快)
# https://huggingface.co/jbs99/SEATrack
pip install huggingface_hub
huggingface-cli download jbs99/SEATrack --local-dir ./models/checkpoints

# 方法2: Google Drive
# https://drive.google.com/drive/folders/1dDKtK11pX8rmP1pYvgpdLndNjX2hqjYQ

# 方法3: 百度网盘 (国内服务器推荐)
# https://pan.baidu.com/s/1QNFkLc0AXvQ8l7LkUYYcCg (提取码 r4s7)

# 放置结构:
#   models/checkpoints/rgbt/SEATrack_ep0060.pth.tar
#   models/checkpoints/rgbd/SEATrack_ep0025.pth.tar
#   models/checkpoints/rgbe/SEATrack_ep0045.pth.tar
```

---

## 4. 路径配置

```bash
cd ~/SEATrack

# 自动生成 local.py 文件
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir /data \
  --save_dir ./output
```

这会自动生成两个文件：
- `lib/train/admin/local.py` — 训练数据路径
- `lib/test/evaluation/local.py` — 测试路径

**验证路径是否正确：**
```bash
cat lib/train/admin/local.py | grep _dir
# 确认所有路径指向 /data/ 下的正确位置
```

---

## 5. 直接测试（验证环境，无需训练）

在训练之前，先跑一遍测试验证整个 pipeline 能通。

### 5.1 确认 checkpoint 路径

编辑 [lib/test/parameter/seatrack.py](lib/test/parameter/seatrack.py)，确认 checkpoint 路径指向你下载的模型：

```python
# 第 27 行附近
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbt/SEATrack_ep0060.pth.tar")
# 第 32 行
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbd/SEATrack_ep0025.pth.tar")
# 第 37 行
params.checkpoint = os.path.join(save_dir, "checkpoints/rgbe/SEATrack_ep0045.pth.tar")
```

> 注意：`save_dir` 来自 `local.py` 中的 `env_settings().save_dir`

### 5.2 测试 RGB-T (LasHeR)

```bash
# 先修改 RGBT_workspace/test_rgbt_mgpus.py 中的路径：
# 第 173-175 行附近
# seq_home = '/data/LasHeR/testingset'

# 运行测试（先只测一个序列验证流程）
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack \
  --dataset_name LasHeR \
  --yaml_name rgbt \
  --mode sequential \
  --video 1boygo \
  --threads 1

# 如果成功，再跑全量测试
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack \
  --dataset_name LasHeR \
  --yaml_name rgbt \
  --threads 30
```

### 5.3 测试 RGB-D (DepthTrack)

```bash
# 1. 确认数据集软链接
mkdir -p Depthtrack_workspace/sequences
ln -s /data/DepthTrack/testset/* Depthtrack_workspace/sequences/

# 2. 修改 trackers.ini 中的 paths
# Depthtrack_workspace/trackers.ini:
#   paths = /home/<user>/SEATrack/lib/test/vot

# 3. 运行 VOT 评估
cd Depthtrack_workspace
vot evaluate --workspace ./ rgbd
vot analysis --nocache --name rgbd
cd ..
```

### 5.4 测试 RGB-E (VisEvent)

```bash
# 修改 RGBE_workspace/test_rgbe_mgpus.py 中的路径：
# 第 133 行附近
# seq_home = '/data/VisEvent/testingset'

# 单序列测试
CUDA_VISIBLE_DEVICES=0 python ./RGBE_workspace/test_rgbe_mgpus.py \
  --script_name seatrack \
  --yaml_name rgbe \
  --mode sequential \
  --video 00142_tank_outdoor2 \
  --threads 1

# 全量测试
CUDA_VISIBLE_DEVICES=0,1 python ./RGBE_workspace/test_rgbe_mgpus.py \
  --script_name seatrack \
  --yaml_name rgbe \
  --threads 30
```

---

## 6. 训练

### 6.1 训练策略

8 卡 4090 可以同时跑三个任务（互不依赖）：

| 任务 | GPU | Epoch | 大概时间 | Batch Size | 学习率 |
|------|-----|-------|---------|------------|--------|
| RGB-T (LasHeR) | 0,1 | 60 | ~8-10h | 32 | 4e-4 |
| RGB-D (DepthTrack) | 2,3 | 25 | ~3-4h | 32 | 4e-4 |
| RGB-E (VisEvent) | 4,5 | 45 | ~6-8h | 32 | 6e-5 |

> LoRA 方案仅 0.6M 可训练参数，训练非常快。

### 6.2 一键启动三个训练

```bash
# 分别在后台 tmux 会话中启动三个训练，断开 SSH 也不会中断

# 1. 创建会话
tmux new-session -d -s train_rgbt
tmux new-session -d -s train_rgbd
tmux new-session -d -s train_rgbe

# 2. 在各会话中执行训练命令

# RGB-T (GPU 0,1) — 60 epoch
tmux send-keys -t train_rgbt \
  "CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py --script seatrack --config rgbt --save_dir ./models --mode multiple" Enter

# RGB-D (GPU 2,3) — 25 epoch
tmux send-keys -t train_rgbd \
  "CUDA_VISIBLE_DEVICES=2,3 python tracking/train.py --script seatrack --config rgbd --save_dir ./models --mode multiple" Enter

# RGB-E (GPU 4,5) — 45 epoch
tmux send-keys -t train_rgbe \
  "CUDA_VISIBLE_DEVICES=4,5 python tracking/train.py --script seatrack --config rgbe --save_dir ./models --mode multiple" Enter

# 3. 查看某个任务的实时输出
tmux attach -t train_rgbt    # 按 Ctrl+B 再按 D 分离并回到 shell

# 4. 查看所有 tmux 会话
tmux ls

# 5. 训练完成后清理
tmux kill-session -t train_rgbt
tmux kill-session -t train_rgbd
tmux kill-session -t train_rgbe
```

### 6.3 单独训练某个任务

```bash
# RGB-T（2卡）
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
  --script seatrack --config rgbt --save_dir ./models --mode multiple

# RGB-D（单卡也够）
python tracking/train.py \
  --script seatrack --config rgbd --save_dir ./models --mode multiple
```

### 6.4 训练监控

```bash
# 查看 TensorBoard
tensorboard --logdir ./models/tensorboard --port 6006
# 本地浏览器访问: http://<服务器IP>:6006

# 查看训练日志
tail -f ./models/logs/seatrack-rgbt.log
tail -f ./models/logs/seatrack-rgbd.log
tail -f ./models/logs/seatrack-rgbe.log

# 查看 GPU 使用率
watch -n 1 nvidia-smi
```

### 6.5 训练产物

训练完成后 `models/checkpoints/` 下的结构：

```
models/
├── checkpoints/
│   ├── rgbt/
│   │   └── SEATrack_ep0060.pth.tar
│   ├── rgbd/
│   │   └── SEATrack_ep0025.pth.tar
│   └── rgbe/
│       └── SEATrack_ep0045.pth.tar
├── logs/
│   ├── seatrack-rgbt.log
│   ├── seatrack-rgbd.log
│   └── seatrack-rgbe.log
└── tensorboard/
```

---

## 7. 快速参考：下载链接汇总

### 数据集

| 数据集 | 百度网盘 | Google Drive | OneDrive |
|--------|----------|-------------|----------|
| LasHeR | [链接](https://pan.baidu.com) 提取码 `mmic` | - | TeraBox: `yfi0` |
| RGBT234 | [链接](https://pan.baidu.com/share/init?surl=weaiBh0_yH2BQni5eTxHgg) 提取码 `qvsq` | - | - |
| DepthTrack | - | [GitHub DeT](https://github.com/xiaozai/DeT) | - |
| VisEvent | [百度](https://pan.baidu.com/s/1VhdORXT4OvG8TUESfDZHfw) 提取码 `AHUE` | ✅ | **[Dropbox](https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn&st=rzf95buf&dl=0)** 推荐 |
| VOT22-RGBD | - | - | [VOT 官网](https://www.votchallenge.net/vot2022/dataset.html) |

### 预训练模型

| 模型 | Google Drive | HuggingFace | 百度网盘 |
|------|-------------|-------------|----------|
| OSTrack Backbone | [链接](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy) | - | - |
| SEATrack 已训练 | [链接](https://drive.google.com/drive/folders/1dDKtK11pX8rmP1pYvgpdLndNjX2hqjYQ) | [hf.co/jbs99/SEATrack](https://huggingface.co/jbs99/SEATrack) | 提取码 `r4s7` |

---

## 8. 关键文件对照表（服务器上修改用）

| 改什么 | 文件位置 |
|--------|----------|
| 训练数据路径 | `lib/train/admin/local.py` |
| 测试数据/模型路径 | `lib/test/evaluation/local.py` |
| 测试 checkpoint | `lib/test/parameter/seatrack.py` |
| 训练超参 | `experiments/seatrack/rgbt.yaml` / `rgbd.yaml` / `rgbe.yaml` |
| RGB-T 测试数据路径 | `RGBT_workspace/test_rgbt_mgpus.py` (seq_home) |
| RGB-E 测试数据路径 | `RGBE_workspace/test_rgbe_mgpus.py` (seq_home) |
| VOT tracker 路径 | `Depthtrack_workspace/trackers.ini` / `VOT22RGBD_workspace/trackers.ini` |

---

## 9. 常见问题

### Q: `ImportError: lib.train.admin.local`

没有运行路径配置脚本。执行：
```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir /data --save_dir ./output
```

### Q: 服务器上百度网盘怎么下？

三种方式按推荐度排序：
1. **本地 Windows 下载 → scp 上传**（最稳但需要中转）
2. **bypy**（`pip install bypy`，百度官方 API，需授权）
3. **BaiduPCS-Go**（第三方，速度快但可能被封）

VisEvent 优先用 **OneDrive**（见第 2 节方案 C），DepthTrack 优先用 **gdown + Google Drive**。

### Q: 显存不足 (OOM)

在对应 YAML 中降低 batch size：
```yaml
TRAIN:
  BATCH_SIZE: 16   # 从 32 降到 16
```

### Q: VOT 评估报错 `TraX support not found`

```bash
pip install vot-trax
```

### Q: VisEvent 太大，磁盘不够

VisEvent 训练集约 180GB，测试集约 36GB。至少需要测试集才能跑评估。
可以先只下载测试集测试，训练等磁盘扩容后再做。

### Q: 训练到一半断了怎么恢复？

训练脚本默认 `load_latest=True`，重新运行相同命令会自动加载最新 checkpoint 续训。
