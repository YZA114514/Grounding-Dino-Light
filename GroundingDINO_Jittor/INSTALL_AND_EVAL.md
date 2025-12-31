# GroundingDINO Jittor 安装与评估说明

## 1. 环境安装

### 1.1 激活.venv环境

```bash
# 激活环境
source ./shared-nvme/GroundingDINO-Light/.venv/bin/activate
```

### 1.2 核心依赖说明

| 依赖包 | 版本要求 | 用途 |
|--------|----------|------|
| **jittor** | >=1.3.0 | Jittor 深度学习框架（核心） |
| **torch** | >=1.13.0 | 用于权重转换 |
| **torchvision** | >=0.14.0 | 用于权重转换 |
| **transformers** | >=4.20.0 | BERT Tokenizer |
| **timm** | >=0.6.0 | Swin Transformer backbone |
| **pycocotools** | >=2.0.4 | COCO/LVIS 评估工具 |
| **pillow** | >=8.0.0 | 图像处理 |
| **numpy** | >=1.21.0 | 数值计算 |

### 1.3 cuDNN 安装

如果首次运行时提示需要下载 cuDNN，按提示完成安装即可。Jittor 会自动下载并配置。

---

## 2. 权重转换

### 2.1 下载预训练权重

从官方仓库下载 PyTorch 权重：
- `groundingdino_swint_ogc.pth`

放置到 `weights/` 目录。

### 2.2 转换为 Jittor 格式

```bash
cd GroundingDINO_Jittor

python scripts/convert_weights_pytorch_to_jittor.py \
    --pytorch_weight weights/groundingdino_swint_ogc.pth \
    --output weights/groundingdino_swint_ogc_jittor.pkl
```

转换成功后会显示权重数量（约 940 个）。

---

## 3. 数据准备

### 3.1 LVIS 数据集

1. **下载 LVIS 标注文件**：
   - `lvis_v1_val.json` → 放置到 `../data/lvis_v1_val.json`

2. **下载 LVIS val 图像**：
   - 图像文件 → 放置到 `../LVIS/val/`

### 3.2 目录结构

```
GroundingDINO_Light/  # Root directory
├── data/
│   └── lvis_v1_val.json
├── LVIS/
│   └── val/
│       ├── 000000000139.jpg
│       ├── 000000000285.jpg
│       └── ...
└── GroundingDINO_Jittor/
    ├── weights/
    │   ├── groundingdino_swint_ogc.pth
    │   └── groundingdino_swint_ogc_jittor.pkl
    └── models/
        └── bert-base-uncased/   # BERT 模型文件
```

---

## 4. Zero-Shot 评估

### 4.1 运行评估

```bash
cd GroundingDINO_Jittor

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 运行评估
python scripts/eval_lvis_zeroshot_full.py \
    --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --lvis_ann ../data/lvis_v1_val.json \
    --image_dir ../LVIS/val \
    --output_dir outputs/lvis_zeroshot \
    --gpu 0
```

### 4.2 后台运行

```bash
nohup python scripts/eval_lvis_zeroshot_full.py \
    --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --lvis_ann ../data/lvis_v1_val.json \
    --image_dir ../LVIS/val \
    --output_dir outputs/lvis_zeroshot \
    --gpu 0 > lvis_zeroshot_run.log 2>&1 &
```

### 4.3 查看进度

```bash
# 查看日志
tail -f lvis_zeroshot_run.log

# 查看进度条
grep "Evaluating" lvis_zeroshot_run.log | tail -5
```

### 4.4 评估结果

评估完成后，结果保存在：
- **控制台/日志文件**：显示 AP (Average Precision) 等指标
- **输出目录**：`outputs/lvis_zeroshot/`
  - `predictions.json` - 预测结果
  - `evaluation_results.json` - 评估指标

主要评估指标：
- **AP** (Average Precision)
- **AP50** (IoU=0.5 时的 AP)
- **AP75** (IoU=0.75 时的 AP)
- **APr** (Rare 类别 AP)
- **APc** (Common 类别 AP)
- **APf** (Frequent 类别 AP)

---

## 5. 常见问题

### 5.1 CUDA 版本不匹配

如果出现 CUDA 相关错误，确保 PyTorch 与系统 CUDA 版本兼容：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.2 内存不足

降低 batch size 或使用更大显存的 GPU。

### 5.3 网络问题

BERT 模型已包含在 `models/bert-base-uncased/` 目录，无需网络连接。

---

## 6. 项目实现说明

本项目使用纯 Jittor 框架实现了 GroundingDINO 模型的所有核心组件：

- ✅ Swin Transformer Backbone
- ✅ Transformer Encoder/Decoder
- ✅ Multi-Scale Deformable Attention
- ✅ BiMultiHeadAttention (跨模态融合)
- ✅ BERT Text Encoder (纯 Jittor 实现)
- ✅ ContrastiveEmbed (分类头)
- ✅ Zero-Shot 评估流程

权重加载支持从 PyTorch 预训练权重完整迁移。
