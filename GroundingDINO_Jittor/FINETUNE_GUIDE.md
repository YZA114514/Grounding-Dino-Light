# Grounding DINO Jittor - 微调与评估指南

本文档说明如何在 Linux GPU 服务器上进行 LVIS 数据集的 Zero-Shot 评估和微调实验。

---

## 1. 环境配置

### 1.1 克隆代码

```bash
git clone https://github.com/YZA114514/Grounding-Dino-Light.git
cd Grounding-Dino-Light/GroundingDINO_Jittor
```

### 1.2 创建 Conda 环境

```bash
conda create -n groundingdino_jittor python=3.9 -y
conda activate groundingdino_jittor

# 安装依赖
pip install jittor                 # Jittor (GPU 版本自动检测 CUDA)
pip install torch torchvision      # 用于 BERT 和权重转换
pip install transformers           # BERT tokenizer
pip install timm                   # Swin Transformer 组件
pip install pycocotools            # COCO/LVIS 评估
pip install pillow numpy matplotlib tqdm
```

### 1.3 验证 GPU 支持

```python
import jittor as jt
print(f"Jittor version: {jt.__version__}")
print(f"CUDA available: {jt.has_cuda}")
jt.flags.use_cuda = 1
print(f"GPU enabled: {jt.flags.use_cuda}")
```

---

## 2. 数据准备

### 2.1 下载 LVIS 标注

```bash
mkdir -p data/lvis && cd data/lvis

# 下载 LVIS v1 标注
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
unzip lvis_v1_train.json.zip
unzip lvis_v1_val.json.zip

cd ../..
```

### 2.2 下载 COCO 图像

LVIS 使用 COCO 2017 图像：

```bash
mkdir -p data/coco && cd data/coco

# 验证集图像 (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 训练集图像 (~18GB，微调时需要)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

cd ../..
```

### 2.3 LVIS 数据格式

LVIS 使用 COCO-style JSON 格式：

```json
{
    "images": [{"id": 123456, "file_name": "000000123456.jpg", "height": 480, "width": 640}],
    "annotations": [{"id": 1, "image_id": 123456, "category_id": 42, "bbox": [x, y, w, h], "area": 1234.5}],
    "categories": [{"id": 42, "name": "cat", "synset": "cat.n.01", "frequency": "f"}]
}
```

**数据集统计：** 1203 类别 | ~100K 训练图像 | ~20K 验证图像 | ~1.3M 标注

---

## 3. 准备预训练权重

```bash
mkdir -p weights && cd weights

# 下载 Swin-T 版本 (~694MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ..

# 转换为 Jittor 格式
python scripts/convert_weights_pytorch_to_jittor.py \
    --pytorch_weight weights/groundingdino_swint_ogc.pth \
    --output weights/groundingdino_swint_ogc_jittor.pkl
```

---

## 4. Zero-Shot 评估

直接使用预训练权重在 LVIS 上评估，不进行任何微调：

```bash
python scripts/eval_lvis_zeroshot.py \
    --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --lvis_ann data/lvis/lvis_v1_val.json \
    --image_dir data/coco \
    --output_dir outputs/lvis_zeroshot \
    --box_threshold 0.25 \
    --nms_threshold 0.5 \
    --use_gpu
```

**输出：** `predictions.json` (COCO 格式预测结果) 和 `results.json` (评估指标)

---

## 5. LVIS 微调

### 5.1 基本微调命令

```bash
python scripts/finetune_lvis.py \
    --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --lvis_train data/lvis/lvis_v1_train.json \
    --lvis_val data/lvis/lvis_v1_val.json \
    --image_dir data/coco \
    --output_dir outputs/lvis_finetune \
    --epochs 12 \
    --batch_size 4 \
    --lr 1e-4 \
    --freeze_backbone \
    --freeze_text_encoder \
    --use_gpu
```

### 5.2 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 预训练权重路径 |
| `--lvis_train` | - | LVIS 训练集标注 |
| `--lvis_val` | - | LVIS 验证集标注 |
| `--image_dir` | - | COCO 图像目录 |
| `--output_dir` | outputs/lvis_finetune | 输出目录 |
| `--epochs` | 12 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--lr_backbone` | 1e-5 | Backbone 学习率 |
| `--weight_decay` | 0.0001 | 权重衰减 |
| `--freeze_backbone` | True | 冻结 Backbone |
| `--freeze_text_encoder` | True | 冻结文本编码器 |
| `--num_workers` | 4 | 数据加载线程数 |
| `--log_interval` | 50 | 日志打印间隔 |
| `--save_interval` | 1 | 模型保存间隔 |
| `--resume` | None | 从检查点恢复训练 |

### 5.3 微调策略建议

**冻结策略：**

| 阶段 | Backbone | Text Encoder | Transformer | Head |
|------|----------|--------------|-------------|------|
| 阶段1 (前5轮) | ❄️ 冻结 | ❄️ 冻结 | 🔥 训练 | 🔥 训练 |
| 阶段2 (后续) | 🔥 低lr (1e-5) | ❄️ 冻结 | 🔥 训练 | 🔥 训练 |

**学习率建议：**
- Backbone: `1e-5` (很小，避免破坏预训练特征)
- Text Encoder: 冻结或 `1e-6`
- Transformer/Head: `1e-4`

---

## 6. 微调后评估

```bash
python scripts/eval_lvis_zeroshot.py \
    --checkpoint outputs/lvis_finetune/best_model.pkl \
    --lvis_ann data/lvis/lvis_v1_val.json \
    --image_dir data/coco \
    --output_dir outputs/lvis_finetuned_eval \
    --use_gpu
```

---

## 7. 一键运行所有实验

修改 `scripts/run_lvis_experiments.sh` 中的路径配置后运行：

```bash
chmod +x scripts/run_lvis_experiments.sh
./scripts/run_lvis_experiments.sh
```

该脚本依次执行：准备权重 → Zero-Shot 评估 → 微调训练 → 微调后评估 → 输出对比

---

## 8. 评估指标

| 指标 | 说明 |
|------|------|
| AP | Average Precision @ IoU=0.50:0.95 |
| AP50 | AP @ IoU=0.50 |
| AP75 | AP @ IoU=0.75 |
| APs / APm / APl | AP for small / medium / large objects |
| APr / APc / APf | AP for rare / common / frequent categories (LVIS 特有) |

**预期性能参考：**

| 模型 | Zero-Shot AP | 微调后 AP |
|------|-------------|----------|
| GroundingDINO Swin-T | ~25 | ~28-30 |
| GroundingDINO Swin-B | ~30 | ~33-35 |

---

## 9. 常见问题

### Q1: GPU 显存不足
```bash
python scripts/finetune_lvis.py --batch_size 1 ...
```

### Q2: 训练太慢
```bash
python scripts/finetune_lvis.py --num_workers 16 ...
```

### Q3: 评估指标为 0
- 检查预测文件是否为空
- 检查类别 ID 映射是否正确
- 降低 `--box_threshold` 尝试

### Q4: 找不到图像文件
确保目录结构：
```
data/coco/
├── train2017/
│   └── 000000000001.jpg ...
└── val2017/
    └── 000000000139.jpg ...
```

---

## 10. 目录结构

```
GroundingDINO_Jittor/
├── data/
│   ├── lvis/
│   │   ├── lvis_v1_train.json
│   │   └── lvis_v1_val.json
│   └── coco/
│       ├── train2017/
│       └── val2017/
├── weights/
│   ├── groundingdino_swint_ogc.pth
│   └── groundingdino_swint_ogc_jittor.pkl
├── outputs/
│   ├── lvis_zeroshot/
│   │   ├── predictions.json
│   │   └── results.json
│   ├── lvis_finetune/
│   │   ├── best_model.pkl
│   │   ├── checkpoint_epoch12.pkl
│   │   └── train_log.txt
│   └── lvis_finetuned_eval/
│       └── results.json
└── scripts/
    ├── eval_lvis_zeroshot.py
    ├── finetune_lvis.py
    ├── convert_weights_pytorch_to_jittor.py
    └── run_lvis_experiments.sh
```

