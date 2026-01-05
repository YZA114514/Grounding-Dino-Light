# GroundingDINO Jittor Implementation

This project is a Jittor implementation of GroundingDINO, as part of the 2025 ANN Final Project (Tsinghua University).

**Paper**: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

**Code**: [https://github.com/YZA114514/Grounding-Dino-Light.git](https://github.com/YZA114514/Grounding-Dino-Light.git)

## 🎯 Highlights

- ✅ **83%+ of original performance**: Achieved **21.4 AP** on LVIS minival (paper: 25.6 AP)
- ✅ **Pure Jittor implementation**: Multi-scale deformable attention, bi-directional cross-modal attention, BERT encoder
- ✅ **Complete training pipeline**: Zero-shot evaluation, fine-tuning, ablation studies
- ✅ **63% inference speedup**: Optimized from 14.8s to 5.5s per image

## 📊 Experimental Results

### Zero-Shot Detection (LVIS minival, 4752 images)

| Metric | Jittor (Ours) | Paper Target | Ratio |
|--------|--------------|--------------|-------|
| **AP** | 21.4% | 25.6% | 83.6% |
| **AP₅₀** | 28.7% | - | - |
| **APᵣ** (rare) | 12.7% | 14.4% | 88.2% |
| **APc** (common) | 18.9% | 19.6% | 96.4% |
| **APf** (frequent) | 25.3% | 32.2% | 78.6% |

### Comparison with OWL-ViT (LVIS minival)

| Method | AP | AP₅₀ | APₛ | APₘ | APₗ | Time/img |
|--------|-----|------|-----|-----|-----|----------|
| **Grounding DINO (Jittor)** | **21.4** | **28.7** | **13.7** | **30.5** | **39.1** | 5.5s |
| OWL-ViT | 17.9 | 28.2 | 9.1 | 24.0 | 35.7 | **2.4s** |
| **Δ** | **+3.5** | +0.5 | **+4.6** | **+6.5** | +3.4 | - |

**Key findings**:
- Grounding DINO outperforms OWL-ViT by **+3.5 AP** overall
- Significant advantage on small/medium objects (**+4.6 / +6.5 AP**) due to multi-scale feature fusion
- OWL-ViT is 2.3× faster due to single-pass inference

### Fine-tuning Results (100-image subset evaluation)

| Method | AP | AP₅₀ | APᵣ | APc | APf |
|--------|-----|------|-----|-----|-----|
| Zero-shot (Jittor) | 36.5 | 47.5 | 16.7 | 22.8 | 38.0 |
| Fine-tuned (640², 5ep, 1k samples) | **41.4** | **50.1** | **23.3** | **29.1** | **42.8** |
| *Improvement* | *+4.9* | *+2.6* | *+6.6* | *+6.3* | *+4.8* |

Fine-tuning with just 1k samples improves AP by **+4.9pp**, with rare categories benefiting the most (**+6.6pp**).

### Ablation Study (LVIS minival)

| Setting | AP | AP₅₀ | APᵣ | APc | APf | ΔAP |
|---------|-----|------|-----|-----|-----|-----|
| Jittor Baseline | 21.4 | 28.7 | 12.7 | 18.9 | 25.3 | - |
| w/o text cross-attn | 8.5 | 12.6 | 6.1 | 7.5 | 9.9 | **-60%** |
| random text | 0.3 | 0.4 | 0.0 | 0.0 | 0.3 | **-99%** |

Removing text cross-attention drops AP by 60%, confirming cross-modal fusion is the core mechanism.

## 🔧 Implementation Highlights

1. **Multi-Scale Deformable Attention**: Pure Jittor implementation using `grid_sample` for bilinear interpolation
2. **Pure Jittor BERT**: Complete BERT-base architecture compatible with HuggingFace weights
3. **Weight Mapping**: Handles `module.` prefix removal, `in_proj` splitting (Q/K/V), nested tensor wrappers
4. **JIT Compilation Fix**: Resolved multi-GPU resource contention via `JT_COMPILE_PARALLEL` limiting
5. **Category Batching**: 60 categories/batch (~215 tokens) to stay within BERT's 256 token limit

## Project Structure

```
GroundingDINO_Jittor/
├── jittor_implementation/        # Core codebase
│   ├── models/                   # Model architecture
│   │   ├── backbone/             # Swin Transformer
│   │   ├── attention/            # MS Deformable Attention
│   │   ├── transformer/          # Encoder & Decoder
│   │   ├── text_encoder/         # Pure Jittor BERT
│   │   ├── fusion/               # Bi-directional Cross-Attention
│   │   ├── query/                # Language-guided Query Selection
│   │   └── groundingdino.py      # Full model assembly
│   ├── data/                     # Data loading & transforms
│   ├── losses/                   # Focal, GIoU, L1, Grounding losses
│   ├── eval/                     # LVIS evaluator
│   └── train/                    # Training pipeline
├── scripts/                      # Utility scripts
│   ├── eval_lvis_zeroshot_full.py  # Official zero-shot evaluation
│   ├── finetune_lvis_full.py       # LVIS fine-tuning
│   ├── eval_owlvit_lvis.py         # OWL-ViT comparison
│   ├── eval_Gdino_ablation.py      # Ablation experiments
│   └── convert_weights_pytorch_to_jittor.py
└── weights/                      # Model checkpoints
```

## Installation

### Prerequisites
- Python 3.9
- CUDA 11.x (for GPU acceleration)
- PyTorch (for weight conversion only)

### Step 1: Clone and Setup Environment

```bash
# Clone repository
git clone https://github.com/YZA114514/Grounding-Dino-Light.git
cd Grounding-Dino-Light

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
cd GroundingDINO_Jittor
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verify installation
python -c "import jittor as jt; print(f'Jittor: {jt.__version__}')"
```

### Step 2: Download and Setup BERT Model

BERT 模型用于文本编码，需要从 HuggingFace 下载 `bert-base-uncased`：

```bash
# 方法1: 使用 transformers 自动下载（推荐）
python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('bert-base-uncased'); \
    AutoModel.from_pretrained('bert-base-uncased')"

# 方法2: 手动下载并放置到指定目录
# 下载地址: https://huggingface.co/bert-base-uncased
# 放置位置: GroundingDINO_Jittor/models/bert-base-uncased/
#   - config.json
#   - vocab.txt
#   - pytorch_model.bin (或 model.safetensors)
```

下载完成后，设置离线模式以加速推理：
```bash
export HF_HUB_OFFLINE=1  # Linux/Mac
# 或 Windows: set HF_HUB_OFFLINE=1
```

### Step 3: Download and Convert Official Weights

从官方仓库下载 PyTorch 预训练权重，并转换为 Jittor 格式：

```bash
# 创建权重目录
mkdir -p weights && cd weights

# 下载 Swin-T 官方权重 (~694MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 返回项目目录
cd ..

# 转换为 Jittor 格式 (需要安装 PyTorch)
python scripts/convert_weights_pytorch_to_jittor.py \
    --pytorch_weight weights/groundingdino_swint_ogc.pth \
    --output weights/groundingdino_swint_ogc_jittor.pkl \
    --verify
```

转换脚本功能：
- 加载 PyTorch 权重 (.pth)
- 处理权重名称映射（移除 `module.` 前缀、拆分 `in_proj` 权重）
- 转换为 Jittor 格式 (.pkl)
- 验证转换正确性

### Step 4: Download LVIS Dataset

LVIS 数据集用于零样本评估和微调：

```bash
# 创建数据目录
mkdir -p ../LVIS/minival

# 下载 LVIS minival 标注文件
# 官方地址: https://www.lvisdataset.org/dataset
# 放置位置: ../LVIS/minival/lvis_v1_minival.json

# 下载 COCO 2017 验证集图像 (~1GB)
# 官方地址: https://cocodataset.org/#download
# 放置位置: ../LVIS/minival/ (符号链接或复制相关图像)
```

数据集目录结构：
```
../LVIS/
├── minival/
│   ├── lvis_v1_minival.json     # LVIS minival 标注
│   └── *.jpg                     # COCO val2017 图像
└── lvis_v1_val.json             # (可选) 完整 LVIS 验证集标注
```

**注意**: minival 是 LVIS 验证集的子集 (4,752 张图像)，排除了与 COCO 2017 训练集重叠的样本，用于公平评估。

## Quick Start

### Inference Demo

```bash
# Demo mode (使用内置测试图像)
python scripts/run_inference.py --demo

# Custom image (自定义图像和文本)
python scripts/run_inference.py \
    --image your_image.jpg \
    --text "cat . dog . person ." \
    --output result.jpg \
    --box_threshold 0.3
```

---

## 🔬 实验运行指南

### 1. Zero-Shot 评估 (LVIS)

使用 `eval_lvis_zeroshot_full.py` 进行零样本检测评估：

```bash
# 快速测试 (100 张图像, ~5 分钟)
python scripts/eval_lvis_zeroshot_full.py \
    --num_images 100 \
    --gpu 0 \
    --output_dir outputs/zeroshot_test

# 完整 LVIS minival 评估 (4752 张图像, ~7 小时)
python scripts/eval_lvis_zeroshot_full.py \
    --full \
    --gpu 0 \
    --output_dir outputs/zeroshot_full

# 使用超优化模式 (减少 GPU-CPU 同步, 提升 15-25%)
python scripts/eval_lvis_zeroshot_full.py \
    --full \
    --ultra_optimized \
    --gpu 0

# 断点续传 (从中断处继续)
python scripts/eval_lvis_zeroshot_full.py \
    --full \
    --resume \
    --output_dir outputs/zeroshot_full
```

**主要参数说明**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | Jittor 权重路径 | `weights/groundingdino_swint_ogc_jittor.pkl` |
| `--num_images` | 评估图像数量 | 100 |
| `--full` | 评估完整 minival | False |
| `--batch_size` | 类别批大小 (BERT token 限制) | 60 |
| `--box_threshold` | 置信度阈值 | 0.1 |
| `--ultra_optimized` | 启用超优化模式 | False |
| `--resume` | 断点续传 | False |
| `--checkpoint_interval` | 保存检查点间隔 | 250 |

### 2. 微调 (Fine-tuning)

使用 `finetune_lvis_v2.py` 进行 LVIS 微调：

```bash
# 快速测试 (验证训练流程)
python scripts/finetune_lvis_v2.py \
    --test_only \
    --num_samples 10 \
    --epochs 2

# 小规模微调 (100 样本, 5 epochs)
python scripts/finetune_lvis_v2.py \
    --num_samples 100 \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation 16 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --output_dir outputs/finetune_100

# 大规模微调 (1000 样本, 推荐配置)
python scripts/finetune_lvis_v2.py \
    --num_samples 1000 \
    --epochs 24 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --freeze_text_encoder \
    --output_dir outputs/finetune_1k
```

**微调参数说明**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 预训练权重路径 | Jittor 权重 |
| `--num_samples` | 训练样本数量 | 100 |
| `--epochs` | 训练轮数 | 24 |
| `--batch_size` | 批大小 | 4 |
| `--gradient_accumulation` | 梯度累积步数 | 4 |
| `--lr` | 学习率 | 1e-4 |
| `--lr_backbone` | 骨干网络学习率 | 1e-5 |
| `--freeze_text_encoder` | 冻结 BERT | True |
| `--freeze_backbone` | 冻结 Swin-T | False |
| `--clip_grad_norm` | 梯度裁剪 | 0.1 |

### 3. 评估微调后模型

```bash
# 使用微调权重进行评估
python scripts/eval_lvis_zeroshot_full.py \
    --finetuned_checkpoint outputs/finetune_1k/checkpoint_best.pkl \
    --base_checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --num_images 100 \
    --output_dir outputs/eval_finetuned
```

### 4. 消融实验 (Ablation)

使用 `eval_Gdino_ablation.py` 验证关键组件的作用：

```bash
# 消融1: 移除文本交叉注意力
python scripts/eval_Gdino_ablation.py \
    --ablation no_text_cross_attn \
    --num_images 100 \
    --output_dir outputs/ablation_no_cross_attn

# 消融2: 随机文本嵌入
python scripts/eval_Gdino_ablation.py \
    --ablation random_text \
    --num_images 100 \
    --output_dir outputs/ablation_random_text

# 完整消融实验 (4752 张图像)
python scripts/eval_Gdino_ablation.py \
    --ablation no_text_cross_attn \
    --full \
    --output_dir outputs/ablation_full
```

### 5. OWL-ViT 对比实验

使用 `eval_owlvit_lvis.py` 与 OWL-ViT 进行对比：

```bash
# 快速对比 (100 张图像)
python scripts/eval_owlvit_lvis.py \
    --num_images 100 \
    --batch_size 25 \
    --output_dir outputs/owlvit_test

# 完整对比 (4752 张图像)
python scripts/eval_owlvit_lvis.py \
    --full \
    --batch_size 25 \
    --resume \
    --output_dir outputs/owlvit_full
```

### 6. 结果可视化

```bash
# 可视化检测结果
python scripts/visualize_lvis_predictions.py \
    --predictions outputs/zeroshot_full/lvis_predictions.json \
    --lvis_ann ../LVIS/minival/lvis_v1_minival.json \
    --image_dir ../LVIS/minival \
    --output_dir outputs/visualizations \
    --score_threshold 0.3 \
    --max_boxes 50
```

## Evaluation Scripts Overview

| Script | Purpose | Categories | Speed |
|--------|---------|------------|-------|
| `eval_lvis_zeroshot_full.py` | Official benchmarking | All 1203 | ~5.5s/img |
| `quick_test_zeroshot.py` | Debugging & visualization | GT only | ~0.3s/img |
| `eval_owlvit_lvis.py` | OWL-ViT comparison | All 1203 | ~2.4s/img |
| `eval_Gdino_ablation.py` | Ablation studies | All 1203 | ~5.5s/img |
| `finetune_lvis_v2.py` | LVIS fine-tuning | All 1203 | - |
| `visualize_lvis_predictions.py` | Result visualization | - | - |

## Output Files

评估和训练脚本会生成以下输出文件：

### Zero-Shot 评估输出
```
outputs/zeroshot_full/
├── predictions.jsonl          # 逐行 JSON 预测结果 (支持断点续传)
├── progress.json              # 断点续传进度
├── lvis_predictions.json      # 完整预测结果 (LVIS 格式)
├── lvis_zeroshot_results.json # 评估指标汇总
└── eval.log                   # 运行日志
```

### 微调输出
```
outputs/finetune_1k/
├── checkpoint_epoch_XX.pkl    # 各 epoch 检查点
├── checkpoint_best.pkl        # 最佳模型 (按验证 AP)
├── training_log.json          # 训练损失曲线
└── config.json                # 训练配置
```

## Performance Optimization

| Optimization | Before | After | Speedup |
|--------------|--------|-------|---------|
| Vision feature caching | 14.8s/img | 5.5s/img | **63%** |
| Vectorized post-processing | - | - | included |
| Category batching (60/batch) | OOM | stable | - |

## Troubleshooting

### 常见问题

**1. BERT 模型加载失败**
```bash
# 确保 transformers 已安装
pip install transformers

# 首次运行需要联网下载，之后可以设置离线模式
export HF_HUB_OFFLINE=1
```

**2. CUDA 内存不足 (OOM)**
```bash
# 减小 batch_size
python scripts/eval_lvis_zeroshot_full.py --batch_size 30

# 或使用 CPU 模式
CUDA_VISIBLE_DEVICES="" python scripts/eval_lvis_zeroshot_full.py
```

**3. JIT 编译冲突 (多 GPU)**
```bash
# 限制 JIT 并行编译数
export JT_COMPILE_PARALLEL=1
```

**4. 权重转换失败**
```bash
# 确保同时安装了 PyTorch 和 Jittor
pip install torch
pip install jittor
```

**5. 图像路径不匹配**
```bash
# 检查图像目录结构
ls ../LVIS/minival/*.jpg | head -5

# 如果图像在子目录中，使用软链接
ln -s /path/to/coco/val2017/*.jpg ../LVIS/minival/
```

## Citation

```bibtex
@inproceedings{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## Team

- 张毅 (2022010387, 工22) - grounding-dino的Jittor复现及其他脚本撰写
- 杨弘毅 (2023011638, 英31) - zero-shot及模型对比和消融实验等额外任务
- 苏博宇 (2023011277, 物理32) - 微调及训练pipeline

## License

This project is for educational purposes as part of the Tsinghua University ANN course final project.
