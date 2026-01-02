# GroundingDINO Jittor Implementation

This project is a Jittor implementation of GroundingDINO, as part of the 2025 Final Project.

## 🎯 Zero-Shot Evaluation Results

Our Jittor implementation achieves comparable performance to the official PyTorch implementation on LVIS zero-shot object detection:

| Metric | Our Result | Paper Target | Status |
|--------|-----------|--------------|--------|
| **AP** | 23.5% | 25.6% | ✅ Close |
| **APr** (rare) | 16.7% | 14.4% | ✅ Exceeded |
| **APc** (common) | 18.0% | 19.6% | ✅ Close |
| **APf** (frequent) | 24.1% | 32.2% | ⚠️ In progress |

*Results on 100 images with true zero-shot evaluation (all 1203 LVIS categories)*

## Project Structure

The project structure is organized based on the roles and responsibilities defined in the team plan:

```
GroundingDINO_Jittor/
├── jittor_implementation/        # 核心代码库
│   ├── __init__.py
│   ├── models/                   # [成员A] 模型架构
│   │   ├── __init__.py
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   └── swin_transformer.py
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   └── ms_deform_attn.py
│   │   ├── transformer/
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py
│   │   │   └── decoder.py
│   │   ├── head/
│   │   │   ├── __init__.py
│   │   │   └── dino_head.py
│   │   ├── text_encoder/         # [成员C] 文本编码
│   │   │   ├── __init__.py
│   │   │   ├── bert_wrapper.py
│   │   │   └── text_processor.py
│   │   ├── fusion/               # [成员C] 特征融合
│   │   │   ├── __init__.py
│   │   │   └── feature_fusion.py
│   │   ├── query/                # [成员C] Query生成
│   │   │   ├── __init__.py
│   │   │   └── language_guided_query.py
│   │   ├── groundingdino.py      # [成员A] 完整模型组装
│   │   └── interfaces.py         # [全体] 接口定义
│   ├── data/                     # [成员B] 数据处理
│   │   ├── __init__.py
│   │   ├── transforms.py         # 数据预处理
│   │   ├── dataset.py            # 数据集加载 (LVISDataset等)
│   │   └── sampler.py            # 采样策略
│   ├── losses/                   # [成员B] 损失函数
│   │   ├── __init__.py
│   │   ├── focal_loss.py
│   │   ├── giou_loss.py
│   │   ├── l1_loss.py
│   │   └── grounding_loss.py
│   ├── eval/                     # [成员B] 评估
│   │   ├── __init__.py
│   │   └── lvis_evaluator.py
│   ├── train/                    # [成员C] 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── config.py
│   │   └── utils.py
│   └── experiments/              # [成员C] 实验
│       ├── __init__.py
│       └── vlm_comparison.py
├── scripts/                      # 工具脚本
│   ├── convert_weights_pytorch_to_jittor.py # 权重转换
│   ├── eval_lvis_zeroshot_full.py  # LVIS Zero-Shot 完整评估
│   ├── quick_test_zeroshot.py      # 快速推理测试
│   ├── run_inference.py            # 推理脚本
│   ├── finetune.py                 # 微调脚本
│   ├── coco2odvg.py                # COCO格式转换
│   └── goldg2odvg.py               # GoldG格式转换
├── requirements.txt
└── README.md
```

## Evaluation Scripts Comparison

This project includes several evaluation scripts with different purposes and trade-offs:

### Script Overview

| Script | Purpose | Zero-Shot | Speed | Use Case |
|--------|----------|-----------|-------|----------|
| `eval_lvis_zeroshot_full.py` | Official benchmarking | ✓ True (all 1203 cats) | Slow (~1-5s/img) | Research, papers |
| `quick_test_zeroshot.py` | Development/testing | ✗ Uses GT | Fast (~0.1-0.5s/img) | Debugging, visualization |
| `eval_lvis_zeroshot.py` | Alternative evaluation | ✗ Uses GT | Medium | Development |
| `eval_lvis_zeroshot_final.py` | Debug version | ✗ Partial (25 cats) | Medium | Token mapping debugging |

### Key Differences

#### 1. Category Handling

**`eval_lvis_zeroshot_full.py` (True Zero-Shot)**
- Processes ALL 1203 LVIS categories in batches (default 80 per batch)
- Uses PyTorch's `build_captions_and_token_span()` for proper token mapping
- Multiple forward passes per image (~15 for full evaluation)
- Results comparable to Grounding DINO paper

**`quick_test_zeroshot.py` (Non-Zero-Shot)**
- Uses ONLY ground truth categories from each image
- Typically 2-10 categories per image
- Single forward pass per image
- Good for quick sanity checks but NOT for benchmarking

#### 2. Token-to-Category Mapping

**`eval_lvis_zeroshot_full.py`**
```python
# Uses positive map matrix from PyTorch utilities
positive_map = create_positive_map_from_span(tokenized, tokenspanlist, max_text_len)
prob_to_label = prob_to_token @ positive_map_np.T
```

**`quick_test_zeroshot.py`**
```python
# Simple argmax approach
pred_probs = jt.sigmoid(pred_logits)
max_probs, pred_labels = jt.argmax(pred_probs, dim=-1)
```

#### 3. Evaluation Method

**`eval_lvis_zeroshot_full.py`**
- Official COCO/LVIS evaluation
- Full metric suite: AP, AP50, AP75, APs, APm, APl, APr, APc, APf
- Reproducible and comparable with paper results

**`quick_test_zeroshot.py`**
- Custom IoU-based TP calculation
- Simple precision/recall/F1
- Includes visualization of bounding boxes

### When to Use Which Script?

#### Use `eval_lvis_zeroshot_full.py` when:
- ✓ Running official benchmarks for research papers
- ✓ Comparing with Grounding DINO paper metrics
- ✓ Need all COCO/LVIS metrics

#### Use `quick_test_zeroshot.py` when:
- ✓ Debugging model inference
- ✓ Visualizing predictions on sample images
- ✓ Quick sanity checks during development
- ✓ Testing model loading and basic functionality
- ✓ Verifying output format

### Performance Characteristics

| Metric | eval_lvis_zeroshot_full | quick_test_zeroshot |
|--------|-------------------------|---------------------|
| **Categories processed** | 1203 | ~5 (GT only) |
| **Forward passes/image** | ~15 | 1 |
| **Memory usage** | Higher | Lower |
| **Time per image** | 1-5 seconds | 0.1-0.5 seconds |
| **Total time (100 images)** | ~2-8 minutes | ~10-50 seconds |
| **Visualization** | No | Yes |
| **Official metrics** | Yes | No |

### OWL-ViT Comparison Script

**`eval_owlvit_lvis.py`** - Compare with OWL-ViT baseline model

This script evaluates Google's OWL-ViT model on the same LVIS dataset for direct performance comparison with Grounding DINO.

#### Key Features:
- Uses HuggingFace `transformers` library for OWL-ViT
- Processes same LVIS minival dataset (1203 categories)
- Generates identical output format and metrics as GroundingDINO
- Enables direct quantitative comparison (AP, APr, APc, APf)

#### Usage:
```bash
# Quick test (100 images)
python scripts/eval_owlvit_lvis.py --num_images 100 --batch_size 25

# Full evaluation
python scripts/eval_owlvit_lvis.py --full --batch_size 25

# Custom model variant
python scripts/eval_owlvit_lvis.py \
    --model_name 'google/owlvit-large-patch14' \
    --num_images 500 \
    --output_dir outputs/owlvit_large
```

#### Test Setup:
```bash
# Verify installation and data access
python scripts/test_owlvit_quick.py
```

#### Requirements:
- `transformers >= 4.20.0`
- `torch >= 1.13.0`
- `torchvision >= 0.14.0`
- LVIS dataset (same as GroundingDINO evaluation)

#### Output:
- `outputs/owlvit/predictions.jsonl` - Incremental predictions
- `outputs/owlvit/lvis_predictions.json` - Final predictions for LVISEval
- `outputs/owlvit/lvis_zeroshot_results.json` - Metrics comparable to GroundingDINO

#### When to Use:
- ✓ Benchmarking against OWL-ViT baseline
- ✓ VLM performance comparison studies
- ✓ Understanding open-vocabulary detection capabilities
- ✓ Research requiring multiple model comparisons

### Example Usage

```bash
# True zero-shot evaluation (use for benchmarks)
python scripts/eval_lvis_zeroshot_full.py --num_images 100 --gpu 0

# Quick testing with visualization (use for debugging)
python scripts/quick_test_zeroshot.py \
    --num_images 10 \
    --output_dir outputs/quick_test \
    --box_threshold 0.1
```

## Installation

### 前置要求

- Anaconda 或 Miniconda (推荐使用 conda 管理环境)
- Python 3.9
- CUDA (可选，用于 GPU 加速)

### 快速安装 (推荐方法)

**注意**: 请使用项目根目录 `GroundingDINO-Light/.venv` 中的虚拟环境，而不是 `GroundingDINO_Jittor/.venv`。

如果 conda 创建环境很慢，建议直接使用以下命令：

```bash
# 1. 使用项目根目录的虚拟环境
cd ..  # 返回到 GroundingDINO-Light 根目录
source .venv/bin/activate  # 使用根目录的 .venv

# 2. 进入 Jittor 项目目录
cd GroundingDINO_Jittor

# 3. 使用 pip 安装所有依赖 (更快)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 验证安装
python -c "import jittor as jt; print(f'Jittor: {jt.__version__}')"
python -c "import torch, transformers, timm, pycocotools; print('所有依赖安装成功!')"
```

### 使用 Conda 环境文件 (较慢)

如果网络较好，可以使用：

```bash
# 配置国内镜像源 (加速)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate groundingdino_jittor
```

### 主要依赖

- **jittor** >= 1.3.0 - 核心深度学习框架
- **torch** >= 1.13.0 - 用于 BERT 模型和权重转换
- **transformers** >= 4.20.0 - BERT 文本编码器
- **timm** >= 0.6.0 - Swin Transformer backbone
- **pycocotools** >= 2.0.4 - LVIS/COCO 评估
- numpy, pillow, matplotlib - 数据处理和可视化

### 常见问题

- **conda 命令找不到**: 使用 Anaconda Prompt (Windows) 或重启终端
- **环境创建失败**: 检查网络连接，或手动创建环境后使用 `pip install -r requirements.txt`
- **GPU 支持**: Jittor 会自动检测 CUDA，无需手动配置

## Quick Start - Inference

### 1. 下载预训练权重

从官方 GitHub 下载 PyTorch 预训练权重：

```bash
# 创建 weights 目录
mkdir weights
cd weights

# 下载 Swin-T 版本权重 (~694MB)
# 方法1: 使用 wget (Linux/Mac)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 方法2: 使用浏览器直接下载
# 访问: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ..
```

### 2. 转换权重到 Jittor 格式

```bash
python scripts/convert_weights_pytorch_to_jittor.py \
    --pytorch_weight weights/groundingdino_swint_ogc.pth \
    --output weights/groundingdino_swint_ogc_jittor.pkl
```

转换成功后会显示：
```
成功加载 940 个权重
成功保存 940 个权重
转换完成！
```
### 下载bert模型放在Grounding-Dino-Light/GroundingDINO_Jittor/models
### 下载数据到Grounding-Dino-Light/GroundingDINO_Jittor/data/coco/val2017；Grounding-Dino-Light/GroundingDINO_Jittor/data/lvis_notation

### 3. 运行推理

#### 演示模式（自动创建测试图像）

```bash
python scripts/run_inference.py --demo
```

#### 自定义图像推理

```bash
python scripts/run_inference.py \
    --image your_image.jpg \
    --text "cat . dog . person ." \
    --output result.jpg
```

#### 完整参数

```bash
python scripts/run_inference.py \
    --image <图像路径> \
    --text <文本提示，用 . 分隔不同类别> \
    --output <输出路径> \
    --box_threshold 0.35 \
    --text_threshold 0.25
```

### LVIS Zero-Shot Evaluation

Run the full zero-shot evaluation on LVIS dataset:

```bash
# Quick test on 100 images
python scripts/eval_lvis_zeroshot_full.py --num_images 100 --gpu 0

# Full validation set (~17K images, ~85 hours)
python scripts/eval_lvis_zeroshot_full.py --full --gpu 0

# Custom parameters
python scripts/eval_lvis_zeroshot_full.py \
    --num_images 500 \
    --batch_size 80 \
    --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
    --lvis_ann data/lvis_notation/lvis_v1_val.json/lvis_v1_val.json \
    --image_dir data/coco/val2017 \
    --output_dir outputs
```

### LVIS Fine-tuning

Fine-tune Grounding DINO on LVIS dataset to achieve **AP 52.1** (target from paper):

```bash
# Quick test (verify script works)
python scripts/finetune_lvis_full.py --test_only --num_samples 10 --epochs 2 --gpu 0

# Full fine-tuning (recommended settings from paper)
python scripts/finetune_lvis_full.py \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_drop 15 \
    --output_dir outputs/finetune_lvis \
    --gpu 0

# With frozen backbone (faster, less memory)
python scripts/finetune_lvis_full.py \
    --epochs 20 \
    --batch_size 8 \
    --freeze_backbone \
    --output_dir outputs/finetune_frozen_backbone \
    --gpu 0
```

**Fine-tuning Targets:**

| Metric | Target |
|--------|--------|
| AP | 52.1% |
| APr (rare) | 35.4% |
| APc (common) | 51.3% |
| APf (frequent) | 55.7% |

**Training Notes:**
- Full training on LVIS (~100K images) takes approximately 40-60 hours on a single GPU
- Recommended: Use multi-GPU training or freeze backbone to reduce training time
- Learning rate drops by 10x at epoch 15 (configurable via `--lr_drop`)
- Checkpoints saved every 5 epochs and at best validation loss

### 推理示例

```python
from jittor_implementation.util.inference import GroundingDINOInference

# 初始化模型
model = GroundingDINOInference(
    weight_path="weights/groundingdino_swint_ogc_jittor.pkl",
    device="cuda",
    box_threshold=0.35,
    text_threshold=0.25,
)

# 执行推理
boxes, scores, phrases = model.predict(
    image="path/to/image.jpg",
    caption="cat . dog . person ."
)

# 推理并可视化
result_image = model.predict_and_visualize(
    image_path="path/to/image.jpg",
    caption="cat . dog . person .",
    output_path="output.jpg"
)
```

---

## Usage

### Data Loading

```python
from jittor_implementation.data import build_dataset, get_dataloader

# Build dataset
dataset = build_dataset('train', args)

# Create dataloader with LVIS sampler
dataloader = get_dataloader(
    dataset, 
    batch_size=4, 
    sampler_type='lvis',
    sampler_kwargs={'samples_per_epoch': 1000}
)
```

### Loss Functions

```python
from jittor_implementation.losses import GroundingLoss, SetCriterion

# Create loss function
criterion = SetCriterion(
    num_classes=1203,
    weight_dict={'loss_ce': 2.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
    losses=['labels', 'boxes', 'giou']
)

# Calculate loss
outputs = model(images)
losses = criterion(outputs, targets)
```

### Evaluation

```python
from jittor_implementation.eval import evaluate_lvis

# Evaluate model
metrics = evaluate_lvis(
    model, 
    dataloader, 
    ann_file='path/to/lvis_val.json',
    output_dir='./eval_results'
)

print(f"AP: {metrics['AP']:.4f}")
```

### Data Format Conversion

```bash
# Convert COCO to ODVG
python scripts/coco2odvg.py --coco_path path/to/coco.json --output_path path/to/odvg.json --image_dir path/to/images

# Convert GoldG to ODVG
python scripts/goldg2odvg.py --goldg_path path/to/goldg.json --output_path path/to/odvg.json --image_dir path/to/images
```

### Text Encoding

```python
from jittor_implementation.models.text_encoder import BERTWrapper

# Initialize text encoder
text_encoder = BERTWrapper(
    model_name='bert-base-uncased',
    max_text_len=256
)

# Process text
text = ["person . dog . cat"]
text_dict = text_encoder(text)

# Access encoded features
encoded_text = text_dict["encoded_text"]  # (B, L, D)
text_token_mask = text_dict["text_token_mask"]  # (B, L)
position_ids = text_dict["position_ids"]  # (B, L)
```

### Feature Fusion

```python
from jittor_implementation.models.fusion import FeatureFusion

# Initialize fusion module
fusion = FeatureFusion(
    hidden_dim=256,
    num_heads=8,
    dropout=0.1
)

# Fuse visual and text features
fused_features = fusion(
    visual_features,  # (B, H, W, D) or (B, N, D)
    text_features,    # (B, L, D)
    text_token_mask   # (B, L)
)
```

### Query Generation

```python
from jittor_implementation.models.query import LanguageGuidedQuery

# Initialize query generator
query_generator = LanguageGuidedQuery(
    hidden_dim=256,
    num_queries=900
)

# Generate queries from text
queries = query_generator(
    text_features,  # (B, L, D)
    text_token_mask  # (B, L)
)
```

### Training

```python
from jittor_implementation.train.config import TrainingConfig
from jittor_implementation.train.trainer import Trainer

# Create configuration
config = TrainingConfig()
config.model_name = "groundingdino_swin-t"
config.batch_size = 4
config.epochs = 40

# Create trainer
trainer = Trainer(
    model=model,
    text_encoder=text_encoder,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)

# Start training
trainer.train()
```

### Command-line Training

```bash
# Train model with default configuration
python -m jittor_implementation.train.trainer \
  --model_name groundingdino_swin-t \
  --batch_size 4 \
  --epochs 40 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --data_path /path/to/dataset \
  --output_dir ./outputs \
  --checkpoint_dir ./checkpoints

# Resume training from checkpoint
python -m jittor_implementation.train.trainer \
  --model_name groundingdino_swin-t \
  --resume ./checkpoints/groundingdino_latest.pth \
  --data_path /path/to/dataset \
  --output_dir ./outputs \
  --checkpoint_dir ./checkpoints
```

### VLM Comparison

```python
from jittor_implementation.experiments.vlm_comparison import VLMComparator

# Initialize comparator
comparator = VLMComparator(
    model=model,
    text_encoder=text_encoder,
    config=config,
    output_dir="./comparison_results"
)

# Process images with text prompts
results = comparator.run_comparison(
    image_list=["image1.jpg", "image2.jpg"],
    text_prompts=["person", "dog", "cat"],
    save_visualizations=True
)
```

### Command-line VLM Comparison

```bash
# Compare model outputs on test images
python -m jittor_implementation.experiments.vlm_comparison \
  --checkpoint_path ./checkpoints/groundingdino_best.pth \
  --image_list image1.jpg image2.jpg image3.jpg \
  --text_prompts "person . dog" "car . bicycle" "cat . bird" \
  --output_dir ./comparison_results \
  --save_visualizations
```
```bash
# Start two gpu run on the whole LVIS/val dataset
 cd GroundingDINO_Jittor && source ../.venv/bin/activate && python scripts/eval_lvis_zeroshot_full.py --full --n_gpus 2 --checkpoint_interval 500 --image_dir ../val2017 --image_dir_fallback ../train2017 --output_dir outputs/lvis_full_2gpu --resume 2>&1 | tee lvis_eval_fixed.log
```
```bash
# new startup
cd GroundingDINO_Jittor && source ../.venv/bin/activate && python scripts/eval_lvis_zeroshot_full.py --num_images 10
```
```bash
# new ablation
source .venv/bin/activate && cd GroundingDINO_Jittor && python scripts/eval_Gdino_ablation.py --ablation no_text_cross_attn --num_images 10
```