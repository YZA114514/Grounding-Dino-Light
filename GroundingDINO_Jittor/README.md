# GroundingDINO Jittor Implementation

This project is a Jittor implementation of GroundingDINO, as part of the 2025 Final Project.

## Environment Setup

To run this project, you must use the provided virtual environment where Jittor and other dependencies are installed.

```bash
source /root/shared-nvme/GroundingDINO-Light/.venv/bin/activate
```

Or run python directly from the virtual environment:

```bash
/root/shared-nvme/GroundingDINO-Light/.venv/bin/python <script_name>.py
```

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
│   ├── convert_weights_pytorch_to_jittor.py # [成员A]
│   ├── coco2odvg.py              # [成员B]
│   └── goldg2odvg.py             # [成员B]
├── requirements.txt
└── README.md
```

## Implementation Status

### Member B (Data Processing & Evaluation) - ✅ COMPLETED

#### 1. Data Format Conversion Scripts ✅
- `scripts/coco2odvg.py`: Converts COCO format to ODVG format
- `scripts/goldg2odvg.py`: Converts GoldG format to ODVG format

#### 2. Data Preprocessing Module ✅
- `data/transforms.py`: Implements data transformations for Jittor
  - Image transformations: RandomCrop, RandomSizeCrop, CenterCrop, RandomHorizontalFlip, RandomResize
  - Tensor transformations: ToTensor, Normalize
  - Utility functions: crop, hflip, resize, pad
  - `build_transforms()`: Builds transformation pipeline for training/evaluation

#### 3. LVIS Data Loader ✅
- `data/dataset.py`: Implements dataset classes for Jittor
  - `LVISDataset`: LVIS dataset loader with proper handling of annotations
  - `ODVGDataset`: ODVG format dataset loader
  - `build_dataset()`: Factory function to create datasets based on configuration

#### 4. Data Sampling Strategy ✅
- `data/sampler.py`: Implements sampling strategies for long-tailed distribution
  - `LVISSampler`: Handles LVIS long-tailed distribution with repeat factors
  - `BalancedSampler`: Ensures equal representation of categories
  - `DistributedSampler`: For multi-GPU training
  - `get_dataloader()`: Factory function to create dataloaders with samplers

#### 5. Loss Functions ✅
- `losses/focal_loss.py`: Focal loss implementation for class imbalance
  - `FocalLoss`: Standard focal loss for classification
  - `SigmoidFocalLoss`: Sigmoid-based focal loss for multi-label classification

- `losses/giou_loss.py`: IoU-based losses for bounding box regression
  - `GIoULoss`: Generalized IoU loss
  - `DIoULoss`: Distance IoU loss
  - `CIoULoss`: Complete IoU loss
  - Utility functions: box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_area, box_iou, box_giou

- `losses/l1_loss.py`: L1-based losses for bounding box regression
  - `L1Loss`: Standard L1 loss
  - `SmoothL1Loss`: Smooth L1 loss (Huber loss)
  - `WeightedL1Loss`: Weighted L1 loss
  - `WeightedSmoothL1Loss`: Weighted smooth L1 loss

- `losses/grounding_loss.py`: Combined loss for Grounding DINO
  - `GroundingLoss`: Combines classification and bounding box regression losses
  - `SetCriterion`: Set criterion for DETR-style models
  - Matching algorithm for predictions and targets

#### 6. LVIS Evaluation Script ✅
- `eval/lvis_evaluator.py`: LVIS evaluation implementation
  - `LVISEvaluator`: Main evaluator class
  - `evaluate_lvis()`: Function to evaluate model on LVIS dataset
  - Support for both pycocotools and simple evaluation
  - Metrics: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl

### Member C (Text Processing & Training) - ✅ COMPLETED

#### 1. Interface Definitions ✅
- `models/interfaces.py`: Defines interfaces between different components
  - Model input/output interfaces for compatibility between modules
  - Data interfaces for standardized data flow
  - Text encoder interfaces
  - Feature fusion interfaces
  - Query generation interfaces
  - Training and evaluation interfaces

#### 2. BERT Text Encoder Wrapper ✅
- `models/text_encoder/bert_wrapper.py`: Complete BERT text encoding implementation
  - `BertModelWarper`: Wrapper for PyTorch's BERT model to work with Jittor
  - `TextEncoderShell`: Shell wrapper for text encoding
  - `BERTWrapper`: Complete BERT wrapper for GroundingDINO with special token handling
  - Functions for generating attention masks with special tokens

#### 3. Text Processor ✅
- `models/text_encoder/text_processor.py`: Clause-level text processing
  - `TextProcessor`: Handles clause-level text processing with phrase extraction
  - `PhraseProcessor`: Processes text features at the phrase level
  - Support for sub-sentence presentation and category-to-token masking

#### 4. Feature Fusion Module ✅
- `models/fusion/feature_fusion.py`: Multiple fusion strategies for visual-language features
  - `FeatureFusion`: Basic visual-language feature fusion using cross-attention
  - `ContrastiveEmbed`: Contrastive embedding for classification
  - `LanguageGuidedFusion`: Language-guided feature fusion
  - `DynamicFusion`: Dynamic fusion with multiple strategies (concat, add, gate)

#### 5. Language-Guided Query Generation ✅
- `models/query/language_guided_query.py`: Multiple query generation strategies
  - `LanguageGuidedQuery`: Basic language-guided query generation
  - `DynamicQueryGenerator`: Dynamic query generator based on text content
  - `AdaptiveQueryGenerator`: Adaptive number of queries based on text complexity
  - `TextConditionalQueryGenerator`: Text-conditional query generation
  - `PositionalEncoding`: Positional encoding for queries

#### 6. Training Configuration ✅
- `train/config.py`: Comprehensive training configuration
  - `TrainingConfig`: Complete configuration class with all training hyperparameters
  - Argument parser for command-line configuration
  - Predefined configurations for different models (Swin-T, Swin-B)
  - Debug configuration for testing

#### 7. Training Utilities ✅
- `train/utils.py`: Utility functions for training
  - Reproducibility functions (seed setting)
  - Model saving/loading functions
  - Distributed training setup functions
  - Metric logging utilities
  - Image visualization functions
  - Learning rate adjustment functions
  - Optimizer parameter grouping for different learning rates
  - Data format conversion between PyTorch and Jittor

#### 8. Training Script ✅
- `train/trainer.py`: Complete training implementation
  - `Trainer`: Complete trainer class with training loop
  - Support for validation and evaluation
  - Model checkpointing and best model saving
  - Integration with Weights & Biases for logging
  - Main function for training

#### 9. VLM Comparison Experiment ✅
- `experiments/vlm_comparison.py`: Vision-Language Model comparison
  - `VLMComparator`: Class for comparing Vision-Language Models
  - Image processing and visualization
  - Comparison with baseline models
  - Main function for running experiments

### Member A (Model Architecture) - ✅ COMPLETED

#### 1. Multi-Scale Deformable Attention ✅
- `models/attention/ms_deform_attn.py`: Multi-Scale Deformable Attention
  - `MSDeformAttn`: Core deformable attention module
  - Support for multi-scale feature maps
  - Pure Jittor implementation (no CUDA kernel)

#### 2. MultiheadAttention ✅
- `models/attention/multihead_attention.py`: Standard multi-head attention
  - Custom implementation for Jittor compatibility

#### 3. Transformer Encoder ✅
- `models/transformer/encoder.py`: Transformer Encoder
  - `DeformableTransformerEncoderLayer`: Encoder layer with deformable attention
  - `TransformerEncoder`: Full encoder stack
  - `BiAttentionBlock`: Bi-directional attention for feature fusion

#### 4. Transformer Decoder ✅
- `models/transformer/decoder.py`: Transformer Decoder
  - `DeformableTransformerDecoderLayer`: Decoder layer with text cross-attention
  - `TransformerDecoder`: Full decoder stack with iterative refinement
  - `MLP`: Multi-layer perceptron for predictions

#### 5. DINO Detection Head ✅
- `models/head/dino_head.py`: DINO detection head
  - `ContrastiveEmbed`: Contrastive embedding for open-vocabulary classification
  - `DINOHead`: Complete detection head with bbox regression
  - `MLP`: Bounding box regression network

#### 6. Swin Transformer Backbone ✅
- `models/backbone/swin_transformer.py`: Swin Transformer backbone
  - Full Swin-T/Swin-B implementation
  - Multi-scale feature extraction
  - Converted from PyTorch to Jittor API

#### 7. Complete Model Assembly ✅
- `models/groundingdino.py`: Complete GroundingDINO model
  - Integration of all components
  - Support for captions input
  - Text encoding and feature fusion

#### 8. Weight Conversion Script ✅
- `scripts/convert_weights_pytorch_to_jittor.py`: PyTorch to Jittor weight conversion
  - Converts official PyTorch weights to Jittor format
  - Supports both Swin-T and Swin-B models

#### 9. Inference Utilities ✅
- `util/inference.py`: Complete inference pipeline
  - Image preprocessing
  - Text processing
  - Post-processing and visualization
  - `GroundingDINOInference`: Easy-to-use inference class

## Installation

### 前置要求

- Anaconda 或 Miniconda (推荐使用 conda 管理环境)
- Python 3.9
- CUDA (可选，用于 GPU 加速)

### 快速安装 (推荐方法)

如果 conda 创建环境很慢，建议直接使用以下命令：

```bash
# 1. 创建基础环境
conda create -n groundingdino_jittor python=3.19 -y

# 2. 激活环境
conda activate groundingdino_jittor

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

