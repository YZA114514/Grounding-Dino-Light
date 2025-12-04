# GroundingDINO Jittor Implementation

This project is a Jittor implementation of GroundingDINO, as part of the 2025 Final Project.

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

## Installation

```bash
pip install -r requirements.txt
```

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

