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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

(Instructions will be added as modules are implemented)

