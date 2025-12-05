# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# 微调示例代码
# ------------------------------------------------------------------------
"""
微调示例代码

展示如何：
1. 加载预训练权重
2. 冻结 backbone 和 text encoder
3. 只训练检测头和融合模块
4. 在自定义数据集上微调

使用方法:
    python finetune_example.py --pretrained_weights ../weights/groundingdino_swint_ogc_jittor.pkl
"""

import os
import sys
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import jittor as jt
    from jittor import nn
    from jittor.dataset import Dataset
except ImportError:
    print("请先安装 Jittor: pip install jittor")
    sys.exit(1)


# ============================================================
# 冻结/解冻权重的工具函数
# ============================================================

def freeze_params(params: List[jt.Var]):
    """
    冻结参数（停止计算梯度）
    
    在 Jittor 中，使用 stop_grad() 来冻结参数
    """
    for param in params:
        param.stop_grad()


def unfreeze_params(params: List[jt.Var]):
    """
    解冻参数（恢复计算梯度）
    """
    for param in params:
        param.start_grad()


def freeze_module(module: nn.Module):
    """
    冻结整个模块的所有参数
    """
    for param in module.parameters():
        param.stop_grad()


def unfreeze_module(module: nn.Module):
    """
    解冻整个模块的所有参数
    """
    for param in module.parameters():
        param.start_grad()


def get_param_groups(model: nn.Module, freeze_backbone: bool = True, freeze_text_encoder: bool = True):
    """
    获取参数分组，用于差异化学习率
    
    Args:
        model: 模型
        freeze_backbone: 是否冻结 backbone
        freeze_text_encoder: 是否冻结文本编码器
        
    Returns:
        param_groups: 参数分组列表，每组可以设置不同的学习率
    """
    backbone_params = []
    text_encoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name or 'swin' in name.lower():
            backbone_params.append(param)
            if freeze_backbone:
                param.stop_grad()
        elif 'bert' in name.lower() or 'text_encoder' in name:
            text_encoder_params.append(param)
            if freeze_text_encoder:
                param.stop_grad()
        else:
            other_params.append(param)
    
    print(f"\n参数分组:")
    print(f"  Backbone: {len(backbone_params)} 个参数 {'(冻结)' if freeze_backbone else '(可训练)'}")
    print(f"  Text Encoder: {len(text_encoder_params)} 个参数 {'(冻结)' if freeze_text_encoder else '(可训练)'}")
    print(f"  其他 (Transformer/Head): {len(other_params)} 个参数 (可训练)")
    
    # 返回参数组（只返回可训练的参数）
    param_groups = []
    
    if not freeze_backbone and backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': 1e-5,  # backbone 用较小的学习率
            'name': 'backbone'
        })
    
    if not freeze_text_encoder and text_encoder_params:
        param_groups.append({
            'params': text_encoder_params,
            'lr': 1e-5,  # text encoder 用较小的学习率
            'name': 'text_encoder'
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': 1e-4,  # 其他模块用较大的学习率
            'name': 'other'
        })
    
    return param_groups


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Returns:
        (total_params, trainable_params)
    """
    total = 0
    trainable = 0
    
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    
    return total, trainable


# ============================================================
# 加载预训练权重
# ============================================================

def load_pretrained_weights(
    model: nn.Module,
    weight_path: str,
    freeze_backbone: bool = True,
    freeze_text_encoder: bool = True,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """
    加载预训练权重并进行冻结设置
    
    Args:
        model: Jittor 模型
        weight_path: 权重文件路径
        freeze_backbone: 是否冻结 backbone
        freeze_text_encoder: 是否冻结文本编码器
        strict: 是否严格匹配所有权重
        
    Returns:
        加载结果字典
    """
    print(f"\n{'='*60}")
    print("加载预训练权重")
    print(f"{'='*60}")
    print(f"权重文件: {weight_path}")
    
    # 加载权重
    if weight_path.endswith('.pkl'):
        with open(weight_path, 'rb') as f:
            pretrained_weights = pickle.load(f)
    else:
        # 假设是 PyTorch 格式
        try:
            import torch
            checkpoint = torch.load(weight_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            pretrained_weights = {k: v.numpy() for k, v in state_dict.items()}
        except ImportError:
            raise RuntimeError("加载 .pth 文件需要安装 PyTorch")
    
    print(f"预训练权重数量: {len(pretrained_weights)}")
    
    # 获取模型的 state_dict
    model_state = model.state_dict()
    print(f"模型权重数量: {len(model_state)}")
    
    # 加载权重
    matched = []
    missing_in_pretrained = []
    missing_in_model = []
    shape_mismatch = []
    
    for name, param in model_state.items():
        if name in pretrained_weights:
            weight = pretrained_weights[name]
            if param.shape == tuple(weight.shape):
                param.update(jt.array(weight))
                matched.append(name)
            else:
                shape_mismatch.append((name, param.shape, tuple(weight.shape)))
        else:
            missing_in_pretrained.append(name)
    
    for name in pretrained_weights:
        if name not in model_state:
            missing_in_model.append(name)
    
    # 打印结果
    print(f"\n加载结果:")
    print(f"  ✓ 成功加载: {len(matched)}")
    print(f"  ✗ 模型中缺少: {len(missing_in_model)}")
    print(f"  ✗ 预训练中缺少: {len(missing_in_pretrained)}")
    print(f"  ✗ 形状不匹配: {len(shape_mismatch)}")
    
    if shape_mismatch:
        print("\n形状不匹配的权重:")
        for name, model_shape, weight_shape in shape_mismatch[:5]:
            print(f"  {name}: 模型{model_shape} vs 预训练{weight_shape}")
    
    # 冻结设置
    print(f"\n冻结设置:")
    print(f"  Backbone: {'冻结' if freeze_backbone else '可训练'}")
    print(f"  Text Encoder: {'冻结' if freeze_text_encoder else '可训练'}")
    
    for name, param in model.named_parameters():
        should_freeze = False
        
        if freeze_backbone and ('backbone' in name or 'swin' in name.lower()):
            should_freeze = True
        elif freeze_text_encoder and ('bert' in name.lower() or 'text_encoder' in name):
            should_freeze = True
        
        if should_freeze:
            param.stop_grad()
    
    # 统计参数
    total, trainable = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数: {total:,} ({total/1e6:.2f}M)")
    print(f"  可训练: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  冻结: {total - trainable:,} ({(total-trainable)/1e6:.2f}M)")
    
    return {
        'matched': matched,
        'missing_in_pretrained': missing_in_pretrained,
        'missing_in_model': missing_in_model,
        'shape_mismatch': shape_mismatch,
    }


# ============================================================
# 简单的模型示例（用于演示）
# ============================================================

class DummyGroundingDINO(nn.Module):
    """
    用于演示的简化模型
    
    实际使用时替换为完整的 GroundingDINO 模型
    """
    def __init__(self, d_model=256):
        super().__init__()
        
        # Backbone (Swin Transformer) - 通常冻结
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),
        )
        
        # Text Encoder (BERT) - 通常冻结
        self.text_encoder = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Transformer - 可训练
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.transformer_decoder = nn.TransformerDecoderLayer(d_model, nhead=8)
        
        # Detection Head - 可训练
        self.class_embed = nn.Linear(d_model, 256)  # 256 = max_text_len
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )
    
    def execute(self, images, text_features):
        # 这是简化的前向传播
        visual_features = self.backbone(images)
        text_features = self.text_encoder(text_features)
        
        # ... 省略 transformer 处理 ...
        
        # 输出
        bs = images.shape[0]
        query_features = jt.randn(bs, 100, 256)  # 假设 100 个 query
        
        pred_logits = self.class_embed(query_features)
        pred_boxes = jt.sigmoid(self.bbox_embed(query_features))
        
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}


# ============================================================
# 训练循环示例
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: jt.optim.Optimizer,
    epoch: int,
):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # 前向传播
        outputs = model(images, targets['text_features'])
        
        # 计算损失（简化版）
        loss_cls = nn.cross_entropy_loss(
            outputs['pred_logits'].reshape(-1, 256),
            targets['labels'].reshape(-1),
        )
        loss_bbox = nn.l1_loss(outputs['pred_boxes'], targets['boxes'])
        
        loss = loss_cls + 5 * loss_bbox
        
        # 反向传播和优化
        optimizer.step(loss)
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="微调示例")
    
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="预训练权重路径"
    )
    
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="冻结 backbone"
    )
    
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        default=True,
        help="冻结文本编码器"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grounding DINO 微调示例")
    print("=" * 60)
    
    # 创建模型
    print("\n创建模型...")
    model = DummyGroundingDINO(d_model=256)
    
    # 加载预训练权重（如果提供）
    if args.pretrained_weights:
        load_pretrained_weights(
            model,
            args.pretrained_weights,
            freeze_backbone=args.freeze_backbone,
            freeze_text_encoder=args.freeze_text_encoder,
        )
    else:
        print("\n未提供预训练权重，使用随机初始化")
        
        # 手动冻结设置
        if args.freeze_backbone:
            freeze_module(model.backbone)
            print("已冻结 backbone")
        
        if args.freeze_text_encoder:
            freeze_module(model.text_encoder)
            print("已冻结 text_encoder")
    
    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数数量: {len(trainable_params)}")
    
    # 创建优化器（只优化可训练参数）
    optimizer = jt.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    print(f"\n配置:")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  Backbone: {'冻结' if args.freeze_backbone else '可训练'}")
    print(f"  Text Encoder: {'冻结' if args.freeze_text_encoder else '可训练'}")
    
    print("\n" + "=" * 60)
    print("准备开始训练...")
    print("（这是示例代码，实际训练需要提供数据集）")
    print("=" * 60)
    
    # 示例：打印模型结构
    print("\n模型结构:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status = "可训练" if trainable_count > 0 else "冻结"
        print(f"  {name}: {param_count:,} 参数 ({status})")


if __name__ == "__main__":
    main()



