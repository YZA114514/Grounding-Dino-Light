#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - 微调脚本

用法:
    python scripts/finetune.py \
        --pretrained_weights weights/groundingdino_swint_ogc_jittor.pkl \
        --data_path /path/to/lvis \
        --output_dir ./outputs \
        --epochs 10 \
        --batch_size 4 \
        --lr 1e-4
"""

import os
import sys
import argparse
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np

try:
    import jittor as jt
    from jittor import nn
    from jittor.dataset import Dataset
except ImportError:
    print("错误: 请先安装 Jittor")
    print("  pip install jittor")
    sys.exit(1)


# ============================================================
# 工具函数
# ============================================================

def freeze_module(module: nn.Module, name: str = ""):
    """冻结模块的所有参数"""
    count = 0
    for param in module.parameters():
        param.stop_grad()
        count += 1
    if name:
        print(f"  冻结 {name}: {count} 个参数")
    return count


def count_parameters(model: nn.Module):
    """统计模型参数"""
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def load_pretrained_weights(model: nn.Module, weight_path: str):
    """加载预训练权重"""
    print(f"\n加载预训练权重: {weight_path}")
    
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    # 清理权重名称
    cleaned = {}
    for k, v in weights.items():
        if k.startswith('module.'):
            k = k[7:]
        cleaned[k] = v
    
    # 加载到模型
    model_state = model.state_dict()
    loaded = 0
    
    for k, v in cleaned.items():
        if k in model_state and model_state[k].shape == tuple(v.shape):
            model_state[k] = jt.array(v)
            loaded += 1
    
    model.load_state_dict(model_state)
    print(f"成功加载 {loaded}/{len(cleaned)} 个权重")
    
    return loaded


# ============================================================
# 简单数据集（用于快速测试）
# ============================================================

class SimpleDataset(Dataset):
    """简单数据集（用于快速测试训练流程）"""
    
    def __init__(self, num_samples=100, image_size=800, num_queries=100):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_queries = num_queries
        self.total_len = num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像
        image = jt.randn(3, self.image_size, self.image_size) * 0.1
        
        # 生成随机目标
        num_objects = np.random.randint(1, 10)
        boxes = jt.rand(num_objects, 4)
        # 确保 x2 > x1, y2 > y1
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.5
        boxes = jt.clamp(boxes, 0, 1)
        
        labels = jt.randint(0, 256, (num_objects,))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
        }
        
        return image, target
    
    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    """数据整理函数"""
    images = jt.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================
# 训练函数
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, config):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # 创建文本特征（简化）
        bs = images.shape[0]
        text_dict = {
            'encoded_text': jt.randn(bs, 256, 256) * 0.1,
            'text_token_mask': jt.ones(bs, 256).bool(),
        }
        
        # 前向传播
        outputs = model(images, text_dict=text_dict)
        
        # 简化的损失计算
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # 分类损失
        loss_cls = jt.mean(pred_logits ** 2) * 0.1
        
        # 边界框损失
        loss_bbox = jt.mean((pred_boxes - 0.5) ** 2)
        
        # 总损失
        loss = loss_cls + loss_bbox * 5
        
        # 反向传播
        optimizer.step(loss)
        
        total_loss += loss.item()
        num_batches += 1
        
        # 打印进度
        if batch_idx % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, config):
    """验证"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with jt.no_grad():
        for images, targets in dataloader:
            bs = images.shape[0]
            text_dict = {
                'encoded_text': jt.randn(bs, 256, 256) * 0.1,
                'text_token_mask': jt.ones(bs, 256).bool(),
            }
            
            outputs = model(images, text_dict=text_dict)
            
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']
            
            loss_cls = jt.mean(pred_logits ** 2) * 0.1
            loss_bbox = jt.mean((pred_boxes - 0.5) ** 2)
            loss = loss_cls + loss_bbox * 5
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Grounding DINO Jittor 微调')
    
    # 路径参数
    parser.add_argument('--pretrained_weights', type=str, 
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='预训练权重路径')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径（如不提供则使用模拟数据）')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Backbone 学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    
    # 冻结参数 (默认冻结)
    parser.add_argument('--unfreeze_backbone', action='store_true', default=False,
                        help='解冻 backbone (默认冻结)')
    parser.add_argument('--unfreeze_text_encoder', action='store_true', default=False,
                        help='解冻文本编码器 (默认冻结)')
    
    # 其他参数
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=1, help='保存间隔')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--no_gpu', action='store_true', default=False, help='禁用 GPU')
    parser.add_argument('--test_only', action='store_true', help='只测试，不训练')
    
    config = parser.parse_args()
    
    # 设置 Jittor
    if not config.no_gpu:
        jt.flags.use_cuda = 1
        print("使用 GPU 加速")
    else:
        jt.flags.use_cuda = 0
        print("使用 CPU")
    
    # 设置冻结标志
    config.freeze_backbone = not config.unfreeze_backbone
    config.freeze_text_encoder = not config.unfreeze_text_encoder
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Grounding DINO Jittor 微调")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {config.output_dir}")
    
    # ============================================================
    # 1. 创建模型
    # ============================================================
    print("\n[1/4] 创建模型...")
    
    from jittor_implementation.models.groundingdino import GroundingDINO
    
    model = GroundingDINO(
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
    )
    
    # ============================================================
    # 2. 加载预训练权重
    # ============================================================
    print("\n[2/4] 加载预训练权重...")
    
    weight_path = os.path.join(project_root, config.pretrained_weights)
    if os.path.exists(weight_path):
        load_pretrained_weights(model, weight_path)
    else:
        print(f"警告: 权重文件不存在: {weight_path}")
        print("使用随机初始化")
    
    # ============================================================
    # 3. 冻结参数
    # ============================================================
    print("\n[3/4] 配置参数冻结...")
    
    frozen_count = 0
    if config.freeze_backbone:
        for name, module in model.named_modules():
            if 'backbone' in name or 'input_proj' in name:
                frozen_count += freeze_module(module, name)
    
    if config.freeze_text_encoder:
        for name, module in model.named_modules():
            if 'bert' in name.lower() or 'text_encoder' in name or 'feat_map' in name:
                frozen_count += freeze_module(module, name)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  已冻结: {total_params - trainable_params:,}")
    
    # ============================================================
    # 4. 创建数据集和优化器
    # ============================================================
    print("\n[4/4] 创建数据集和优化器...")
    
    # 数据集
    if config.data_path and os.path.exists(config.data_path):
        from jittor_implementation.data import build_dataset, get_dataloader
        train_dataset = build_dataset('train', config)
        val_dataset = build_dataset('val', config)
        train_loader = get_dataloader(train_dataset, config.batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, config.batch_size, shuffle=False)
        print(f"加载数据集: {config.data_path}")
    else:
        print("使用模拟数据集（用于测试训练流程）")
        train_dataset = SimpleDataset(num_samples=100)
        val_dataset = SimpleDataset(num_samples=20)
        train_loader = train_dataset.set_attrs(batch_size=config.batch_size, shuffle=True)
        val_loader = val_dataset.set_attrs(batch_size=config.batch_size, shuffle=False)
    
    # 优化器
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = jt.optim.AdamW(
        trainable_params_list,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    print(f"\n训练配置:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  可训练参数组: {len(trainable_params_list)}")
    
    # ============================================================
    # 5. 训练循环
    # ============================================================
    if config.test_only:
        print("\n测试模式，跳过训练")
        return
    
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, None, optimizer, epoch, config
        )
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss = validate(model, val_loader, config)
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存模型
        if epoch % config.save_interval == 0:
            save_path = os.path.join(config.output_dir, f'checkpoint_epoch{epoch}.pkl')
            state_dict = {k: v.numpy() for k, v in model.state_dict().items()}
            with open(save_path, 'wb') as f:
                pickle.dump(state_dict, f)
            print(f"保存检查点: {save_path}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(config.output_dir, 'best_model.pkl')
            state_dict = {k: v.numpy() for k, v in model.state_dict().items()}
            with open(best_path, 'wb') as f:
                pickle.dump(state_dict, f)
            print(f"保存最佳模型: {best_path} (loss: {best_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"模型保存在: {config.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()




