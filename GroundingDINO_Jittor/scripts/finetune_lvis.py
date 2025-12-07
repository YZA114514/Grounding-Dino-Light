#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - LVIS 微调脚本

在 LVIS 数据集上微调 Grounding DINO 模型。
冻结 backbone 和 text encoder，只训练 transformer 和检测头。

用法:
    python scripts/finetune_lvis.py \
        --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
        --lvis_train data/lvis/lvis_v1_train.json \
        --lvis_val data/lvis/lvis_v1_val.json \
        --image_dir data/coco \
        --output_dir outputs/lvis_finetune \
        --epochs 12 \
        --batch_size 4 \
        --lr 1e-4
"""

import os
import sys
import json
import argparse
import pickle
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import jittor as jt
    from jittor import nn
    from jittor.dataset import Dataset
except ImportError:
    print("错误: 请安装 Jittor: pip install jittor")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================

class TrainConfig:
    """训练配置"""
    def __init__(self):
        # 模型参数
        self.num_queries = 900
        self.hidden_dim = 256
        self.num_feature_levels = 4
        self.nheads = 8
        self.max_text_len = 256
        
        # 训练参数
        self.epochs = 12
        self.batch_size = 4
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.weight_decay = 0.0001
        self.clip_max_norm = 0.1
        
        # 冻结设置
        self.freeze_backbone = True
        self.freeze_text_encoder = True
        
        # 损失系数
        self.cls_loss_coef = 2.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        
        # 数据参数
        self.image_size = 800
        self.max_size = 1333
        
        # 日志参数
        self.log_interval = 50
        self.save_interval = 1
        self.eval_interval = 1


# ============================================================
# LVIS 训练数据集
# ============================================================

class LVISTrainDataset(Dataset):
    """LVIS 训练数据集"""
    
    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        config: TrainConfig,
        max_objects: int = 100,
    ):
        super().__init__()
        
        self.image_dir = image_dir
        self.config = config
        self.max_objects = max_objects
        
        print(f"Loading LVIS training data from {ann_file}...")
        with open(ann_file, 'r') as f:
            lvis_data = json.load(f)
        
        self.images = {img['id']: img for img in lvis_data['images']}
        self.categories = {cat['id']: cat for cat in lvis_data['categories']}
        
        # 按图像分组标注
        self.img_to_anns = defaultdict(list)
        for ann in lvis_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # 只保留有标注的图像
        self.image_ids = [img_id for img_id in self.images.keys() 
                        if len(self.img_to_anns[img_id]) > 0]
        
        # 类别映射
        self.cat_ids = sorted(self.categories.keys())
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(self.cat_ids)}
        self.category_names = [self.categories[cid]['name'] for cid in self.cat_ids]
        
        self.total_len = len(self.image_ids)
        
        print(f"  Training images: {self.total_len}")
        print(f"  Categories: {len(self.categories)}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        # 加载图像
        # LVIS 图像在 COCO 的 train2017 或 val2017 目录中
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # 尝试不同的图像路径
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, 'train2017', img_info['file_name'])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, 'val2017', img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 缩放图像
        scale = self.config.image_size / min(orig_h, orig_w)
        if max(orig_h, orig_w) * scale > self.config.max_size:
            scale = self.config.max_size / max(orig_h, orig_w)
        
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # 转换为张量
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = img_array.transpose(2, 0, 1)
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # 获取标注
        anns = self.img_to_anns[image_id]
        
        boxes = []
        labels = []
        cat_ids_in_image = []
        
        for ann in anns[:self.max_objects]:
            x, y, w, h = ann['bbox']
            
            # 缩放边界框
            x = x * scale
            y = y * scale
            w = w * scale
            h = h * scale
            
            # 转换为归一化的 [cx, cy, w, h]
            cx = (x + w / 2) / new_w
            cy = (y + h / 2) / new_h
            nw = w / new_w
            nh = h / new_h
            
            if nw > 0 and nh > 0:
                boxes.append([cx, cy, nw, nh])
                cat_id = ann['category_id']
                labels.append(self.cat_id_to_idx[cat_id])
                cat_ids_in_image.append(cat_id)
        
        # 构建文本 prompt
        unique_cats = list(set(cat_ids_in_image))
        cat_names = [self.categories[cid]['name'] for cid in unique_cats]
        caption = '. '.join(cat_names) + '.'
        
        # 转换为数组
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        target = {
            'image_id': image_id,
            'boxes': boxes,
            'labels': labels,
            'caption': caption,
            'orig_size': np.array([orig_h, orig_w]),
            'size': np.array([new_h, new_w]),
        }
        
        return img_tensor, target
    
    def collate_fn(self, batch):
        """批次整理函数"""
        images = [jt.array(item[0]) for item in batch]
        targets = [item[1] for item in batch]
        
        # 填充图像到相同尺寸
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        padded_images = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            if pad_h > 0 or pad_w > 0:
                padded = jt.zeros((3, max_h, max_w))
                padded[:, :img.shape[1], :img.shape[2]] = img
                padded_images.append(padded)
            else:
                padded_images.append(img)
        
        batched_images = jt.stack(padded_images, dim=0)
        
        # 转换 target 中的数组为 Jittor 张量
        for t in targets:
            t['boxes'] = jt.array(t['boxes'])
            t['labels'] = jt.array(t['labels'])
        
        return batched_images, targets


# ============================================================
# 损失函数
# ============================================================

def compute_loss(outputs: Dict, targets: List[Dict], config: TrainConfig) -> Dict:
    """
    计算损失
    
    Args:
        outputs: 模型输出 {'pred_logits': [B, num_queries, max_text_len], 
                          'pred_boxes': [B, num_queries, 4]}
        targets: 目标列表
        config: 配置
        
    Returns:
        损失字典
    """
    pred_logits = outputs['pred_logits']  # [B, num_queries, max_text_len]
    pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
    
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]
    
    total_cls_loss = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    num_boxes = 0
    
    for b in range(batch_size):
        gt_boxes = targets[b]['boxes']
        gt_labels = targets[b]['labels']
        
        if len(gt_boxes) == 0:
            continue
        
        # 简化的匹配：使用 IoU 最大的 query 匹配每个 GT
        pred_box = pred_boxes[b]  # [num_queries, 4]
        
        # 计算 IoU 矩阵
        # pred_box: [num_queries, 4] (cx, cy, w, h)
        # gt_boxes: [num_gt, 4] (cx, cy, w, h)
        
        # 转换为 xyxy 格式
        pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
        pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
        pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
        pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2
        
        gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
        gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
        gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
        gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
        
        # 计算交集
        inter_x1 = jt.maximum(pred_x1.unsqueeze(1), gt_x1.unsqueeze(0))
        inter_y1 = jt.maximum(pred_y1.unsqueeze(1), gt_y1.unsqueeze(0))
        inter_x2 = jt.minimum(pred_x2.unsqueeze(1), gt_x2.unsqueeze(0))
        inter_y2 = jt.minimum(pred_y2.unsqueeze(1), gt_y2.unsqueeze(0))
        
        inter_w = jt.clamp(inter_x2 - inter_x1, min_v=0)
        inter_h = jt.clamp(inter_y2 - inter_y1, min_v=0)
        inter_area = inter_w * inter_h
        
        # 计算并集
        pred_area = pred_box[:, 2] * pred_box[:, 3]
        gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
        union_area = pred_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
        
        iou = inter_area / (union_area + 1e-6)  # [num_queries, num_gt]
        
        # 贪婪匹配
        matched_pred_idx = []
        matched_gt_idx = []
        
        for gt_idx in range(len(gt_boxes)):
            # 找到与该 GT 最匹配的 query
            ious_for_gt = iou[:, gt_idx]
            best_pred_idx = jt.argmax(ious_for_gt, dim=0)[0].item()
            matched_pred_idx.append(best_pred_idx)
            matched_gt_idx.append(gt_idx)
        
        if len(matched_pred_idx) == 0:
            continue
        
        # 计算匹配对的损失
        for pred_idx, gt_idx in zip(matched_pred_idx, matched_gt_idx):
            # 边界框 L1 损失
            bbox_loss = jt.abs(pred_box[pred_idx] - gt_boxes[gt_idx]).sum()
            total_bbox_loss += bbox_loss
            
            # GIoU 损失
            giou = iou[pred_idx, gt_idx]
            giou_loss = 1 - giou
            total_giou_loss += giou_loss
            
            # 分类损失 (简化版)
            pred_logit = pred_logits[b, pred_idx]  # [max_text_len]
            target_label = gt_labels[gt_idx].item()
            
            if target_label < pred_logit.shape[0]:
                # Focal loss 风格的分类损失
                prob = jt.sigmoid(pred_logit[target_label])
                cls_loss = -jt.log(prob + 1e-6)
                total_cls_loss += cls_loss
            
            num_boxes += 1
    
    # 归一化
    num_boxes = max(num_boxes, 1)
    
    losses = {
        'loss_cls': total_cls_loss / num_boxes * config.cls_loss_coef,
        'loss_bbox': total_bbox_loss / num_boxes * config.bbox_loss_coef,
        'loss_giou': total_giou_loss / num_boxes * config.giou_loss_coef,
    }
    
    losses['loss_total'] = losses['loss_cls'] + losses['loss_bbox'] + losses['loss_giou']
    
    return losses


# ============================================================
# 工具函数
# ============================================================

def freeze_module(module: nn.Module, name: str = ""):
    """冻结模块"""
    count = 0
    for param in module.parameters():
        param.stop_grad()
        count += 1
    if name:
        print(f"  Frozen {name}: {count} parameters")
    return count


def count_parameters(model: nn.Module):
    """统计参数"""
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
    print(f"Loading pretrained weights from {weight_path}...")
    
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    cleaned = {}
    for k, v in weights.items():
        if k.startswith('module.'):
            k = k[7:]
        cleaned[k] = v
    
    model_state = model.state_dict()
    loaded = 0
    
    for k, v in cleaned.items():
        if k in model_state and model_state[k].shape == tuple(v.shape):
            model_state[k] = jt.array(v)
            loaded += 1
    
    model.load_state_dict(model_state)
    print(f"Loaded {loaded}/{len(cleaned)} weights")
    
    return loaded


def save_checkpoint(model: nn.Module, optimizer, epoch: int, 
                   metrics: Dict, output_dir: str, name: str = None):
    """保存检查点"""
    if name is None:
        name = f'checkpoint_epoch{epoch}.pkl'
    
    save_path = os.path.join(output_dir, name)
    
    state_dict = {k: v.numpy() for k, v in model.state_dict().items()}
    
    checkpoint = {
        'epoch': epoch,
        'model': state_dict,
        'metrics': metrics,
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint to {save_path}")
    
    return save_path


# ============================================================
# 训练和评估函数
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    epoch: int,
    config: TrainConfig
) -> Dict:
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 构建文本输入
        captions = [t['caption'] for t in targets]
        
        # 前向传播
        outputs = model(images, captions=captions)
        
        # 计算损失
        losses = compute_loss(outputs, targets, config)
        
        # 反向传播
        optimizer.step(losses['loss_total'])
        
        # 累积损失
        total_loss += losses['loss_total'].item()
        total_cls_loss += losses['loss_cls'].item()
        total_bbox_loss += losses['loss_bbox'].item()
        total_giou_loss += losses['loss_giou'].item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses['loss_total'].item():.4f}",
            'cls': f"{losses['loss_cls'].item():.4f}",
            'bbox': f"{losses['loss_bbox'].item():.4f}",
        })
        
        # 打印日志
        if (batch_idx + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {losses['loss_total'].item():.4f} "
                  f"(cls: {losses['loss_cls'].item():.4f}, "
                  f"bbox: {losses['loss_bbox'].item():.4f}, "
                  f"giou: {losses['loss_giou'].item():.4f}) "
                  f"Time: {elapsed:.1f}s")
    
    # 计算平均损失
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_bbox_loss = total_bbox_loss / num_batches
    avg_giou_loss = total_giou_loss / num_batches
    
    return {
        'loss': avg_loss,
        'loss_cls': avg_cls_loss,
        'loss_bbox': avg_bbox_loss,
        'loss_giou': avg_giou_loss,
    }


def evaluate(
    model: nn.Module,
    val_loader,
    config: TrainConfig
) -> Dict:
    """验证"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with jt.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            captions = [t['caption'] for t in targets]
            
            outputs = model(images, captions=captions)
            losses = compute_loss(outputs, targets, config)
            
            total_loss += losses['loss_total'].item()
            num_batches += 1
    
    return {'val_loss': total_loss / max(num_batches, 1)}


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LVIS Fine-tuning')
    
    # 数据路径
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Pretrained checkpoint path')
    parser.add_argument('--lvis_train', type=str, required=True,
                        help='LVIS training annotation file')
    parser.add_argument('--lvis_val', type=str, required=True,
                        help='LVIS validation annotation file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image directory (COCO images)')
    parser.add_argument('--output_dir', type=str, default='outputs/lvis_finetune',
                        help='Output directory')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Backbone LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    
    # 冻结设置
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True,
                        help='Freeze text encoder')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                        help='Unfreeze backbone (overrides freeze_backbone)')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    parser.add_argument('--eval_interval', type=int, default=1, help='Eval interval')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU')
    
    args = parser.parse_args()
    
    # 设置 Jittor
    if not args.no_gpu:
        jt.flags.use_cuda = 1
        print("Using GPU")
    else:
        print("Using CPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置
    config = TrainConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.lr_backbone = args.lr_backbone
    config.weight_decay = args.weight_decay
    config.freeze_backbone = not args.unfreeze_backbone
    config.freeze_text_encoder = not args.unfreeze_text_encoder
    config.log_interval = args.log_interval
    config.save_interval = args.save_interval
    config.eval_interval = args.eval_interval
    
    print("=" * 60)
    print("LVIS Fine-tuning")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Freeze backbone: {config.freeze_backbone}")
    print(f"Freeze text encoder: {config.freeze_text_encoder}")
    
    # ============================================================
    # 1. 创建模型
    # ============================================================
    print("\n[1/5] Creating model...")
    
    from jittor_implementation.models.groundingdino import GroundingDINO
    
    model = GroundingDINO(
        num_queries=config.num_queries,
        hidden_dim=config.hidden_dim,
        num_feature_levels=config.num_feature_levels,
        nheads=config.nheads,
        max_text_len=config.max_text_len,
    )
    
    # ============================================================
    # 2. 加载预训练权重
    # ============================================================
    print("\n[2/5] Loading pretrained weights...")
    load_pretrained_weights(model, args.checkpoint)
    
    # ============================================================
    # 3. 冻结参数
    # ============================================================
    print("\n[3/5] Configuring parameter freezing...")
    
    if config.freeze_backbone:
        for name, module in model.named_modules():
            if 'backbone' in name or 'input_proj' in name:
                freeze_module(module, name)
    
    if config.freeze_text_encoder:
        for name, module in model.named_modules():
            if 'bert' in name.lower() or 'text_encoder' in name or 'feat_map' in name:
                freeze_module(module, name)
    
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameter statistics:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # ============================================================
    # 4. 创建数据集和优化器
    # ============================================================
    print("\n[4/5] Creating datasets and optimizer...")
    
    train_dataset = LVISTrainDataset(args.lvis_train, args.image_dir, config)
    val_dataset = LVISTrainDataset(args.lvis_val, args.image_dir, config)
    
    train_loader = train_dataset.set_attrs(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    val_loader = val_dataset.set_attrs(
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # 优化器
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = jt.optim.AdamW(
        trainable_params_list,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    lr_scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=config.epochs // 2, gamma=0.1)
    
    # ============================================================
    # 5. 训练循环
    # ============================================================
    print("\n[5/5] Starting training...")
    print("=" * 60)
    
    best_loss = float('inf')
    start_epoch = 1
    
    # 恢复训练
    if args.resume:
        print(f"Resuming from {args.resume}")
        with open(args.resume, 'rb') as f:
            checkpoint = pickle.load(f)
        model.load_state_dict({k: jt.array(v) for k, v in checkpoint['model'].items()})
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # 保存配置
    config_file = os.path.join(args.output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 训练日志
    log_file = os.path.join(args.output_dir, 'train_log.txt')
    
    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("-" * 40)
        
        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, epoch, config)
        
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(cls: {train_metrics['loss_cls']:.4f}, "
              f"bbox: {train_metrics['loss_bbox']:.4f}, "
              f"giou: {train_metrics['loss_giou']:.4f})")
        
        # 验证
        if epoch % config.eval_interval == 0:
            val_metrics = evaluate(model, val_loader, config)
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        else:
            val_metrics = {}
        
        # 更新学习率
        lr_scheduler.step()
        current_lr = optimizer.lr
        print(f"Learning rate: {current_lr}")
        
        # 保存检查点
        if epoch % config.save_interval == 0:
            metrics = {**train_metrics, **val_metrics}
            save_checkpoint(model, optimizer, epoch, metrics, args.output_dir)
        
        # 保存最佳模型
        current_loss = val_metrics.get('val_loss', train_metrics['loss'])
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(model, optimizer, epoch, 
                          {**train_metrics, **val_metrics}, 
                          args.output_dir, 'best_model.pkl')
            print(f"New best model! Loss: {best_loss:.4f}")
        
        # 写入日志
        with open(log_file, 'a') as f:
            log_entry = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
            if val_metrics:
                log_entry += f", val_loss={val_metrics['val_loss']:.4f}"
            log_entry += f", lr={current_lr}\n"
            f.write(log_entry)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
