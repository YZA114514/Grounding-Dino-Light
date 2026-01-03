#!/usr/bin/env python3
"""
Grounding DINO LVIS Fine-tuning Script v2

Improved version with cached dataset loading, proper loss functions,
gradient accumulation, and optimized training loop.

Usage:
    # Quick test
    python scripts/finetune_lvis_v2.py --test_only --num_samples 100

    # Full training
    python scripts/finetune_lvis_v2.py \
        --epochs 24 \
        --batch_size 4 \
        --lr 1e-4 \
        --output_dir outputs/finetune_lvis_v2
"""

import os
import sys
import json
import pickle
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Set GPU before importing jittor
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add GroundingDINO_Jittor to path for imports
sys.path.insert(0, os.path.join(BASE_DIR, 'GroundingDINO_Jittor'))
sys.path.insert(0, os.path.join(BASE_DIR, 'GroundingDINO_Jittor', 'jittor_implementation'))
sys.path.insert(0, BASE_DIR)

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import math

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset

jt.flags.use_cuda = 1
jt.flags.log_silent = 1  # Suppress [w] warnings
jt.flags.use_cuda_managed_allocator = 1  # 更激进的内存回收


# ============================================================
# Configuration
# ============================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning"""
    # Model
    hidden_dim = 256
    num_queries = 900
    max_text_len = 256

    # Data
    image_size = 640

    # Loss weights (matching costs vs final loss weights are different!)
    matching_cost_class = 2.0   # Hungarian matching costs
    matching_cost_bbox = 5.0
    matching_cost_giou = 2.0

    final_loss_class = 1.0      # Final loss weights (different from matching!)
    final_loss_bbox = 5.0
    final_loss_giou = 2.0

    # Training
    batch_size = 4
    gradient_accumulation = 4  # effective batch = batch_size * gradient_accumulation
    epochs = 24
    lr = 1e-4
    lr_backbone = 1e-5
    weight_decay = 1e-4
    clip_grad_norm = 0.1

    # Learning rate schedule
    warmup_epochs = 1

    # Freeze settings
    freeze_backbone = False
    freeze_text_encoder = True

    # Checkpointing
    save_interval = 4  # Save every N epochs
    sync_interval = 20  # Sync every N batches

    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation


def parse_args():
    parser = argparse.ArgumentParser(description='Grounding DINO LVIS Fine-tuning v2')

    # Paths
    parser.add_argument('--checkpoint', type=str,
                        default='GroundingDINO_Jittor/weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--cache_dir', type=str,
                        default='data/lvis_finetune_preload_cache_square',
                        help='Path to cached LVIS dataset')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/finetune_lvis_v2',
                        help='Output directory')

    # Training
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_norm', type=float, default=0.1)

    # Freeze settings
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True)

    # Multi-GPU
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    # Testing
    parser.add_argument('--test_only', action='store_true',
                        help='Only run quick test')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for quick test')

    return parser.parse_args()


# ============================================================
# Cached Dataset
# ============================================================

def build_positive_map(cat_names: List[str], tokenizer) -> Dict[int, List[int]]:
    """
    Map box indices to token indices where category names appear in caption.

    Args:
        cat_names: List of category names for this image's boxes
        tokenizer: HuggingFace BertTokenizer

    Returns:
        positive_map: {box_idx: [token_indices]}
    """
    # Build caption: "cat. dog. person."
    caption = '. '.join(cat_names) + '.'

    # Tokenize full caption
    encoding = tokenizer(caption, return_offsets_mapping=True)
    tokens = encoding['input_ids']
    offsets = encoding['offset_mapping']

    positive_map = {}
    char_pos = 0

    for box_idx, cat_name in enumerate(cat_names):
        cat_start = char_pos
        cat_end = char_pos + len(cat_name)

        # Find tokens overlapping with this category's character span
        token_indices = []
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start >= cat_start and tok_end <= cat_end:
                token_indices.append(tok_idx)

        positive_map[box_idx] = token_indices
        char_pos = cat_end + 2  # Skip ". "

    return positive_map


class CachedLVISDataset(Dataset):
    """Dataset that loads preprocessed LVIS data from cache"""

    def __init__(self, cache_dir: str, max_samples: Optional[int] = None):
        super().__init__()

        self.cache_dir = cache_dir

        # Initialize tokenizer for positive_map computation
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Load index
        index_path = os.path.join(cache_dir, 'index.pkl')
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)

        # Load categories
        categories_path = os.path.join(cache_dir, 'categories.pkl')
        with open(categories_path, 'rb') as f:
            self.categories = pickle.load(f)

        # Limit samples if specified
        if max_samples:
            self.index = self.index[:max_samples]

        self.num_samples = len(self.index)
        print(f"Loaded {self.num_samples} samples from cache")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[jt.Var, Dict]:
        """Load sample from cache"""
        entry = self.index[idx]

        # Load npz file
        npz_path = os.path.join(self.cache_dir, f"{entry['id']}.npz")
        data = np.load(npz_path)

        # Convert fp16 to fp32 for training
        image = data['image'].astype(np.float32)  # (3, H, W) float32
        boxes = data['boxes']  # (N, 4) float32 normalized cxcywh
        labels = data['labels']  # (N,) int32
        orig_size = data['orig_size']  # (2,) int32 [H, W]
        new_size = data['new_size']  # (2,) int32 [H, W]

        # Build caption from category names
        cat_names = entry['cat_names']
        caption = '. '.join(cat_names) + '.'

        # Compute positive_map on-the-fly
        positive_map = build_positive_map(cat_names, self.tokenizer)

        # Convert to Jittor tensors
        image_tensor = jt.array(image)
        boxes_tensor = jt.array(boxes)
        labels_tensor = jt.array(labels.astype(np.int64))

        target = {
            'image_id': entry['id'],
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'orig_size': orig_size,
            'new_size': new_size,
            'caption': caption,
            'cat_names': cat_names,
            'positive_map': positive_map,
        }

        return image_tensor, target


# ============================================================
# Loss Functions
# ============================================================

def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between two sets of boxes

    Args:
        boxes1: (N, 4) tensor in cxcywh format
        boxes2: (M, 4) tensor in cxcywh format

    Returns:
        giou: (N, M) tensor
    """
    # Convert to xyxy
    cx1, cy1, w1, h1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    cx2, cy2, w2, h2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    x1 = cx1 - w1 / 2
    y1 = cy1 - h1 / 2
    x2 = cx1 + w1 / 2
    y2 = cy1 + h1 / 2

    x3 = cx2 - w2 / 2
    y3 = cy2 - h2 / 2
    x4 = cx2 + w2 / 2
    y4 = cy2 + h2 / 2

    # Intersection
    inter_x1 = jt.maximum(x1[:, None], x3[None, :])
    inter_y1 = jt.maximum(y1[:, None], y3[None, :])
    inter_x2 = jt.minimum(x2[:, None], x4[None, :])
    inter_y2 = jt.minimum(y2[:, None], y4[None, :])

    inter_w = jt.clamp(inter_x2 - inter_x1, min_v=0)
    inter_h = jt.clamp(inter_y2 - inter_y1, min_v=0)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1[:, None] * h1[:, None]
    area2 = w2[None, :] * h2[None, :]
    union_area = area1 + area2 - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-6)

    # Enclosing box
    enc_x1 = jt.minimum(x1[:, None], x3[None, :])
    enc_y1 = jt.minimum(y1[:, None], y3[None, :])
    enc_x2 = jt.maximum(x2[:, None], x4[None, :])
    enc_y2 = jt.maximum(y2[:, None], y4[None, :])

    enc_w = enc_x2 - enc_x1
    enc_h = enc_y2 - enc_y1
    enc_area = enc_w * enc_h

    # GIoU
    giou = iou - (enc_area - union_area) / (enc_area + 1e-6)

    return giou


class HungarianMatcher:
    """Hungarian matcher with proper cost computation"""

    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of target dicts

        Returns:
            list of (pred_indices, target_indices) tuples
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]

        indices = []
        for i in range(bs):
            tgt_boxes = targets[i]['boxes']
            num_targets = tgt_boxes.shape[0]

            if num_targets == 0:
                indices.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                continue

            # Get predictions for this batch
            pred_logits = outputs['pred_logits'][i]  # (num_queries, max_text_len)
            pred_boxes = outputs['pred_boxes'][i]    # (num_queries, 4)

            # Classification cost: use sigmoid and max probability
            pred_probs = jt.sigmoid(pred_logits)  # (num_queries, max_text_len)
            class_cost = -pred_probs.max(dim=-1)[0].numpy()  # (num_queries,)

            # L1 bbox cost
            bbox_cost = jt.abs(pred_boxes[:, None] - tgt_boxes[None, :]).sum(dim=-1).numpy()

            # GIoU cost
            giou_cost = -generalized_box_iou(pred_boxes, tgt_boxes).numpy()

            # Total cost matrix
            C = (self.cost_class * class_cost[:, None] +
                 self.cost_bbox * bbox_cost +
                 self.cost_giou * giou_cost)

            # Hungarian matching
            pred_idx, tgt_idx = linear_sum_assignment(C)

            indices.append((pred_idx, tgt_idx))

        return indices


class SetCriterion(nn.Module):
    """DETR-style criterion with proper loss computation"""

    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def loss_labels(self, outputs, targets, indices):
        """Classification loss with positive_map"""
        pred_logits = outputs['pred_logits']  # (bs, num_queries, max_text_len)
        bs, num_queries, num_tokens = pred_logits.shape

        # Create target tensor: (bs, num_queries, max_text_len)
        # Default all zeros (background)
        target_map = jt.zeros_like(pred_logits)

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            # Build positive_map at runtime using cat_names order
            cat_names = targets[batch_idx]['cat_names']

            # Build position mapping: category → position in caption
            cat_to_pos = {}
            for pos, cat_name in enumerate(cat_names):
                cat_to_pos[cat_name] = pos

            # Build positive_map: box_idx → list of [caption_positions]
            box_positive_map = {}
            for box_idx, cat_name in enumerate(cat_names):
                if cat_name in cat_to_pos:
                    box_positive_map[box_idx] = [cat_to_pos[cat_name]]

            for p_idx, t_idx in zip(pred_idx, tgt_idx):
                # Get caption positions for this target box
                caption_positions = box_positive_map.get(int(t_idx.item()), [])

                # Set those specific caption positions to 1.0
                for pos in caption_positions:
                    if pos < num_tokens:
                        target_map[batch_idx, p_idx, pos] = 1.0

        # Binary cross entropy loss (manual implementation for Jittor)
        pred_probs = jt.sigmoid(pred_logits)
        # Manual BCE: -[y*log(p) + (1-y)*log(1-p)]
        loss = -jt.mean(
            target_map * jt.log(pred_probs + 1e-8) +
            (1 - target_map) * jt.log(1 - pred_probs + 1e-8)
        )

        return {'loss_ce': loss}

    def loss_boxes(self, outputs, targets, indices):
        """L1 box regression loss"""
        pred_boxes = outputs['pred_boxes']

        src_boxes = []
        target_boxes = []

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                src_boxes.append(pred_boxes[i, pred_idx])
                target_boxes.append(targets[i]['boxes'][tgt_idx])

        if len(src_boxes) == 0:
            return {'loss_bbox': jt.array(0.0)}

        src_boxes = jt.concat(src_boxes, dim=0)
        target_boxes = jt.concat(target_boxes, dim=0)

        loss_bbox = jt.abs(src_boxes - target_boxes).mean()

        return {'loss_bbox': loss_bbox}

    def loss_giou(self, outputs, targets, indices):
        """GIoU loss"""
        pred_boxes = outputs['pred_boxes']

        src_boxes = []
        target_boxes = []

        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                src_boxes.append(pred_boxes[i, pred_idx])
                target_boxes.append(targets[i]['boxes'][tgt_idx])

        if len(src_boxes) == 0:
            return {'loss_giou': jt.array(0.0)}

        src_boxes = jt.concat(src_boxes, dim=0)
        target_boxes = jt.concat(target_boxes, dim=0)

        giou = generalized_box_iou(src_boxes, target_boxes).diag()
        loss_giou = (1 - giou).mean()

        return {'loss_giou': loss_giou}

    def execute(self, outputs, targets):
        """Compute all losses"""
        # Get matches
        indices = self.matcher(outputs, targets)

        # Compute losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_giou(outputs, targets, indices))

        # Apply weights
        weighted_losses = {}
        for k, v in losses.items():
            if k in self.weight_dict:
                weighted_losses[k] = v * self.weight_dict[k]
            else:
                weighted_losses[k] = v

        return weighted_losses


# ============================================================
# Learning Rate Scheduler
# ============================================================

class CosineLRScheduler:
    """Cosine learning rate scheduler with warmup"""

    def __init__(self, optimizers, warmup_epochs, total_epochs, base_lr, backbone_lr):
        self.optimizers = optimizers  # List of optimizers
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.backbone_lr = backbone_lr

    def step(self, epoch):
        """Update learning rate for given epoch"""
        if epoch <= self.warmup_epochs:
            # Gradual warmup: epoch 1 → 50% LR, epoch 2 (if warmup=2) → 100%
            lr = self.base_lr * epoch / (self.warmup_epochs + 1)
            backbone_lr = self.backbone_lr * epoch / (self.warmup_epochs + 1)
        else:
            # Cosine decay with minimum floor
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = max(self.base_lr * 0.5 * (1 + math.cos(math.pi * progress)), 1e-6)
            backbone_lr = max(self.backbone_lr * 0.5 * (1 + math.cos(math.pi * progress)), 1e-7)

        # Update optimizers
        self.optimizers[0].lr = lr  # optimizer_other
        self.optimizers[1].lr = backbone_lr  # optimizer_backbone

        return lr


# ============================================================
# Model Utilities
# ============================================================

def map_weight_name(name):
    """映射权重名称"""
    # 移除 module. 前缀
    if name.startswith('module.'):
        name = name[7:]

    # backbone.0. -> backbone.
    if name.startswith('backbone.0.'):
        name = 'backbone.' + name[11:]

    # transformer.level_embed -> level_embed
    if name == 'transformer.level_embed':
        name = 'level_embed'

    # transformer.tgt_embed -> tgt_embed
    if name.startswith('transformer.tgt_embed'):
        name = name.replace('transformer.tgt_embed', 'tgt_embed')

    # transformer.enc_output -> enc_output (注意：不是 enc_out_bbox_embed)
    # enc_output 是独立的，enc_out_bbox_embed 保留在 transformer 下
    if name == 'transformer.enc_output.weight' or name == 'transformer.enc_output.bias':
        name = name.replace('transformer.enc_output', 'enc_output')
    if name == 'transformer.enc_output_norm.weight' or name == 'transformer.enc_output_norm.bias':
        name = name.replace('transformer.enc_output_norm', 'enc_output_norm')

    # transformer.enc_out_bbox_embed 保持不变，因为模型中也在 transformer. 下
    # transformer.enc_out_class_embed 保持不变

    # bbox_embed.X. -> transformer.decoder.bbox_embed.X. (顶层独立 bbox_embed)
    if name.startswith('bbox_embed.') and not name.startswith('bbox_embed.layers'):
        import re
        match = re.match(r'bbox_embed\.(\d+)\.(.*)', name)
        if match:
            layer_idx, rest = match.groups()
            name = f'transformer.decoder.bbox_embed.{layer_idx}.{rest}'

    return name


def split_in_proj(weights):
    """拆分 in_proj_weight/bias 为 q/k/v_proj"""
    result = {}
    for k, v in weights.items():
        if '.in_proj_weight' in k:
            d = v.shape[0] // 3
            base = k.replace('.in_proj_weight', '.')
            result[base + 'q_proj.weight'] = v[:d, :]
            result[base + 'k_proj.weight'] = v[d:2*d, :]
            result[base + 'v_proj.weight'] = v[2*d:, :]
        elif '.in_proj_bias' in k:
            d = v.shape[0] // 3
            base = k.replace('.in_proj_bias', '.')
            result[base + 'q_proj.bias'] = v[:d]
            result[base + 'k_proj.bias'] = v[d:2*d]
            result[base + 'v_proj.bias'] = v[2*d:]
        else:
            result[k] = v
    return result


def load_model(checkpoint_path: str):
    """Load Grounding DINO model"""
    print(f"Loading model from {checkpoint_path}...")

    from jittor_implementation.models.groundingdino import GroundingDINO
    from jittor_implementation.models.backbone.swin_transformer import build_swin_transformer

    backbone = build_swin_transformer(
        modelname="swin_T_224_1k",
        pretrain_img_size=224,
        out_indices=(1, 2, 3),
        dilation=False,
    )

    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
        two_stage_type="standard",
        dec_pred_bbox_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=False,
    )

    with open(checkpoint_path, 'rb') as f:
        weights = pickle.load(f)

    # 分离 BERT 权重和其他权重
    bert_weights = {}
    other_weights = {}
    for k, v in weights.items():
        clean_k = k[7:] if k.startswith('module.') else k
        if clean_k.startswith('bert.'):
            bert_weights[clean_k] = v
        else:
            other_weights[k] = v

    # 映射权重名称
    mapped = {}
    for k, v in other_weights.items():
        new_k = map_weight_name(k)
        mapped[new_k] = v

    # 拆分 in_proj
    mapped = split_in_proj(mapped)

    # 加载权重
    model_state = model.state_dict()
    loaded = 0
    missing = []

    for k, v in mapped.items():
        if k in model_state:
            if model_state[k].shape == tuple(v.shape):
                model_state[k] = jt.array(v)
                loaded += 1
            else:
                print(f"  Shape mismatch: {k}: model {model_state[k].shape} vs ckpt {v.shape}")
        else:
            missing.append(k)

    model.load_state_dict(model_state)

    # 加载 BERT 权重
    bert_loaded = 0
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
        bert_module = model.text_encoder.bert
        is_huggingface = hasattr(bert_module, 'config') and hasattr(bert_module.config, 'model_type')

        if is_huggingface:
            # HuggingFace BERT - 需要从 checkpoint 加载训练后的权重
            import torch
            bert_state = bert_module.state_dict()

            for k, v in bert_weights.items():
                # 移除 'bert.' 前缀
                bert_key = k[5:] if k.startswith('bert.') else k
                if bert_key in bert_state:
                    if bert_state[bert_key].shape == tuple(v.shape):
                        # 转换为 PyTorch tensor
                        bert_state[bert_key] = torch.from_numpy(v)
                        bert_loaded += 1

            # 加载到 HuggingFace BERT
            bert_module.load_state_dict(bert_state)
            print(f"  Loaded checkpoint BERT weights into HuggingFace BERT")
        else:
            # Jittor BERT，加载权重
            bert_state = bert_module.state_dict()
            for k, v in bert_weights.items():
                bert_key = k[5:] if k.startswith('bert.') else k
                if bert_key in bert_state:
                    if bert_state[bert_key].shape == tuple(v.shape):
                        bert_state[bert_key] = jt.array(v)
                        bert_loaded += 1
            bert_module.load_state_dict(bert_state)

    print(f"  Loaded {loaded} model weights + {bert_loaded} BERT weights")
    if missing and len(missing) < 10:
        print(f"  Missing in model: {missing}")

    model.eval()

    # Enable gradients for training
    for param in model.parameters():
        param.start_grad()

    model.train()

    # 🔍 检查 execute 签名 (Jittor 使用 execute 而不是 forward)
    import inspect
    sig = inspect.signature(model.execute)
    print(f"\n{'='*50} Model execute signature: {sig}")
    print(f"{'='*50} Parameters: {list(sig.parameters.keys())}")
    print(f"{'='*50} Accepts 'mask': {'mask' in sig.parameters}")

    return model


def freeze_parameters(model, config):
    """Freeze specified parameters"""
    frozen_count = 0

    if config.freeze_backbone:
        for name, param in model.named_parameters():
            if 'backbone' in name or 'input_proj' in name:
                param.stop_grad()
                frozen_count += 1
        print(f"Froze {frozen_count} backbone parameters")

    if config.freeze_text_encoder and hasattr(model, 'text_encoder'):
        if hasattr(model.text_encoder, 'bert'):
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = False
            print("Froze BERT text encoder")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")


def wrap_model(model, num_gpus=1):
    """Wrap model for multi-GPU training"""
    if num_gpus > 1:
        # Jittor DataParallel equivalent
        model = jt.DataParallel(model, device_ids=list(range(num_gpus)))
    return model


def unwrap_model(model):
    """Unwrap model for checkpoint saving"""
    if hasattr(model, 'module'):
        return model.module
    return model


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, criterion, optimizers, lr_scheduler, dataset,
                    config, epoch, num_gpus=1):
    """Train for one epoch with gradient accumulation"""
    model.train()

    total_loss = 0
    total_loss_cls = 0
    total_loss_bbox = 0
    total_loss_giou = 0
    num_batches = 0

    accum_steps = config.gradient_accumulation
    for opt in optimizers:
        opt.zero_grad()

    # Create batches
    batch_size = config.batch_size
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)

    pbar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch}")
    for batch_idx, batch_start in enumerate(pbar):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = indices[batch_start:batch_end]

        # Load batch data
        batch_images = []
        batch_targets = []

        for idx in batch_indices:
            image, target = dataset[idx]
            batch_images.append(image)
            batch_targets.append(target)

        # Skip if batch is empty
        if len(batch_images) == 0:
            continue

        # Stack images (no padding since model doesn't support masks)
        images = jt.stack(batch_images, dim=0)

        # Forward pass
        captions = [t['caption'] for t in batch_targets]
        outputs = model(images, captions=captions)

        # Compute loss
        losses = criterion(outputs, batch_targets)
        loss = sum(losses.values())

        # Scale loss for gradient accumulation
        loss = loss / accum_steps

        # Backward pass
        for opt in optimizers:
            opt.backward(loss)

        # Update weights every accum_steps
        if (batch_idx + 1) % accum_steps == 0:
            # Clip gradients manually (Jittor doesn't have utils.clip_grad_norm_)
            total_norm = 0.0
            for param in model.parameters():
                # Find which optimizer manages this parameter
                for opt in optimizers:
                    try:
                        grad = param.opt_grad(opt)
                        if grad is not None:
                            # Sum of squared elements, then get scalar with .sum().item()
                            total_norm += (grad ** 2).sum().item()
                        break
                    except RuntimeError:
                        continue  # Parameter not managed by this optimizer
            total_norm = total_norm ** 0.5

            clip_coef = config.clip_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for param in model.parameters():
                    # Find which optimizer manages this parameter
                    for opt in optimizers:
                        try:
                            grad = param.opt_grad(opt)
                            if grad is not None:
                                grad.mul_(clip_coef)
                            break
                        except RuntimeError:
                            continue  # Parameter not managed by this optimizer

            # Optimizer step
            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            # Clear gradient graph to prevent memory accumulation
            jt.gc()

        # Record losses
        total_loss += loss.item() * accum_steps  # Scale back for logging
        if 'loss_ce' in losses:
            total_loss_cls += losses['loss_ce'].item()
        if 'loss_bbox' in losses:
            total_loss_bbox += losses['loss_bbox'].item()
        if 'loss_giou' in losses:
            total_loss_giou += losses['loss_giou'].item()
        num_batches += 1

        # Update progress bar
        avg_loss = total_loss / max(num_batches, 1)
        pbar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'cls': f"{total_loss_cls/max(num_batches,1):.4f}",
            'bbox': f"{total_loss_bbox/max(num_batches,1):.4f}",
            'lr': f"{optimizers[0].lr:.6f}"
        })

        # Periodic sync (reduced frequency)
        if batch_idx % config.sync_interval == 0:
            jt.sync_all()
            jt.gc()

    # Handle remaining gradients if any
    if (batch_idx + 1) % accum_steps != 0:
        # Clip gradients manually
        total_norm = 0.0
        for param in model.parameters():
            # Find which optimizer manages this parameter
            for opt in optimizers:
                try:
                    grad = param.opt_grad(opt)
                    if grad is not None:
                        # Sum of squared elements, then get scalar with .sum().item()
                        total_norm += (grad ** 2).sum().item()
                    break
                except RuntimeError:
                    continue  # Parameter not managed by this optimizer
        total_norm = total_norm ** 0.5

        clip_coef = config.clip_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                # Find which optimizer manages this parameter
                for opt in optimizers:
                    try:
                        grad = param.opt_grad(opt)
                        if grad is not None:
                            grad.mul_(clip_coef)
                        break
                    except RuntimeError:
                        continue  # Parameter not managed by this optimizer

        # Optimizer step
        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        # Clear gradient graph to prevent memory accumulation
        jt.gc()

    # Compute averages
    avg_losses = {
        'loss': total_loss / max(num_batches, 1),
        'loss_cls': total_loss_cls / max(num_batches, 1),
        'loss_bbox': total_loss_bbox / max(num_batches, 1),
        'loss_giou': total_loss_giou / max(num_batches, 1),
    }

    return avg_losses


def save_checkpoint(model, optimizer, epoch, loss, output_dir, name='checkpoint'):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap model for saving
    model = unwrap_model(model)

    # Convert to numpy for saving
    state_dict = {}
    for k, v in model.state_dict().items():
        if isinstance(v, jt.Var):
            state_dict[k] = v.numpy()
        else:
            state_dict[k] = v

    checkpoint = {
        'model_state_dict': state_dict,
        'epoch': epoch,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }

    # Save specific checkpoint
    save_path = os.path.join(output_dir, f'{name}_epoch{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    # Always save latest (overwrite)
    latest_path = os.path.join(output_dir, f'{name}_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Saved checkpoint: {save_path}")
    print(f"Saved latest: {latest_path}")

    return save_path


def save_best_checkpoint(model, optimizer, loss, output_dir):
    """Save best model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)

    model = unwrap_model(model)
    state_dict = {}
    for k, v in model.state_dict().items():
        if isinstance(v, jt.Var):
            state_dict[k] = v.numpy()
        else:
            state_dict[k] = v

    checkpoint = {
        'model_state_dict': state_dict,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }

    best_path = os.path.join(output_dir, 'checkpoint_best.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Saved best model: {best_path}")
    return best_path


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create config
    config = FinetuneConfig()
    config.batch_size = args.batch_size
    config.gradient_accumulation = args.gradient_accumulation
    config.epochs = args.epochs
    config.lr = args.lr
    config.lr_backbone = args.lr_backbone
    config.weight_decay = args.weight_decay
    config.clip_grad_norm = args.clip_grad_norm
    config.freeze_backbone = args.freeze_backbone
    config.freeze_text_encoder = args.freeze_text_encoder

    print("=" * 70)
    print("Grounding DINO LVIS Fine-tuning v2")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {config.batch_size}, Gradient accumulation: {config.gradient_accumulation}")
    print(f"Effective batch size: {config.effective_batch_size()}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}")

    # Load dataset
    print("\nLoading cached dataset...")
    if args.test_only:
        dataset = CachedLVISDataset(args.cache_dir, max_samples=args.num_samples)
    else:
        dataset = CachedLVISDataset(args.cache_dir, max_samples=args.num_samples if args.num_samples else None)

    dataloader = dataset
    print(f"Dataset size: {len(dataset)}")

    # Load model
    print("\nLoading model...")
    checkpoint_path = os.path.join(BASE_DIR, args.checkpoint)
    model = load_model(checkpoint_path)

    # Freeze parameters
    freeze_parameters(model, config)

    # Wrap model for multi-GPU
    model = wrap_model(model, args.gpus)

    # Create criterion (loss function)
    matcher = HungarianMatcher(
        cost_class=config.matching_cost_class,
        cost_bbox=config.matching_cost_bbox,
        cost_giou=config.matching_cost_giou
    )

    weight_dict = {
        'loss_ce': config.final_loss_class,
        'loss_bbox': config.final_loss_bbox,
        'loss_giou': config.final_loss_giou,
    }

    criterion = SetCriterion(
        num_classes=len(dataset.categories),
        matcher=matcher,
        weight_dict=weight_dict
    )

    # Create optimizer with separate param groups
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'input_proj' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # Jittor doesn't support parameter groups like PyTorch
    # Create separate optimizers for backbone and other params
    optimizer_backbone = jt.optim.AdamW(backbone_params, lr=config.lr_backbone, weight_decay=config.weight_decay)
    optimizer_other = jt.optim.AdamW(other_params, lr=config.lr, weight_decay=config.weight_decay)

    # Store optimizers in a list for easier handling
    optimizers = [optimizer_other, optimizer_backbone]

    print(f"Param groups: {len(other_params)} main ({config.lr}), {len(backbone_params)} backbone ({config.lr_backbone})")

    # Create LR scheduler
    lr_scheduler = CosineLRScheduler(optimizers, config.warmup_epochs, config.epochs, config.lr, config.lr_backbone)

    print(f"\nOptimizer: AdamW, LR: {config.lr}, Weight decay: {config.weight_decay}")
    print(f"Total trainable parameters: {len(other_params) + len(backbone_params)}")

    # Test mode
    if args.test_only:
        print("\n" + "=" * 50)
        print("TEST MODE - Quick validation")
        print("=" * 50)

        model.eval()
        with jt.no_grad():
            # Process a few individual samples
            test_losses = []
            for i in range(min(5, len(dataset))):  # Test up to 5 samples
                # Get single sample
                image, target = dataset[i]

                images = image.unsqueeze(0)  # Add batch dimension
                targets = [target]           # Wrap in list for criterion

                captions = [target['caption']]
                outputs = model(images, captions=captions)
                losses = criterion(outputs, targets)
                test_losses.append({k: v.item() for k, v in losses.items()})

            # Average test losses
            avg_test_loss = {}
            for k in test_losses[0].keys():
                avg_test_loss[k] = sum(t[k] for t in test_losses) / len(test_losses)

            print("Test completed successfully!")
            print(f"Average test loss: {sum(avg_test_loss.values()):.4f}")
            for k, v in avg_test_loss.items():
                print(f"  {k}: {v:.4f}")

        return

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    # Resume from checkpoint if available
    resume_path = os.path.join(args.output_dir, 'checkpoint_latest.pkl')
    start_epoch = 1
    best_loss = float('inf')

    if os.path.exists(resume_path) and not args.test_only:
        print(f"\n{'='*50} Found previous checkpoint at {resume_path} {'='*50}")

        # Load checkpoint info without resuming yet
        with open(resume_path, 'rb') as f:
            checkpoint = pickle.load(f)

        print(f"Last saved checkpoint settings:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss: {checkpoint['loss']:.4f}")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  Output directory: {args.output_dir}")

        # Ask user if they want to resume
        while True:
            try:
                response = input("\nResume from this checkpoint? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    print("Resuming from checkpoint...")

                    # Load model state
                    model_state = checkpoint['model_state_dict']
                    unwrapped_model = unwrap_model(model)
                    unwrapped_model.load_state_dict({
                        k: jt.array(v) if isinstance(v, np.ndarray) else v
                        for k, v in model_state.items()
                    })

                    # Resume training state
                    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
                    best_loss = checkpoint['loss']  # Preserve previous best loss

                    print(f"  Resumed from epoch {checkpoint['epoch']}")
                    print(f"  Previous best loss: {best_loss:.4f}")
                    print(f"  Continuing from epoch {start_epoch}")
                    break

                elif response in ['no', 'n']:
                    print("Archiving previous checkpoint and starting fresh...")

                    # Archive previous run (similar to eval_lvis_zeroshot_full.py)
                    archive_dir = os.path.join(args.output_dir, f'archive_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                    os.makedirs(archive_dir, exist_ok=True)

                    # Move checkpoint files to archive
                    checkpoint_files = [
                        'checkpoint_latest.pkl',
                        'checkpoint_best.pkl'
                    ]
                    for fname in checkpoint_files:
                        src = os.path.join(args.output_dir, fname)
                        if os.path.exists(src):
                            dst = os.path.join(archive_dir, fname)
                            os.rename(src, dst)
                            print(f"  Archived {fname} to {archive_dir}")

                    # Also archive any epoch-specific checkpoints
                    for fname in os.listdir(args.output_dir):
                        if fname.startswith('checkpoint_epoch') and fname.endswith('.pkl'):
                            src = os.path.join(args.output_dir, fname)
                            dst = os.path.join(archive_dir, fname)
                            os.rename(src, dst)
                            print(f"  Archived {fname} to {archive_dir}")

                    print(f"Previous run archived to: {archive_dir}")
                    print(f"Starting fresh training from epoch 1")
                    break

                else:
                    print("Please answer 'yes' or 'no'")

            except KeyboardInterrupt:
                print("\nOperation cancelled. Exiting...")
                sys.exit(0)
            except EOFError:
                # Handle non-interactive environments
                print("Non-interactive environment detected. Starting fresh training...")
                break

    else:
        print(f"\n{'='*50} Starting fresh training from epoch 1 {'='*50}")

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{config.epochs} {'='*20}")

        # Update learning rate
        current_lr = lr_scheduler.step(epoch)
        print(f"Learning rate: {current_lr:.6f}")

        # Train
        train_losses = train_one_epoch(
            model, criterion, optimizers, lr_scheduler,
            dataloader, config, epoch, args.gpus
        )

        print(f"\nEpoch {epoch} - Train Loss: {train_losses['loss']:.4f}")
        print(f"  Classification: {train_losses['loss_cls']:.4f}")
        print(f"  BBox L1: {train_losses['loss_bbox']:.4f}")
        print(f"  GIoU: {train_losses['loss_giou']:.4f}")

        # Save checkpoints
        if epoch % config.save_interval == 0 or epoch == config.epochs:
            save_checkpoint(model, optimizers, epoch, train_losses['loss'],
                          args.output_dir, 'checkpoint')

        # Save best model
        if train_losses['loss'] < best_loss:
            best_loss = train_losses['loss']
            save_best_checkpoint(model, optimizers, train_losses['loss'], args.output_dir)

        # Final cleanup
        jt.sync_all()
        jt.gc()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
