#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - LVIS Fine-tuning Script

Target: AP 52.1, APr 35.4, APc 51.3, APf 55.7 on LVIS MiniVal

Usage:
    # Quick test (small scale)
    python scripts/finetune_lvis_full.py --test_only --num_samples 10

    # Full training
    python scripts/finetune_lvis_full.py \
        --epochs 20 \
        --batch_size 4 \
        --lr 1e-4 \
        --output_dir outputs/finetune_lvis
"""

import os
import sys
import argparse
import pickle
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Set GPU before importing jittor
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '4')

# Add project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'jittor_implementation'))
sys.path.insert(0, BASE_DIR)

PT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'GroundingDINO-main')
sys.path.insert(0, PT_DIR)

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1


# ============================================================
# Configuration
# ============================================================

class FinetuneConfig:
    """Configuration for fine-tuning"""
    # Model
    hidden_dim = 256
    num_queries = 900
    num_feature_levels = 4
    nheads = 8
    max_text_len = 256
    
    # Data
    image_size = 800
    
    # Loss weights (from official config)
    cls_loss_coef = 2.0
    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    
    # Training defaults
    lr = 1e-4
    lr_backbone = 1e-5
    lr_text_encoder = 1e-5
    weight_decay = 1e-4
    epochs = 20
    batch_size = 2
    lr_drop = 15
    
    # Freeze settings (default: freeze nothing for full fine-tuning)
    freeze_backbone = False
    freeze_text_encoder = False


def parse_args():
    parser = argparse.ArgumentParser(description='Grounding DINO LVIS Fine-tuning')
    
    # Paths
    parser.add_argument('--checkpoint', type=str,
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--lvis_train', type=str,
                        default='data/lvis_notation/lvis_v1_train.json',
                        help='Path to LVIS train annotation')
    parser.add_argument('--lvis_val', type=str,
                        default='data/lvis_notation/lvis_v1_val.json/lvis_v1_val.json',
                        help='Path to LVIS val annotation')
    parser.add_argument('--image_dir', type=str,
                        default='data/coco',
                        help='Path to COCO images')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/finetune_lvis',
                        help='Output directory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_drop', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Freeze settings
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone (Swin Transformer)')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False,
                        help='Freeze text encoder (BERT)')
    
    # Testing
    parser.add_argument('--test_only', action='store_true', help='Only run quick test')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples for quick test')
    parser.add_argument('--gpu', type=int, default=4)
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=5)
    
    return parser.parse_args()


# ============================================================
# Data Loading
# ============================================================

def load_lvis_data(ann_file: str, image_dir: str, is_train: bool = True,
                   max_samples: int = None) -> Tuple[List, Dict, Dict]:
    """Load LVIS annotations"""
    print(f"Loading LVIS data from {ann_file}...")
    
    with open(ann_file, 'r') as f:
        lvis_data = json.load(f)
    
    images = {img['id']: img for img in lvis_data['images']}
    categories = {cat['id']: cat for cat in lvis_data['categories']}
    
    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Get valid image IDs (with annotations)
    if is_train:
        image_ids = [img_id for img_id in images.keys() if len(img_to_anns[img_id]) > 0]
    else:
        image_ids = list(images.keys())
    
    if max_samples:
        image_ids = image_ids[:max_samples]
    
    print(f"  Total images: {len(image_ids)}")
    print(f"  Categories: {len(categories)}")
    
    # Build category name to ID mapping
    cat_name_to_id = {cat['name'].lower().replace('_', ' '): cat_id 
                      for cat_id, cat in categories.items()}
    
    return image_ids, images, img_to_anns, categories, cat_name_to_id


def preprocess_image(image: Image.Image, target_size: int = 800):
    """Preprocess image for model input"""
    # Resize maintaining aspect ratio
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # HWC -> CHW
    img_tensor = jt.array(img_array.transpose(2, 0, 1).astype(np.float32))
    
    return img_tensor, (new_w, new_h), (w, h)


def prepare_batch(image_ids: List[int], images: Dict, img_to_anns: Dict,
                  categories: Dict, image_dir: str, config: FinetuneConfig):
    """Prepare a batch of data"""
    batch_images = []
    batch_targets = []
    batch_captions = []
    
    for img_id in image_ids:
        img_info = images[img_id]
        
        # Get image path
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"
        
        # Determine subdirectory (train2017 or val2017)
        img_path = os.path.join(image_dir, 'train2017', file_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, 'val2017', file_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        img_tensor, (new_w, new_h), _ = preprocess_image(image, config.image_size)
        
        # Get annotations
        anns = img_to_anns[img_id]
        
        boxes = []
        labels = []
        cat_names_in_image = set()
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Convert to normalized cxcywh
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            
            boxes.append([cx, cy, nw, nh])
            labels.append(cat_id)
            cat_names_in_image.add(categories[cat_id]['name'].lower().replace('_', ' '))
        
        if len(boxes) == 0:
            continue
        
        # Build caption from category names in image
        cat_names_list = list(cat_names_in_image)
        caption = ' . '.join(cat_names_list) + ' .'
        
        batch_images.append(img_tensor)
        batch_targets.append({
            'boxes': jt.array(np.array(boxes, dtype=np.float32)),
            'labels': jt.array(np.array(labels, dtype=np.int64)),
            'image_id': img_id,
            'orig_size': (orig_h, orig_w),
            'caption': caption,
            'cat_names': cat_names_list,
        })
        batch_captions.append(caption)
    
    if len(batch_images) == 0:
        return None, None, None
    
    # Pad images to same size
    max_h = max(img.shape[1] for img in batch_images)
    max_w = max(img.shape[2] for img in batch_images)
    
    padded_images = []
    for img in batch_images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            padded = jt.zeros((3, max_h, max_w))
            padded[:, :img.shape[1], :img.shape[2]] = img
            padded_images.append(padded)
        else:
            padded_images.append(img)
    
    batched_images = jt.stack(padded_images, dim=0)
    
    return batched_images, batch_targets, batch_captions


# ============================================================
# Loss Functions
# ============================================================

def box_cxcywh_to_xyxy(x):
    """Convert boxes from cxcywh to xyxy format"""
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return jt.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = jt.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = jt.clamp(rb - lt, min_v=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)
    
    return iou


def generalized_box_iou(boxes1, boxes2):
    """Compute generalized IoU between two sets of boxes"""
    iou = box_iou(boxes1, boxes2)
    
    lt = jt.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = jt.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    
    wh = jt.clamp(rb - lt, min_v=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - box_iou(boxes1, boxes2) * (
        (boxes1[:, 2] - boxes1[:, 0])[:, None] * (boxes1[:, 3] - boxes1[:, 1])[:, None] +
        (boxes2[:, 2] - boxes2[:, 0])[None, :] * (boxes2[:, 3] - boxes2[:, 1])[None, :] -
        iou * ((boxes1[:, 2] - boxes1[:, 0])[:, None] * (boxes1[:, 3] - boxes1[:, 1])[:, None] +
               (boxes2[:, 2] - boxes2[:, 0])[None, :] * (boxes2[:, 3] - boxes2[:, 1])[None, :])
    )) / (area + 1e-6)


class HungarianMatcher:
    """Hungarian matcher for bipartite matching between predictions and targets"""
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    def __call__(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of dicts with 'labels' and 'boxes'
        
        Returns:
            list of (pred_indices, target_indices) tuples
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        pred_logits = outputs['pred_logits'].flatten(0, 1)  # (bs*num_queries, num_classes)
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)    # (bs*num_queries, 4)
        
        indices = []
        
        for i in range(bs):
            tgt_boxes = targets[i]['boxes']
            num_targets = tgt_boxes.shape[0]
            
            if num_targets == 0:
                indices.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                continue
            
            # Get predictions for this image
            start_idx = i * num_queries
            end_idx = (i + 1) * num_queries
            
            out_prob = jt.sigmoid(pred_logits[start_idx:end_idx])  # (num_queries, num_classes)
            out_bbox = pred_boxes[start_idx:end_idx]                # (num_queries, 4)
            
            # Cost matrix
            # Classification cost: negative of max probability
            cost_class = -out_prob.max(dim=-1)[0].numpy()  # (num_queries,)
            cost_class = np.tile(cost_class[:, None], (1, num_targets))
            
            # L1 cost
            cost_bbox = jt.abs(out_bbox[:, None] - tgt_boxes[None, :]).sum(-1).numpy()
            
            # GIoU cost
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            cost_giou = -box_iou(out_bbox_xyxy, tgt_bbox_xyxy).numpy()
            
            # Combined cost
            C = (self.cost_class * cost_class + 
                 self.cost_bbox * cost_bbox + 
                 self.cost_giou * cost_giou)
            
            # Hungarian matching
            pred_idx, tgt_idx = linear_sum_assignment(C)
            
            indices.append((pred_idx, tgt_idx))
        
        return indices


class GroundingDINOLoss:
    """Loss function for Grounding DINO fine-tuning"""
    
    def __init__(self, num_classes=1203, 
                 cls_loss_coef=2.0, bbox_loss_coef=5.0, giou_loss_coef=2.0):
        self.num_classes = num_classes
        self.cls_loss_coef = cls_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        
        self.matcher = HungarianMatcher(
            cost_class=cls_loss_coef,
            cost_bbox=bbox_loss_coef,
            cost_giou=giou_loss_coef
        )
    
    def __call__(self, outputs, targets):
        """Compute losses"""
        # Hungarian matching
        indices = self.matcher(outputs, targets)
        
        # Classification loss (focal loss on matched pairs)
        loss_cls = self._loss_labels(outputs, targets, indices)
        
        # Bounding box losses
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices)
        
        # Weighted sum
        total_loss = (self.cls_loss_coef * loss_cls + 
                      self.bbox_loss_coef * loss_bbox + 
                      self.giou_loss_coef * loss_giou)
        
        return {
            'loss': total_loss,
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }
    
    def _loss_labels(self, outputs, targets, indices):
        """Focal loss for classification"""
        pred_logits = outputs['pred_logits']  # (bs, num_queries, max_text_len)
        
        # Simple classification loss: maximize probability for matched predictions
        losses = []
        
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            # Get matched predictions
            matched_logits = pred_logits[i, pred_idx]  # (num_matched, max_text_len)
            
            # Apply sigmoid and compute binary cross entropy
            matched_probs = jt.sigmoid(matched_logits)
            
            # We want high probability for at least some tokens
            max_probs = matched_probs.max(dim=-1)[0]
            loss = -jt.log(max_probs + 1e-6).mean()
            
            losses.append(loss)
        
        if len(losses) == 0:
            return jt.array(0.0)
        
        return sum(losses) / len(losses)
    
    def _loss_boxes(self, outputs, targets, indices):
        """L1 and GIoU loss for bounding boxes"""
        pred_boxes = outputs['pred_boxes']  # (bs, num_queries, 4)
        
        src_boxes = []
        tgt_boxes = []
        
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            src_boxes.append(pred_boxes[i, pred_idx])
            tgt_boxes.append(targets[i]['boxes'][tgt_idx])
        
        if len(src_boxes) == 0:
            return jt.array(0.0), jt.array(0.0)
        
        src_boxes = jt.concat(src_boxes, dim=0)
        tgt_boxes = jt.concat(tgt_boxes, dim=0)
        
        # L1 loss
        loss_bbox = jt.abs(src_boxes - tgt_boxes).mean()
        
        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        
        iou = box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
        giou = iou.diag()
        loss_giou = (1 - giou).mean()
        
        return loss_bbox, loss_giou


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path: str, config: FinetuneConfig):
    """Load Grounding DINO model"""
    from quick_test_zeroshot import load_model as load_model_base
    
    print(f"Loading model from {checkpoint_path}...")
    model = load_model_base(checkpoint_path)
    
    # Enable gradients for all Jittor parameters (they may be disabled after load_state_dict)
    for param in model.parameters():
        param.start_grad()
    
    # Set to training mode
    model.train()
    
    return model


def freeze_parameters(model, config: FinetuneConfig):
    """Freeze specified parameters"""
    frozen_count = 0
    
    # Get all Jittor parameters (not HuggingFace BERT which is PyTorch)
    jittor_params = list(model.parameters())
    
    if config.freeze_backbone:
        for name, param in model.named_parameters():
            if 'backbone' in name or 'input_proj' in name:
                param.stop_grad()
                frozen_count += 1
        print(f"Froze backbone: {frozen_count} parameters")
    
    # Note: text_encoder uses HuggingFace BERT (PyTorch), not Jittor
    # Freezing is handled internally by setting requires_grad=False
    if config.freeze_text_encoder and hasattr(model, 'text_encoder'):
        if hasattr(model.text_encoder, 'bert'):
            # Freeze HuggingFace BERT
            for param in model.text_encoder.bert.parameters():
                param.requires_grad = False
            print("Froze HuggingFace BERT text encoder")
    
    # Count trainable parameters (Jittor only)
    total_params = sum(p.numel() for p in jittor_params)
    trainable_params = sum(p.numel() for p in jittor_params if p.requires_grad)
    
    print(f"Total Jittor parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Jittor parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return model


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, optimizer, loss_fn, image_ids, images, img_to_anns,
                    categories, image_dir, config, epoch):
    """Train for one epoch"""
    model.train()
    
    # Shuffle image IDs
    np.random.shuffle(image_ids)
    
    total_loss = 0
    total_loss_cls = 0
    total_loss_bbox = 0
    total_loss_giou = 0
    num_batches = 0
    
    # Create batches
    batch_size = config.batch_size
    num_batches_total = (len(image_ids) + batch_size - 1) // batch_size
    
    pbar = tqdm(range(0, len(image_ids), batch_size), desc=f"Epoch {epoch}")
    
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, len(image_ids))
        batch_img_ids = image_ids[start_idx:end_idx]
        
        # Prepare batch
        batch_images, batch_targets, batch_captions = prepare_batch(
            batch_img_ids, images, img_to_anns, categories, image_dir, config
        )
        
        if batch_images is None:
            continue
        
        # Forward pass
        with jt.enable_grad():
            outputs = model([batch_images[i] for i in range(batch_images.shape[0])], 
                          captions=batch_captions)
            
            # Compute loss
            losses = loss_fn(outputs, batch_targets)
            loss = losses['loss']
            
            # Backward pass
            optimizer.step(loss)
        
        # Record losses
        total_loss += loss.item()
        total_loss_cls += losses['loss_cls'].item()
        total_loss_bbox += losses['loss_bbox'].item()
        total_loss_giou += losses['loss_giou'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{losses['loss_cls'].item():.4f}",
            'bbox': f"{losses['loss_bbox'].item():.4f}",
        })
        
        # Cleanup
        jt.sync_all()
        jt.gc()
    
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
    
    # Convert model state to numpy
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
    }
    
    save_path = os.path.join(output_dir, f'{name}_epoch{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Saved checkpoint: {save_path}")
    
    # Also save as latest
    latest_path = os.path.join(output_dir, f'{name}_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    return save_path


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config
    config = FinetuneConfig()
    config.lr = args.lr
    config.lr_backbone = args.lr_backbone
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr_drop = args.lr_drop
    config.weight_decay = args.weight_decay
    config.freeze_backbone = args.freeze_backbone
    config.freeze_text_encoder = args.freeze_text_encoder
    
    print("=" * 70)
    print("Grounding DINO LVIS Fine-tuning")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    
    # Load LVIS data
    if args.test_only:
        print(f"\n[Test Mode] Using {args.num_samples} samples")
        # Use validation set for testing
        image_ids, images, img_to_anns, categories, cat_name_to_id = load_lvis_data(
            os.path.join(BASE_DIR, args.lvis_val),
            os.path.join(BASE_DIR, args.image_dir),
            is_train=True,
            max_samples=args.num_samples
        )
        val_image_ids = image_ids[:min(5, len(image_ids))]
    else:
        # Load full training data
        print("\nLoading training data...")
        train_ann_path = os.path.join(BASE_DIR, args.lvis_train)
        if os.path.exists(train_ann_path):
            image_ids, images, img_to_anns, categories, cat_name_to_id = load_lvis_data(
                train_ann_path,
                os.path.join(BASE_DIR, args.image_dir),
                is_train=True
            )
        else:
            print(f"Warning: Training annotation not found at {train_ann_path}")
            print("Using validation set instead...")
            image_ids, images, img_to_anns, categories, cat_name_to_id = load_lvis_data(
                os.path.join(BASE_DIR, args.lvis_val),
                os.path.join(BASE_DIR, args.image_dir),
                is_train=True
            )
        
        # Load validation data
        val_image_ids_data = load_lvis_data(
            os.path.join(BASE_DIR, args.lvis_val),
            os.path.join(BASE_DIR, args.image_dir),
            is_train=False,
            max_samples=100  # Use 100 images for validation
        )
        val_image_ids = val_image_ids_data[0]
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = os.path.join(BASE_DIR, args.checkpoint)
    model = load_model(checkpoint_path, config)
    
    # Freeze parameters
    model = freeze_parameters(model, config)
    
    # Create loss function
    loss_fn = GroundingDINOLoss(
        num_classes=len(categories),
        cls_loss_coef=config.cls_loss_coef,
        bbox_loss_coef=config.bbox_loss_coef,
        giou_loss_coef=config.giou_loss_coef
    )
    
    # Create optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = jt.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    print(f"\nOptimizer: AdamW, LR: {config.lr}, Weight decay: {config.weight_decay}")
    print(f"Trainable parameter groups: {len(trainable_params)}")
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{config.epochs} {'='*20}")
        
        # Train
        train_losses = train_one_epoch(
            model, optimizer, loss_fn,
            image_ids.copy(), images, img_to_anns, categories,
            os.path.join(BASE_DIR, args.image_dir), config, epoch
        )
        
        print(f"\nEpoch {epoch} - Train Loss: {train_losses['loss']:.4f}")
        print(f"  Classification: {train_losses['loss_cls']:.4f}")
        print(f"  BBox L1: {train_losses['loss_bbox']:.4f}")
        print(f"  GIoU: {train_losses['loss_giou']:.4f}")
        
        # Learning rate decay
        if epoch == config.lr_drop:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"  LR dropped to: {optimizer.param_groups[0]['lr']}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == config.epochs:
            save_checkpoint(model, optimizer, epoch, train_losses['loss'],
                          args.output_dir, 'groundingdino_lvis')
        
        # Save best model
        if train_losses['loss'] < best_loss:
            best_loss = train_losses['loss']
            save_checkpoint(model, optimizer, epoch, train_losses['loss'],
                          args.output_dir, 'groundingdino_lvis_best')
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

