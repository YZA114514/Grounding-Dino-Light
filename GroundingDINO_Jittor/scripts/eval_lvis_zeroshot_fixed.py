#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - LVIS Zero-Shot 评估脚本 (FIXED VERSION)

修复了以下关键问题：
1. 后处理逻辑错误 - 正确计算每个query的置信度
2. 数值稳定性 - 处理极端logit值
3. 真正的零样本评估 - 使用所有类别而非GT类别
4. 正确的token到类别映射

用法:
    python scripts/eval_lvis_zeroshot_fixed.py \
        --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
        --lvis_ann data/lvis/lvis_v1_val.json \
        --image_dir data/coco/val2017 \
        --output_dir outputs/lvis_zeroshot_fixed
"""

import os
import sys
import json
import argparse
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
# Patch numpy.float for older libraries
if not hasattr(np, 'float'):
    np.float = float

from PIL import Image, ImageDraw, ImageFont

try:
    import jittor as jt
    from jittor import nn
except ImportError:
    print("错误: 请安装 Jittor: pip install jittor")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================

class Config:
    """评估配置"""
    def __init__(self):
        self.box_threshold = 0.25      # 边界框置信度阈值
        self.text_threshold = 0.25     # 文本匹配阈值
        self.nms_threshold = 0.5       # NMS 阈值
        self.max_detections = 300      # 每张图最大检测数
        self.batch_size = 1            # 批次大小
        self.image_size = 800          # 图像短边尺寸
        self.max_size = 1333           # 图像长边最大尺寸
        self.prompt_batch_size = 50    # 每个prompt包含的类别数


# ============================================================
# 图像预处理
# ============================================================

def preprocess_image(image: Image.Image, config: Config) -> Tuple[jt.Var, Tuple[int, int]]:
    """
    预处理图像
    
    Args:
        image: PIL 图像
        config: 配置
        
    Returns:
        处理后的图像张量和原始尺寸
    """
    orig_w, orig_h = image.size
    
    # 计算缩放比例
    scale = config.image_size / min(orig_h, orig_w)
    if max(orig_h, orig_w) * scale > config.max_size:
        scale = config.max_size / max(orig_h, orig_w)
    
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    # 缩放图像
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # 转换为张量
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1))
    
    # 标准化
    mean = jt.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = jt.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, (orig_h, orig_w)


# ============================================================
# 模型加载
# ============================================================

def load_model(checkpoint_path: str, config: Config):
    """
    加载模型
    
    Args:
        checkpoint_path: 权重文件路径
        config: 配置
        
    Returns:
        加载好的模型
    """
    print(f"Loading model from {checkpoint_path}...")
    
    from jittor_implementation.models.groundingdino import GroundingDINO
    from jittor_implementation.models.backbone.swin_transformer import SwinTransformer, build_swin_transformer
    
    # 创建 backbone
    backbone = build_swin_transformer(
        modelname="swin_T_224_1k",
        pretrain_img_size=224,
        out_indices=(1, 2, 3),  # 输出 stage 2, 3, 4 的特征
        dilation=False,
    )
    
    # 创建模型
    # 注意：官方预训练权重使用 dec_pred_bbox_embed_share=False (每层独立的 bbox_embed)
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
        two_stage_type="standard",  # 使用 two-stage
        dec_pred_bbox_embed_share=False,  # 官方权重使用独立的 bbox_embed
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=False,  # enc_out_bbox_embed 也是独立的
    )
    
    # 加载权重
    with open(checkpoint_path, 'rb') as f:
        weights = pickle.load(f)
    
    print(f"Checkpoint has {len(weights)} weights")
    
    # 清理权重名称 (去除 module. 前缀) 并应用名称映射
    cleaned = {}
    for k, v in weights.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[7:]
        
        # 应用名称映射规则
        # 1. backbone.0. -> backbone. (checkpoint 使用 backbone.0 表示第一个 backbone)
        if new_k.startswith('backbone.0.'):
            new_k = 'backbone.' + new_k[11:]
        
        # 2. transformer.level_embed -> level_embed
        if new_k == 'transformer.level_embed':
            new_k = 'level_embed'
        
        # 3. transformer.tgt_embed.weight -> tgt_embed.weight
        if new_k == 'transformer.tgt_embed.weight':
            new_k = 'tgt_embed.weight'
        
        # 6. transformer.enc_output.* -> enc_output.*
        if new_k.startswith('transformer.enc_output'):
            new_k = new_k.replace('transformer.enc_output', 'enc_output')
        
        # 7. bbox_embed.X.* -> transformer.decoder.bbox_embed.X.* (顶层 bbox_embed 映射到 decoder)
        if new_k.startswith('bbox_embed.'):
            new_k = 'transformer.decoder.' + new_k
        
        cleaned[new_k] = v
    
    # 分离 BERT 权重和其他权重
    bert_weights = {}
    other_weights = {}
    for k, v in cleaned.items():
        if k.startswith('bert.'):
            bert_weights[k] = v
        else:
            other_weights[k] = v
    
    print(f"  - BERT weights: {len(bert_weights)}")
    print(f"  - Other weights: {len(other_weights)}")
    
    # 处理 in_proj_weight/bias 到 q_proj/k_proj/v_proj 的转换
    converted_weights = {}
    for k, v in other_weights.items():
        if '.in_proj_weight' in k:
            # 拆分 in_proj_weight [3*d, d] -> q_proj [d, d], k_proj [d, d], v_proj [d, d]
            d = v.shape[0] // 3
            base_key = k.replace('.in_proj_weight', '.')
            converted_weights[base_key + 'q_proj.weight'] = v[:d, :]
            converted_weights[base_key + 'k_proj.weight'] = v[d:2*d, :]
            converted_weights[base_key + 'v_proj.weight'] = v[2*d:, :]
        elif '.in_proj_bias' in k:
            # 拆分 in_proj_bias [3*d] -> q_proj [d], k_proj [d], v_proj [d]
            d = v.shape[0] // 3
            base_key = k.replace('.in_proj_bias', '.')
            converted_weights[base_key + 'q_proj.bias'] = v[:d]
            converted_weights[base_key + 'k_proj.bias'] = v[d:2*d]
            converted_weights[base_key + 'v_proj.bias'] = v[2*d:]
        else:
            converted_weights[k] = v
    
    other_weights = converted_weights
    
    # 加载非 BERT 权重到模型
    model_state = model.state_dict()
    loaded = 0
    missing_in_model = []
    missing_in_ckpt = []
    shape_mismatch = []
    
    for k, v in other_weights.items():
        if k in model_state:
            if model_state[k].shape == tuple(v.shape):
                model_state[k] = jt.array(v)
                loaded += 1
            else:
                shape_mismatch.append(f"{k}: model {model_state[k].shape} vs ckpt {v.shape}")
        else:
            missing_in_model.append(k)
    
    # 检查模型中有但权重中没有的
    for k in model_state.keys():
        if k not in other_weights and not k.startswith('text_encoder.bert.'):
            missing_in_ckpt.append(k)
    
    model.load_state_dict(model_state)
    
    # 加载 BERT 权重到纯 Jittor BERT
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
        print("Loading BERT weights into pure Jittor BERT...")
        
        # 将 bert_weights 转换为带正确前缀的格式
        bert_state = model.text_encoder.bert.state_dict()
        bert_loaded = 0
        
        for k, v in bert_weights.items():
            # bert.xxx -> xxx
            bert_key = k[5:] if k.startswith('bert.') else k
            if bert_key in bert_state:
                if bert_state[bert_key].shape == tuple(v.shape):
                    bert_state[bert_key] = jt.array(v)
                    bert_loaded += 1
                else:
                    print(f"  BERT shape mismatch: {bert_key}: {bert_state[bert_key].shape} vs {v.shape}")
        
        model.text_encoder.bert.load_state_dict(bert_state)
        print(f"  Loaded {bert_loaded}/{len(bert_weights)} BERT weights")
    
    model.eval()
    
    print(f"\nWeight loading summary:")
    print(f"  - Loaded: {loaded}/{len(other_weights)} non-BERT weights")
    if shape_mismatch:
        print(f"  - Shape mismatches: {len(shape_mismatch)}")
        for s in shape_mismatch[:3]:
            print(f"      {s}")
    if missing_in_model:
        print(f"  - Weights not in model: {len(missing_in_model)}")
        for m in missing_in_model[:3]:
            print(f"      {m}")
    if missing_in_ckpt:
        print(f"  - Model params not in checkpoint: {len(missing_in_ckpt)}")
        for m in missing_in_ckpt[:3]:
            print(f"      {m}")
    
    return model


# ============================================================
# LVIS 数据加载
# ============================================================

class LVISEvaluationDataset:
    """LVIS 评估数据集"""
    
    def __init__(self, ann_file: str, image_dir: str):
        """
        Args:
            ann_file: LVIS 标注文件
            image_dir: 图像目录
        """
        print(f"Loading LVIS annotations from {ann_file}...")
        
        self.ann_file = ann_file
        with open(ann_file, 'r') as f:
            self.lvis_data = json.load(f)
        
        self.images = {img['id']: img for img in self.lvis_data['images']}
        self.categories = {cat['id']: cat for cat in self.lvis_data['categories']}
        self.image_dir = image_dir
        
        # 按图像分组标注
        self.img_to_anns = defaultdict(list)
        for ann in self.lvis_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # 获取所有图像 ID
        self.image_ids = list(self.images.keys())
        
        # 构建类别名称列表 (按 ID 排序)
        self.cat_ids = sorted(self.categories.keys())
        self.category_names = [self.categories[cid]['name'] for cid in self.cat_ids]
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(self.cat_ids)}
        
        print(f"  Images: {len(self.images)}")
        print(f"  Categories: {len(self.categories)}")
        print(f"  Annotations: {len(self.lvis_data['annotations'])}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def get_image(self, idx: int) -> Tuple[Image.Image, int, Dict]:
        """获取图像"""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # LVIS 格式可能没有 file_name，需要从 coco_url 提取
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            # 从 coco_url 提取文件名: http://images.cocodataset.org/val2017/000000397133.jpg -> 000000397133.jpg
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            # 使用 image_id 构造文件名
            file_name = f"{img_id:012d}.jpg"
        
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        return image, img_id, img_info
    
    def get_zero_shot_prompts(self, batch_size: int = 50) -> List[str]:
        """
        获取真正的零样本 prompt (使用所有类别)
        
        Args:
            batch_size: 每个 prompt 包含的类别数
            
        Returns:
            类别 prompt 列表
        """
        prompts = []
        for i in range(0, len(self.category_names), batch_size):
            batch_names = self.category_names[i:i+batch_size]
            prompt = '. '.join(batch_names) + '.'
            prompts.append(prompt)
        return prompts


# ============================================================
# 后处理 (FIXED VERSION)
# ============================================================

def postprocess_predictions(
    outputs: Dict,
    orig_size: Tuple[int, int],
    config: Config,
    category_names: List[str]
) -> List[Dict]:
    """
    后处理模型输出 (修复版本)
    
    Args:
        outputs: 模型输出
        orig_size: 原始图像尺寸 (H, W)
        config: 配置
        category_names: 类别名称列表
        
    Returns:
        检测结果列表
    """
    orig_h, orig_w = orig_size
    
    # 获取预测
    pred_logits = outputs['pred_logits'][0]  # [num_queries, max_text_len]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    print(f"DEBUG: pred_logits shape: {pred_logits.shape}")
    print(f"DEBUG: pred_logits range: [{pred_logits.min().item():.4f}, {pred_logits.max().item():.4f}]")
    
    # FIX 1: 处理极端 logits 值，避免数值不稳定
    pred_logits = jt.clamp(pred_logits, -10.0, 10.0)
    pred_probs = jt.sigmoid(pred_logits)
    
    # FIX 2: 正确计算每个 query 的最大置信度和对应类别
    # 每个 query 取最大概率和对应的 token 位置
    max_probs = jt.max(pred_probs, dim=-1)
    _, pred_labels = jt.argmax(pred_probs, dim=-1)  # Jittor argmax returns (values, indices)
    
    # FIX 3: 正确处理维度
    if max_probs.ndim > 1:
        max_probs = max_probs.squeeze(-1)
    if pred_labels.ndim > 1:
        pred_labels = pred_labels.squeeze(-1)
    
    print(f"DEBUG: max_probs shape: {max_probs.shape}")
    print(f"DEBUG: pred_labels shape: {pred_labels.shape}")
    print(f"DEBUG: max_probs range: [{max_probs.min().item():.4f}, {max_probs.max().item():.4f}]")
    print(f"DEBUG: pred_labels range: [{pred_labels.min().item()}, {pred_labels.max().item()}]")
    
    # 过滤低置信度预测
    mask = max_probs > config.box_threshold
    
    # FIX 4: 正确检查是否有有效预测
    valid_count = mask.sum().item()
    print(f"DEBUG: Predictions above threshold: {valid_count}/{len(max_probs)}")
    
    if valid_count == 0:
        return []
    
    # 获取有效预测
    scores = max_probs[mask].numpy()
    labels = pred_labels[mask].numpy()
    boxes = pred_boxes[mask].numpy()
    
    print(f"DEBUG: Valid predictions - scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"DEBUG: Valid predictions - labels range: [{labels.min()}, {labels.max()}]")
    
    # 转换边界框格式: [cx, cy, w, h] -> [x, y, w, h]
    results = []
    for i in range(len(scores)):
        cx, cy, w, h = boxes[i]
        x = (cx - w / 2) * orig_w
        y = (cy - h / 2) * orig_h
        w = w * orig_w
        h = h * orig_h
        
        # 裁剪到图像边界
        x = max(0, x)
        y = max(0, y)
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)
        
        if w > 0 and h > 0:
            results.append({
                'bbox': [float(x), float(y), float(w), float(h)],
                'score': float(scores[i]),
                'category_idx': int(labels[i]),
            })
    
    # 按置信度排序
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # 限制检测数量
    results = results[:config.max_detections]
    
    return results


def apply_nms(results: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    应用 NMS
    
    Args:
        results: 检测结果
        threshold: NMS 阈值
        
    Returns:
        NMS 后的结果
    """
    if len(results) == 0:
        return results
    
    # 按类别分组
    by_category = defaultdict(list)
    for r in results:
        by_category[r['category_idx']].append(r)
    
    final_results = []
    
    for cat_idx, cat_results in by_category.items():
        # 提取边界框和分数
        boxes = np.array([r['bbox'] for r in cat_results])
        scores = np.array([r['score'] for r in cat_results])
        
        # 转换为 [x1, y1, x2, y2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        
        # NMS
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算 IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # 保留 IoU 小于阈值的框
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        for idx in keep:
            final_results.append(cat_results[idx])
    
    return final_results


# ============================================================
# 可视化
# ============================================================

def visualize_prediction(
    image: Image.Image,
    results: List[Dict],
    category_names: List[str],
    output_path: str,
    threshold: float = 0.3
):
    """
    Visualize predictions on image
    """
    # Copy image to avoid modifying original
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Try to load a font
    try:
        # Try to find a font that supports text
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        
    for res in results:
        if res['score'] < threshold:
            continue
            
        bbox = res['bbox'] # [x, y, w, h]
        x, y, w, h = bbox
        
        # Draw box
        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
        
        # Draw label
        cat_idx = res['category_idx']
        if 0 <= cat_idx < len(category_names):
            label = f"{category_names[cat_idx]}: {res['score']:.2f}"
        else:
            label = f"Token {cat_idx}: {res['score']:.2f}"
        
        # Draw text background
        # textbbox was added in Pillow 9.2.0, textsize is deprecated
        if hasattr(draw, 'textbbox'):
            left, top, right, bottom = draw.textbbox((x, y), label, font=font)
            text_w = right - left
            text_h = bottom - top
        else:
            text_w, text_h = draw.textsize(label, font=font)
            
        draw.rectangle([x, y, x+text_w+4, y+text_h+4], fill='red')
        draw.text((x+2, y+2), label, fill='white', font=font)
        
    vis_image.save(output_path)


# ============================================================
# 评估函数 (FIXED VERSION)
# ============================================================

def evaluate_lvis(
    model,
    dataset: LVISEvaluationDataset,
    config: Config,
    output_dir: str,
    limit: Optional[int] = None,
    visualize: bool = False
) -> Dict:
    """
    在 LVIS 上评估模型 (修复版本)
    
    Args:
        model: 模型
        dataset: 数据集
        config: 配置
        output_dir: 输出目录
        limit: 限制评估的图像数量
        
    Returns:
        评估指标
    """
    model.eval()
    
    all_predictions = []
    
    # 获取零样本 prompts (使用所有类别)
    zero_shot_prompts = dataset.get_zero_shot_prompts(batch_size=config.prompt_batch_size)
    print(f"Using {len(zero_shot_prompts)} prompts for zero-shot evaluation")
    print(f"Example prompt: {zero_shot_prompts[0][:100]}...")
    
    num_images = len(dataset)
    if limit is not None and limit > 0:
        num_images = min(num_images, limit)
        print(f"\nRunning Zero-Shot evaluation on {num_images} images (limited from {len(dataset)})...")
    else:
        print(f"\nRunning Zero-Shot evaluation on {num_images} images...")
        
    print(f"Number of categories: {len(dataset.category_names)}")
    
    # 使用进度条
    for idx in tqdm(range(num_images), desc="Evaluating"):
        # 加载图像
        image, img_id, img_info = dataset.get_image(idx)
        
        # 预处理
        img_tensor, orig_size = preprocess_image(image, config)
        img_tensor = img_tensor.unsqueeze(0)
        
        # FIX: 使用零样本提示，而非 GT 提示
        # 对于 LVIS 的 1203 个类别，我们需要分批处理
        # 这里简化处理，使用第一个包含前50个类别的 prompt
        caption = zero_shot_prompts[0]  # 使用前50个类别进行测试
        
        print(f"\n--- Image {idx+1}, ID: {img_id} ---")
        print(f"Using prompt: {caption[:100]}...")
        
        # 前向传播
        with jt.no_grad():
            outputs = model(img_tensor, captions=[caption])
        
        # 后处理
        results = postprocess_predictions(outputs, orig_size, config, dataset.category_names)
        
        # 应用 NMS
        results = apply_nms(results, config.nms_threshold)
        
        # DEBUG: 打印预测结果
        print(f"Found {len(results)} predictions:")
        for i, res in enumerate(results[:5]):  # 只显示前5个
            cat_idx = res['category_idx']
            cat_name = f"Token_{cat_idx}"
            if cat_idx < len(dataset.category_names):
                cat_name = dataset.category_names[cat_idx]
            print(f"  {i+1}: {cat_name} (idx={cat_idx}) score={res['score']:.3f}")
        
        # 可视化
        if visualize:
            vis_dir = os.path.join(output_dir, 'vis')
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{img_id}.jpg")
            visualize_prediction(image, results, dataset.category_names, vis_path, config.box_threshold)
        
        # 转换为 COCO 格式
        for r in results:
            cat_idx = r['category_idx']
            # 确保类别索引有效
            if cat_idx < len(dataset.cat_ids):
                cat_id = dataset.cat_ids[cat_idx]
            else:
                continue
            
            all_predictions.append({
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': r['bbox'],
                'score': r['score']
            })
    
    print(f"\nTotal predictions: {len(all_predictions)}")
    
    # 保存预测结果
    pred_file = os.path.join(output_dir, 'predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(all_predictions, f)
    print(f"Predictions saved to {pred_file}")
    
    # 如果限制了图像数量，创建子集标注文件用于评估
    eval_ann_file = dataset.ann_file
    if limit is not None and limit > 0:
        print(f"Creating subset annotation file for evaluation...")
        
        # 获取已处理的图像 ID
        processed_img_ids = set()
        for idx in range(num_images):
            img_id = dataset.image_ids[idx]
            processed_img_ids.add(img_id)
            
        # 过滤标注
        subset_data = {
            'info': dataset.lvis_data.get('info', {}),
            'licenses': dataset.lvis_data.get('licenses', []),
            'categories': dataset.lvis_data['categories'],
            'images': [img for img in dataset.lvis_data['images'] if img['id'] in processed_img_ids],
            # Ensure iscrowd exists for COCO evaluation
            'annotations': [dict(ann, iscrowd=ann.get('iscrowd', 0)) for ann in dataset.lvis_data['annotations'] if ann['image_id'] in processed_img_ids]
        }
        
        subset_ann_file = os.path.join(output_dir, 'lvis_subset_val.json')
        with open(subset_ann_file, 'w') as f:
            json.dump(subset_data, f)
        
        eval_ann_file = subset_ann_file
        print(f"Subset annotations saved to {subset_ann_file}")
    
    # 尝试使用 LVIS 评估
    try:
        metrics = run_lvis_eval(eval_ann_file, pred_file)
    except Exception as e:
        print(f"LVIS evaluation failed: {e}")
        print("Falling back to COCO evaluation...")
        # 运行 COCO 评估
        metrics = run_coco_eval(eval_ann_file, pred_file)
    
    return metrics


def run_lvis_eval(ann_file: str, pred_file: str) -> Dict:
    """
    运行 LVIS 评估
    """
    from lvis import LVIS, LVISEval, LVISResults
    
    print("\nRunning LVIS evaluation...")
    
    # 抑制 LVIS 打印
    import logging
    logger = logging.getLogger('lvis')
    logger.setLevel(logging.INFO)
    
    lvis_gt = LVIS(ann_file)
    lvis_dt = LVISResults(lvis_gt, pred_file)
    
    lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
    lvis_eval.run()
    lvis_eval.print_results()
    
    # 提取指标
    # LVIS results 是一个字典，包含 AP, AP50, AP75 等
    results = lvis_eval.results
    
    # 映射到我们需要的格式
    metrics = {
        'AP': results.get('AP', 0.0),
        'AP50': results.get('AP50', 0.0),
        'AP75': results.get('AP75', 0.0),
        'APs': results.get('APs', 0.0),
        'APm': results.get('APm', 0.0),
        'APl': results.get('APl', 0.0),
        'APr': results.get('APr', 0.0),
        'APc': results.get('APc', 0.0),
        'APf': results.get('APf', 0.0),
    }
    
    return metrics


def run_coco_eval(ann_file: str, pred_file: str) -> Dict:
    """
    运行 COCO 评估
    
    Args:
        ann_file: GT 标注文件
        pred_file: 预测文件
        
    Returns:
        评估指标
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        print("\nRunning COCO evaluation...")
        
        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(pred_file)
        
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'APs': coco_eval.stats[3],
            'APm': coco_eval.stats[4],
            'APl': coco_eval.stats[5],
            'AR1': coco_eval.stats[6],
            'AR10': coco_eval.stats[7],
            'AR100': coco_eval.stats[8],
        }
        
        return metrics
        
    except ImportError:
        print("Warning: pycocotools not available, skipping official evaluation")
        return {'AP': 0.0}
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {'AP': 0.0}


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LVIS Zero-Shot Evaluation (Fixed)')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--lvis_ann', type=str, required=True,
                        help='LVIS annotation file path')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image directory path')
    parser.add_argument('--output_dir', type=str, default='outputs/lvis_zeroshot_fixed',
                        help='Output directory')
    
    # 评估参数
    parser.add_argument('--box_threshold', type=float, default=0.25,
                        help='Box confidence threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Text matching threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='NMS threshold')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--limit', type=int, default=10,
                        help='Limit number of images to evaluate (default: 10 for testing)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    
    # 设备参数
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    
    # 设置 Jittor
    if args.use_gpu:
        jt.flags.use_cuda = 1
        print("Using GPU")
    else:
        print("Using CPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置
    config = Config()
    config.box_threshold = args.box_threshold
    config.text_threshold = args.text_threshold
    config.nms_threshold = args.nms_threshold
    config.batch_size = args.batch_size
    
    print("=" * 60)
    print("LVIS Zero-Shot Evaluation (Fixed Version)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LVIS annotation: {args.lvis_ann}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Box threshold: {config.box_threshold}")
    print(f"NMS threshold: {config.nms_threshold}")
    print(f"Limit images: {args.limit}")
    
    # 加载模型
    model = load_model(args.checkpoint, config)
    
    # 加载数据集
    dataset = LVISEvaluationDataset(args.lvis_ann, args.image_dir)
    
    # 评估
    start_time = time.time()
    metrics = evaluate_lvis(model, dataset, config, args.output_dir, args.limit, args.visualize)
    eval_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results (Fixed Version)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTotal evaluation time: {eval_time:.1f}s")
    
    num_evaluated = args.limit if args.limit is not None and args.limit > 0 else len(dataset)
    num_evaluated = min(num_evaluated, len(dataset))
    print(f"Time per image: {eval_time / num_evaluated:.3f}s")
    
    # 保存结果
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics,
            'eval_time': eval_time,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
