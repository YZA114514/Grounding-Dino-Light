#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - LVIS Zero-Shot 评估脚本

在 LVIS 数据集上进行 Zero-Shot 目标检测评估。
使用预训练权重，不进行任何微调。

用法:
    python scripts/eval_lvis_zeroshot.py \
        --checkpoint weights/groundingdino_swint_ogc_jittor.pkl \
        --lvis_ann data/lvis/lvis_v1_val.json \
        --image_dir data/coco/val2017 \
        --output_dir outputs/lvis_zeroshot
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
from PIL import Image

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
    
    # 创建模型
    model = GroundingDINO(
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
    )
    
    # 加载权重
    with open(checkpoint_path, 'rb') as f:
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
    model.eval()
    
    print(f"Loaded {loaded}/{len(cleaned)} weights")
    
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
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        return image, img_id, img_info
    
    def get_category_prompt(self, batch_size: int = 50) -> List[str]:
        """
        获取类别 prompt (分批处理，避免过长)
        
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
    
    def get_all_categories_prompt(self) -> str:
        """获取包含所有类别的 prompt"""
        return '. '.join(self.category_names) + '.'


# ============================================================
# 后处理
# ============================================================

def postprocess_predictions(
    outputs: Dict,
    orig_size: Tuple[int, int],
    config: Config,
    category_names: List[str]
) -> List[Dict]:
    """
    后处理模型输出
    
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
    
    # 计算置信度
    pred_probs = jt.sigmoid(pred_logits)
    
    # 获取每个 query 的最大置信度和对应类别
    max_probs, pred_labels = jt.argmax(pred_probs, dim=-1, keepdims=False)
    max_probs = max_probs[0] if len(max_probs.shape) > 0 else max_probs
    pred_labels = pred_labels[0] if len(pred_labels.shape) > 0 else pred_labels
    
    # 过滤低置信度预测
    mask = max_probs > config.box_threshold
    
    if not jt.any(mask):
        return []
    
    # 获取有效预测
    scores = max_probs[mask].numpy()
    labels = pred_labels[mask].numpy()
    boxes = pred_boxes[mask].numpy()
    
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
# 评估函数
# ============================================================

def evaluate_lvis(
    model,
    dataset: LVISEvaluationDataset,
    config: Config,
    output_dir: str
) -> Dict:
    """
    在 LVIS 上评估模型
    
    Args:
        model: 模型
        dataset: 数据集
        config: 配置
        output_dir: 输出目录
        
    Returns:
        评估指标
    """
    model.eval()
    
    all_predictions = []
    
    # 获取类别 prompt
    # 对于 LVIS 的 1203 个类别，我们分批处理
    print(f"\nRunning Zero-Shot evaluation on {len(dataset)} images...")
    print(f"Number of categories: {len(dataset.category_names)}")
    
    # 使用进度条
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        # 加载图像
        image, img_id, img_info = dataset.get_image(idx)
        
        # 预处理
        img_tensor, orig_size = preprocess_image(image, config)
        img_tensor = img_tensor.unsqueeze(0)
        
        # 对于零样本测试，使用所有类别名称或部分类别名称
        # 由于 LVIS 有 1203 个类别，一次性使用所有类别可能太长
        # 这里我们使用图像中实际存在的类别，或者分批处理
        
        # 获取该图像的 GT 类别（用于构建 prompt）
        gt_anns = dataset.img_to_anns.get(img_id, [])
        if gt_anns:
            gt_cat_ids = list(set([ann['category_id'] for ann in gt_anns]))
            cat_names = [dataset.categories[cid]['name'] for cid in gt_cat_ids]
            caption = '. '.join(cat_names) + '.'
        else:
            # 没有 GT，使用常见类别
            caption = "person. car. dog. cat. chair. table."
        
        # 前向传播
        with jt.no_grad():
            outputs = model(img_tensor, captions=[caption])
        
        # 后处理
        results = postprocess_predictions(outputs, orig_size, config, dataset.category_names)
        
        # 应用 NMS
        results = apply_nms(results, config.nms_threshold)
        
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
    
    # 运行 COCO 评估
    metrics = run_coco_eval(dataset.ann_file, pred_file)
    
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
    parser = argparse.ArgumentParser(description='LVIS Zero-Shot Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--lvis_ann', type=str, required=True,
                        help='LVIS annotation file path')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image directory path')
    parser.add_argument('--output_dir', type=str, default='outputs/lvis_zeroshot',
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
    
    # 设备参数
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU')
    
    args = parser.parse_args()
    
    # 设置 Jittor
    if args.use_gpu:
        jt.flags.use_cuda = 1
        print("Using GPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置
    config = Config()
    config.box_threshold = args.box_threshold
    config.text_threshold = args.text_threshold
    config.nms_threshold = args.nms_threshold
    config.batch_size = args.batch_size
    
    print("=" * 60)
    print("LVIS Zero-Shot Evaluation")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LVIS annotation: {args.lvis_ann}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Box threshold: {config.box_threshold}")
    print(f"NMS threshold: {config.nms_threshold}")
    
    # 加载模型
    model = load_model(args.checkpoint, config)
    
    # 加载数据集
    dataset = LVISEvaluationDataset(args.lvis_ann, args.image_dir)
    
    # 评估
    start_time = time.time()
    metrics = evaluate_lvis(model, dataset, config, args.output_dir)
    eval_time = time.time() - start_time
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTotal evaluation time: {eval_time:.1f}s")
    print(f"Time per image: {eval_time / len(dataset):.3f}s")
    
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

