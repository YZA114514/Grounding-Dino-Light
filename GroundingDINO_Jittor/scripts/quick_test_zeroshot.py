#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试 Zero-Shot 检测效果
只运行少量图片，显示检测结果
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import jittor as jt

# ============================================================
# 配置
# ============================================================

class Config:
    box_threshold = 0.1  # Lowered from 0.3 to increase recall
    text_threshold = 0.25
    nms_threshold = 0.5
    max_detections = 100
    image_size = 800
    max_size = 1333


# ============================================================
# 图像预处理
# ============================================================

def preprocess_image(image, config):
    """预处理图像"""
    orig_w, orig_h = image.size
    
    scale = config.image_size / min(orig_h, orig_w)
    if max(orig_h, orig_w) * scale > config.max_size:
        scale = config.max_size / max(orig_h, orig_w)
    
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1))
    
    mean = jt.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = jt.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, (orig_h, orig_w)


# ============================================================
# 模型加载 (复用 eval 脚本的加载逻辑)
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


def load_model(checkpoint_path):
    """加载模型"""
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
    return model


# ============================================================
# 后处理
# ============================================================

def postprocess(outputs, orig_size, config, num_classes):
    """后处理"""
    orig_h, orig_w = orig_size
    
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_tokens]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    
    pred_probs = jt.sigmoid(pred_logits)
    
    # 每个 query 取最大概率
    # jt.max 只返回值，需要用 jt.argmax 获取索引
    max_probs = jt.max(pred_probs, dim=-1)  # [num_queries]
    # jt.argmax 返回 (索引, 值)
    pred_labels, _ = jt.argmax(pred_probs, dim=-1)  # [num_queries]
    
    # 过滤低置信度
    mask = max_probs > config.box_threshold
    
    if not jt.any(mask):
        return []
    
    scores = max_probs[mask].numpy()
    labels = pred_labels[mask].numpy()
    boxes = pred_boxes[mask].numpy()
    
    results = []
    for i in range(len(scores)):
        cx, cy, w, h = boxes[i]
        
        # cxcywh -> xywh
        x = (cx - w / 2) * orig_w
        y = (cy - h / 2) * orig_h
        bw = w * orig_w
        bh = h * orig_h
        
        # 裁剪到图像边界
        x = max(0, x)
        y = max(0, y)
        
        if bw > 0 and bh > 0:
            results.append({
                'bbox': [float(x), float(y), float(bw), float(bh)],
                'score': float(scores[i]),
                'label': int(labels[i]),
            })
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results[:config.max_detections]


def apply_nms(results, threshold=0.5):
    """NMS"""
    if len(results) == 0:
        return results
    
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['label']].append(r)
    
    final = []
    for cat, cat_results in by_cat.items():
        boxes = np.array([r['bbox'] for r in cat_results])
        scores = np.array([r['score'] for r in cat_results])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        for idx in keep:
            final.append(cat_results[idx])
    
    return final


# ============================================================
# 可视化
# ============================================================

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]

def draw_results(image, results, class_names, output_path):
    """绘制检测结果"""
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for r in results:
        x, y, w, h = r['bbox']
        label = r['label']
        score = r['score']
        
        color = COLORS[label % len(COLORS)]
        
        # 绘制边框
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # 标签文本
        if label < len(class_names):
            text = f"{class_names[label]}: {score:.2f}"
        else:
            text = f"class_{label}: {score:.2f}"
        
        # 绘制文本背景
        text_bbox = draw.textbbox((x, y - 16), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 16), text, fill=(255, 255, 255), font=font)
    
    image.save(output_path)
    return image


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='weights/groundingdino_swint_ogc_jittor.pkl')
    parser.add_argument('--lvis_ann', type=str,
                        default='data/lvis_notation/lvis_v1_val.json/lvis_v1_val.json')
    parser.add_argument('--image_dir', type=str,
                        default='data/coco/val2017')
    parser.add_argument('--output_dir', type=str, default='outputs/quick_test')
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--box_threshold', type=float, default=0.1)
    args = parser.parse_args()
    
    # 设置 GPU
    jt.flags.use_cuda = 1
    print("Using GPU")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = Config()
    config.box_threshold = args.box_threshold
    
    # 加载模型
    model = load_model(args.checkpoint)
    
    # 加载 LVIS 标注
    print(f"\nLoading LVIS annotations from {args.lvis_ann}...")
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)
    
    images = {img['id']: img for img in lvis_data['images']}
    categories = {cat['id']: cat for cat in lvis_data['categories']}
    
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # 选择有标注的图片
    image_ids = [img_id for img_id in list(images.keys()) if img_id in img_to_anns]
    image_ids = image_ids[:args.num_images]
    
    print(f"\nTesting {len(image_ids)} images...")
    print("=" * 60)
    
    total_gt = 0
    total_pred = 0
    total_tp = 0
    
    for idx, img_id in enumerate(image_ids):
        img_info = images[img_id]
        
        # 获取文件名
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"
        
        img_path = os.path.join(args.image_dir, file_name)
        
        if not os.path.exists(img_path):
            print(f"[{idx+1}/{len(image_ids)}] Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert('RGB')
        
        # 获取 GT 类别
        gt_anns = img_to_anns[img_id]
        gt_cat_ids = list(set([ann['category_id'] for ann in gt_anns]))
        cat_names = [categories[cid]['name'] for cid in gt_cat_ids]
        caption = '. '.join(cat_names) + '.'
        
        # 预处理
        img_tensor, orig_size = preprocess_image(image, config)
        img_tensor = img_tensor.unsqueeze(0)
        
        # 推理
        with jt.no_grad():
            outputs = model(img_tensor, captions=[caption])
        
        # 清理中间变量，释放内存
        del img_tensor
        jt.sync_all()
        jt.gc()
        
        # 后处理
        results = postprocess(outputs, orig_size, config, len(cat_names))
        results = apply_nms(results, config.nms_threshold)
        
        # 统计
        num_gt = len(gt_anns)
        num_pred = len(results)
        
        # 简单的 TP 计算 (预测框与 GT 框的 IoU > 0.5)
        tp = 0
        for r in results:
            px, py, pw, ph = r['bbox']
            for ann in gt_anns:
                gx, gy, gw, gh = ann['bbox']
                
                # 计算 IoU
                ix1 = max(px, gx)
                iy1 = max(py, gy)
                ix2 = min(px + pw, gx + gw)
                iy2 = min(py + ph, gy + gh)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = pw * ph + gw * gh - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > 0.5:
                        tp += 1
                        break
        
        total_gt += num_gt
        total_pred += num_pred
        total_tp += tp
        
        print(f"\n[{idx+1}/{len(image_ids)}] Image: {file_name}")
        print(f"  Caption: {caption[:80]}...")
        print(f"  GT boxes: {num_gt}, Predictions: {num_pred}, TP(IoU>0.5): {tp}")
        
        # 打印检测结果
        for r in results[:5]:  # 只显示前 5 个
            label = r['label']
            if label < len(cat_names):
                name = cat_names[label]
            else:
                name = f"class_{label}"
            print(f"    - {name}: {r['score']:.3f}, bbox={[int(x) for x in r['bbox']]}")
        
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")
        
        # 保存可视化结果
        output_path = os.path.join(args.output_dir, f"result_{idx+1}_{file_name}")
        draw_results(image, results, cat_names, output_path)
        print(f"  Saved: {output_path}")
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images: {len(image_ids)}")
    print(f"Total GT boxes: {total_gt}")
    print(f"Total predictions: {total_pred}")
    print(f"Total TP (IoU > 0.5): {total_tp}")
    
    if total_gt > 0:
        recall = total_tp / total_gt
        print(f"Recall@0.5: {recall:.4f} ({total_tp}/{total_gt})")
    
    if total_pred > 0:
        precision = total_tp / total_pred
        print(f"Precision@0.5: {precision:.4f} ({total_tp}/{total_pred})")
    
    if total_gt > 0 and total_pred > 0:
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        print(f"F1@0.5: {f1:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

