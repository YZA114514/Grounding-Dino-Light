#!/usr/bin/env python
"""
LVIS Zero-Shot Evaluation Script for Grounding DINO (Jittor)

This script evaluates the Jittor implementation of Grounding DINO on the LVIS dataset
using true zero-shot evaluation (all 1203 categories processed in batches).

Target metrics (from paper):
    AP: 25.6, APr: 14.4, APc: 19.6, APf: 32.2

Usage:
    python scripts/eval_lvis_zeroshot_full.py --num_images 100 --batch_size 80
    python scripts/eval_lvis_zeroshot_full.py --full  # Full validation set (~17K images)
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Set GPU before importing jittor
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '4')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'jittor_implementation'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

PT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'GroundingDINO-main')
sys.path.insert(0, PT_DIR)

import numpy as np
import jittor as jt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from quick_test_zeroshot import load_model, preprocess_image, Config
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from transformers import AutoTokenizer

jt.flags.use_cuda = 1


def parse_args():
    parser = argparse.ArgumentParser(description='LVIS Zero-Shot Evaluation')
    parser.add_argument('--checkpoint', type=str, 
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to Jittor checkpoint')
    parser.add_argument('--lvis_ann', type=str, 
                        default='data/lvis_notation/lvis_v1_val.json/lvis_v1_val.json',
                        help='Path to LVIS annotation file')
    parser.add_argument('--image_dir', type=str, 
                        default='data/coco/val2017',
                        help='Path to COCO val2017 images')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to evaluate (0 for all)')
    parser.add_argument('--full', action='store_true',
                        help='Evaluate on full validation set')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='Number of categories per batch (to fit BERT 512 token limit)')
    parser.add_argument('--num_select', type=int, default=300,
                        help='Number of top predictions to keep per image')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=4,
                        help='GPU device ID')
    return parser.parse_args()


def build_category_batches(categories, tokenizer, batch_size=80, max_text_len=256):
    """Build batches of categories with their positive maps."""
    num_batches = (len(categories) + batch_size - 1) // batch_size
    batch_info = []
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(categories))
        batch_cats = categories[start:end]
        batch_cat_names = [cat['name'].lower().replace('_', ' ') for cat in batch_cats]
        batch_cat_ids = [cat['id'] for cat in batch_cats]
        
        # Build caption and token spans
        caption, cat2tokenspan = build_captions_and_token_span(batch_cat_names, force_lowercase=True)
        tokenized = tokenizer(caption, return_tensors="pt")
        tokenspanlist = [cat2tokenspan.get(name, []) for name in batch_cat_names]
        positive_map = create_positive_map_from_span(tokenized, tokenspanlist, max_text_len=max_text_len)
        
        batch_info.append({
            'caption': caption,
            'cat_ids': batch_cat_ids,
            'cat_names': batch_cat_names,
            'positive_map': positive_map.numpy(),
            'num_tokens': tokenized['input_ids'].shape[1]
        })
    
    return batch_info


def run_inference_batched(model, img_tensor, batch_info, orig_size, num_select=300):
    """Run inference on all category batches and aggregate results."""
    orig_w, orig_h = orig_size
    all_predictions = []
    
    for batch in batch_info:
        caption = batch['caption']
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']
        
        with jt.no_grad():
            outputs = model([img_tensor], captions=[caption])
        
        pred_logits = outputs['pred_logits'][0].numpy()
        pred_boxes = outputs['pred_boxes'][0].numpy()
        
        # Convert logits to probabilities
        prob_to_token = 1 / (1 + np.exp(-pred_logits))  # sigmoid
        prob_to_label = prob_to_token @ positive_map_np.T
        
        # Extract predictions for each query and category
        for q_idx in range(pred_boxes.shape[0]):
            for c_idx, cat_id in enumerate(batch_cat_ids):
                score = prob_to_label[q_idx, c_idx]
                if score < 0.001:  # Low threshold to keep candidates
                    continue
                
                # Convert box from cxcywh normalized to xywh absolute
                cx, cy, w, h = pred_boxes[q_idx]
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                bw = w * orig_w
                bh = h * orig_h
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                bw = min(bw, orig_w - x1)
                bh = min(bh, orig_h - y1)
                
                if bw <= 0 or bh <= 0:
                    continue
                
                all_predictions.append({
                    'category_id': int(cat_id),
                    'bbox': [float(x1), float(y1), float(bw), float(bh)],
                    'score': float(score)
                })
    
    # Keep top-k predictions
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    return all_predictions[:num_select]


def evaluate_with_pycocotools(predictions, lvis_data, image_ids, categories, output_dir):
    """Run official COCO-style evaluation."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    # Build ground truth in COCO format
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    coco_gt_dict = {
        'info': {'description': 'LVIS v1 validation subset', 'date_created': datetime.now().isoformat()},
        'licenses': [],
        'images': [img for img in lvis_data['images'] if img['id'] in image_ids],
        'annotations': [],
        'categories': categories
    }
    
    ann_id = 1
    for img_id in image_ids:
        for ann in img_to_anns.get(img_id, []):
            coco_gt_dict['annotations'].append({
                'id': ann_id,
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['bbox'][2] * ann['bbox'][3],
                'iscrowd': 0
            })
            ann_id += 1
    
    # Save GT and predictions
    gt_file = os.path.join(output_dir, 'lvis_gt_subset.json')
    pred_file = os.path.join(output_dir, 'lvis_predictions.json')
    
    with open(gt_file, 'w') as f:
        json.dump(coco_gt_dict, f)
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)
    
    # Load and evaluate
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)
    
    # Overall AP
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = list(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    results = {
        'AP': coco_eval.stats[0] * 100,
        'AP50': coco_eval.stats[1] * 100,
        'AP75': coco_eval.stats[2] * 100,
        'APs': coco_eval.stats[3] * 100,
        'APm': coco_eval.stats[4] * 100,
        'APl': coco_eval.stats[5] * 100,
    }
    
    # AP by frequency (LVIS-specific)
    rare_cats = [cat['id'] for cat in categories if cat.get('frequency', 'f') == 'r']
    common_cats = [cat['id'] for cat in categories if cat.get('frequency', 'f') == 'c']
    freq_cats = [cat['id'] for cat in categories if cat.get('frequency', 'f') == 'f']
    
    gt_cat_set = set(ann['category_id'] for ann in coco_gt_dict['annotations'])
    
    def compute_ap_for_cats(cat_ids, name):
        valid_cats = [c for c in cat_ids if c in gt_cat_set]
        if not valid_cats:
            return 0.0, 0
        coco_eval_sub = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_sub.params.imgIds = list(image_ids)
        coco_eval_sub.params.catIds = valid_cats
        coco_eval_sub.evaluate()
        coco_eval_sub.accumulate()
        if len(coco_eval_sub.stats) == 0:
            return 0.0, len(valid_cats)
        return coco_eval_sub.stats[0] * 100, len(valid_cats)
    
    results['APr'], n_rare = compute_ap_for_cats(rare_cats, 'rare')
    results['APc'], n_common = compute_ap_for_cats(common_cats, 'common')
    results['APf'], n_freq = compute_ap_for_cats(freq_cats, 'frequent')
    results['n_rare_cats'] = n_rare
    results['n_common_cats'] = n_common
    results['n_freq_cats'] = n_freq
    
    # Cleanup
    os.remove(gt_file)
    
    return results


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = Config()
    
    print("=" * 70)
    print("LVIS Zero-Shot Evaluation - Grounding DINO (Jittor)")
    print("=" * 70)
    
    # Load LVIS annotations
    print(f"\n[1/5] Loading LVIS annotations from {args.lvis_ann}...")
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)
    
    images = {img['id']: img for img in lvis_data['images']}
    categories = lvis_data['categories']
    
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)
    
    # Select images with annotations
    image_ids = [img_id for img_id in images.keys() if img_id in img_to_anns]
    
    if args.full:
        args.num_images = len(image_ids)
    elif args.num_images > 0:
        image_ids = image_ids[:args.num_images]
    
    print(f"  Total images in LVIS val: {len(images)}")
    print(f"  Images with annotations: {len(img_to_anns)}")
    print(f"  Evaluating on: {len(image_ids)} images")
    print(f"  Total categories: {len(categories)}")
    
    # Load tokenizer and build category batches
    print(f"\n[2/5] Building category batches (batch_size={args.batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, 'models/bert-base-uncased'))
    batch_info = build_category_batches(categories, tokenizer, args.batch_size)
    
    print(f"  Number of batches: {len(batch_info)}")
    print(f"  Max tokens per batch: {max(b['num_tokens'] for b in batch_info)}")
    
    # Load model
    print(f"\n[3/5] Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    
    # Run inference
    print(f"\n[4/5] Running inference on {len(image_ids)} images...")
    print(f"  (Each image requires {len(batch_info)} forward passes)")
    
    start_time = time.time()
    all_predictions = []
    
    for img_id in tqdm(image_ids, desc="Processing images"):
        img_info = images[img_id]
        
        # Get filename
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"
        
        img_path = os.path.join(args.image_dir, file_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        img_tensor, _ = preprocess_image(image, config)
        
        # Run batched inference
        img_preds = run_inference_batched(
            model, img_tensor, batch_info, 
            (orig_w, orig_h), args.num_select
        )
        
        # Add image_id to predictions
        for pred in img_preds:
            pred['image_id'] = int(img_id)
        all_predictions.extend(img_preds)
        
        # Cleanup
        del img_tensor
        jt.sync_all()
        jt.gc()
    
    elapsed = time.time() - start_time
    print(f"\n  Inference completed in {elapsed/60:.1f} minutes")
    print(f"  Average time per image: {elapsed/len(image_ids):.1f} seconds")
    print(f"  Total predictions: {len(all_predictions)}")
    
    # Evaluate
    print(f"\n[5/5] Running COCO-style evaluation...")
    results = evaluate_with_pycocotools(
        all_predictions, lvis_data, set(image_ids), categories, args.output_dir
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (True Zero-Shot)")
    print("=" * 70)
    print(f"  Images evaluated: {len(image_ids)}")
    print(f"  Categories: {len(categories)} (r:{results['n_rare_cats']}, c:{results['n_common_cats']}, f:{results['n_freq_cats']} with GT)")
    print("-" * 70)
    print(f"  AP   (IoU=0.50:0.95): {results['AP']:.1f}%  (target: 25.6%)")
    print(f"  AP50 (IoU=0.50):      {results['AP50']:.1f}%")
    print(f"  AP75 (IoU=0.75):      {results['AP75']:.1f}%")
    print("-" * 70)
    print(f"  APr  (rare):          {results['APr']:.1f}%  (target: 14.4%)")
    print(f"  APc  (common):        {results['APc']:.1f}%  (target: 19.6%)")
    print(f"  APf  (frequent):      {results['APf']:.1f}%  (target: 32.2%)")
    print("-" * 70)
    print(f"  APs  (small):         {results['APs']:.1f}%")
    print(f"  APm  (medium):        {results['APm']:.1f}%")
    print(f"  APl  (large):         {results['APl']:.1f}%")
    print("=" * 70)
    
    # Save results
    results['num_images'] = len(image_ids)
    results['inference_time_seconds'] = elapsed
    results['timestamp'] = datetime.now().isoformat()
    
    results_file = os.path.join(args.output_dir, 'lvis_zeroshot_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {os.path.join(args.output_dir, 'lvis_predictions.json')}")


if __name__ == '__main__':
    main()


