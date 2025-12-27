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

# Set GPU before importing jittor (check if GPU is available)
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# Check if CUDA_VISIBLE_DEVICES is set to empty (CPU mode) or invalid GPU
if cuda_visible == '' or not cuda_visible.strip():
    cuda_visible = ''  # Force CPU mode
else:
    # Check if the specified GPU exists
    try:
        gpu_id = int(cuda_visible)
        # Simple check: try to get GPU info
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
        available_gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        if gpu_id not in available_gpus:
            print(f"Warning: GPU {gpu_id} not available. Available GPUs: {available_gpus}")
            cuda_visible = ''  # Fall back to CPU
    except:
        print("Warning: Could not check GPU availability, falling back to CPU")
        cuda_visible = ''

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible

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

# Set CUDA flag based on GPU availability
if cuda_visible == '':
    jt.flags.use_cuda = 0
    print("Using CPU mode")
else:
    jt.flags.use_cuda = 1
    print(f"Using GPU mode (GPU {cuda_visible})")


def parse_args():
    parser = argparse.ArgumentParser(description='LVIS Zero-Shot Evaluation')
    parser.add_argument('--checkpoint', type=str, 
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to Jittor checkpoint')
    parser.add_argument('--lvis_ann', type=str,
                        default='../data/lvis_v1_val.json',
                        help='Path to LVIS annotation file')
    parser.add_argument('--image_dir', type=str,
                        default='../val2017',
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
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def precompute_text_embeddings(model, batch_info):
    """Precompute text embeddings for all batches to avoid recomputing per image."""
    print("Precomputing text embeddings for all category batches...")
    text_cache = []
    for i, batch in enumerate(batch_info):
        print(f"  Batch {i+1}/{len(batch_info)}: {len(batch['cat_ids'])} categories")
        with jt.no_grad():
            text_dict = model.encode_text([batch['caption']])
        text_cache.append(text_dict)
    print("Text embedding precomputation completed.")
    return text_cache


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


def run_inference_batched_optimized(model, img_tensor, batch_info, text_cache, orig_size, num_select=300):
    """Optimized inference using cached vision features and precomputed text embeddings."""
    orig_w, orig_h = orig_size
    all_predictions = []

    # Extract vision features once for this image
    with jt.no_grad():
        vision_features = model.encode_image(img_tensor)

    for batch_idx, batch in enumerate(batch_info):
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']

        # Use cached text features
        with jt.no_grad():
            outputs = model(
                img_tensor,
                text_dict=text_cache[batch_idx],
                vision_features=vision_features
            )

        pred_logits = outputs['pred_logits'][0].numpy()
        pred_boxes = outputs['pred_boxes'][0].numpy()

        # Vectorized computation: Convert logits to probabilities and map to labels
        prob_to_token = 1 / (1 + np.exp(-pred_logits))  # sigmoid
        prob_to_label = prob_to_token @ positive_map_np.T

        # Vectorized filtering and box conversion
        threshold = 0.001
        q_idxs, c_idxs = np.where(prob_to_label >= threshold)
        scores = prob_to_label[q_idxs, c_idxs]

        if len(scores) == 0:
            continue

        # Vectorized box conversion
        boxes = pred_boxes[q_idxs]
        cx, cy, w, h = boxes.T

        # Convert to absolute coordinates (vectorized)
        x1 = np.clip((cx - w/2) * orig_w, 0, orig_w - w*orig_w)
        y1 = np.clip((cy - h/2) * orig_h, 0, orig_h - h*orig_h)
        bw = np.clip(w * orig_w, 0, orig_w - x1)
        bh = np.clip(h * orig_h, 0, orig_h - y1)

        # Filter valid boxes
        valid = (bw > 0) & (bh > 0)
        q_idxs = q_idxs[valid]
        c_idxs = c_idxs[valid]
        scores = scores[valid]
        x1 = x1[valid]
        y1 = y1[valid]
        bw = bw[valid]
        bh = bh[valid]

        # Build predictions (vectorized)
        batch_preds = [
            {
                'category_id': int(batch_cat_ids[c_idx]),
                'bbox': [float(x), float(y), float(w_), float(h_)],
                'score': float(score)
            }
            for x, y, w_, h_, score, c_idx in zip(x1, y1, bw, bh, scores, c_idxs)
        ]

        all_predictions.extend(batch_preds)

    # Keep top-k predictions
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    return all_predictions[:num_select]


def run_inference_batched(model, img_tensor, batch_info, orig_size, num_select=300):
    """Legacy function for backward compatibility - use run_inference_batched_optimized instead."""
    return run_inference_batched_optimized(model, img_tensor, batch_info, None, orig_size, num_select)


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
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_info = build_category_batches(categories, tokenizer, args.batch_size)
    
    print(f"  Number of batches: {len(batch_info)}")
    print(f"  Max tokens per batch: {max(b['num_tokens'] for b in batch_info)}")
    
    # Load model
    print(f"\n[3/5] Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    # Precompute text embeddings (OPTIMIZATION)
    print(f"\n[3.5/5] Precomputing text embeddings...")
    text_cache = precompute_text_embeddings(model, batch_info)

    # Run inference
    print(f"\n[4/5] Running OPTIMIZED inference on {len(image_ids)} images...")
    print(f"  (Vision features cached per image, text embeddings precomputed)")
    print(f"  (Each image now requires only {len(batch_info)} decoder passes)")

    start_time = time.time()
    all_predictions = []
    SYNC_INTERVAL = 10  # Sync every 10 images instead of every image

    for img_idx, img_id in enumerate(tqdm(image_ids, desc="Processing images")):
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

        # Run OPTIMIZED batched inference with cached features
        img_preds = run_inference_batched_optimized(
            model, img_tensor, batch_info, text_cache,
            (orig_w, orig_h), args.num_select
        )

        # Add image_id to predictions
        for pred in img_preds:
            pred['image_id'] = int(img_id)
        all_predictions.extend(img_preds)

        # Periodic cleanup (reduced frequency)
        if (img_idx + 1) % SYNC_INTERVAL == 0:
            jt.sync_all()
            jt.gc()

        # Cleanup tensor
        del img_tensor

    # Final cleanup
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
