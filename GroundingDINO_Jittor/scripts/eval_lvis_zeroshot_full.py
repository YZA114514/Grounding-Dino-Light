#!/usr/bin/env python
"""
LVIS Zero-Shot Evaluation Script for Grounding DINO (Jittor)

This script evaluates the Jittor implementation of Grounding DINO on the LVIS dataset
using true zero-shot evaluation (all 1203 categories processed in batches).

Target metrics (from paper):
    AP: 25.6, APr: 14.4, APc: 19.6, APf: 32.2

Usage:
    python scripts/eval_lvis_zeroshot_full.py --num_images 100 --batch_size 60
    python scripts/eval_lvis_zeroshot_full.py --full  # Full validation set (~17K images)
"""

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime

# Force HuggingFace offline mode (skip network checks for cached models)
os.environ['HF_HUB_OFFLINE'] = '1'

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


def archive_previous_run(output_dir):
    """Archive previous run outputs to timestamped folder if they exist."""
    predictions_file = os.path.join(output_dir, 'predictions.jsonl')
    progress_file = os.path.join(output_dir, 'progress.json')

    # Check if there's a previous run to archive
    if os.path.exists(predictions_file) or os.path.exists(progress_file):
        # Create archive folder with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(output_dir, f'archive_{timestamp}')
        os.makedirs(archive_dir, exist_ok=True)

        # Move files to archive
        files_to_archive = [
            'predictions.jsonl',
            'progress.json',
            'lvis_predictions.json',
            'lvis_zeroshot_results.json'
        ]
        moved = []
        for fname in files_to_archive:
            src = os.path.join(output_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(archive_dir, fname)
                os.rename(src, dst)
                moved.append(fname)

        if moved:
            print(f"  Archived previous run to: {archive_dir}")
            print(f"  Files moved: {', '.join(moved)}")


def setup_logging(output_dir):
    """Set up logging to both console and file."""
    log_file = os.path.join(output_dir, 'eval.log')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(f"Logging to: {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='LVIS Zero-Shot Evaluation')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to Jittor checkpoint')
    parser.add_argument('--lvis_ann', type=str,
                        default='../data/lvis_v1_val.json',
                        help='Path to LVIS annotation file')
    parser.add_argument('--image_dir', type=str,
                        default='../LVIS/val',
                        help='Path to LVIS validation images')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to evaluate (0 for all)')
    parser.add_argument('--full', action='store_true',
                        help='Evaluate on full validation set')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='Number of categories per batch (to fit BERT 512 token limit)')
    parser.add_argument('--box_threshold', type=float,      default=0.1, help='Box score threshold (original: 0.1)')
    parser.add_argument('--num_select', type=int, default=1000000,
                        help='[DEPRECATED] Number of top predictions to keep per image. '
                             'This parameter is obsolete and can harm AP output. '
                             'It will be removed in future versions.')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index for image subset (for multi-GPU parallel processing)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for image subset (for multi-GPU parallel processing)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use in parallel (coordinator mode)')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker ID for tqdm position (for multi-GPU display)')
    parser.add_argument('--checkpoint_interval', type=int, default=250,
                        help='Save checkpoint every N images (default: 250)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint if available')
    parser.add_argument('--eval_interval', type=int, default=0,
                        help='Evaluate AP every N images (0 = disabled, adds overhead)')
    parser.add_argument('--ultra_optimized', action='store_true',
                        help='Use ultra-optimized inference (reduce GPU-CPU syncs from 26 to 2-3 per image)')
    args = parser.parse_args()

    # Deprecation warning for num_select
    if hasattr(args, 'num_select') and args.num_select != 1000000:
        import warnings
        warnings.warn(
            "--num_select is deprecated and can harm AP output. "
            "The parameter will be removed in future versions. "
            "All predictions above threshold are now kept by default.",
            DeprecationWarning,
            stacklevel=2
        )
        print("WARNING: --num_select is deprecated. All predictions above threshold will be kept.")

    return args


def load_checkpoint(output_dir):
    """Load checkpoint and return starting index."""
    progress_file = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_idx', 0) + 1
        timestamp = progress.get('timestamp', 'unknown')
        print(f"  Found checkpoint: last processed image index {progress.get('last_idx', 0)} (at {timestamp})")
        print(f"  Resuming from image index {start_idx}")
        return start_idx
    return 0


def save_checkpoint(output_dir, last_idx):
    """Save progress checkpoint with atomic write."""
    progress_file = os.path.join(output_dir, 'progress.json')
    tmp_file = progress_file + '.tmp'

    progress = {
        'last_idx': last_idx,
        'timestamp': datetime.now().isoformat(),
        'total_processed': last_idx + 1
    }

    # Atomic write: write to .tmp then rename
    with open(tmp_file, 'w') as f:
        json.dump(progress, f, indent=2)
    os.rename(tmp_file, progress_file)


def write_predictions_to_jsonl(output_dir, predictions):
    """Append predictions to JSONL file."""
    jsonl_file = os.path.join(output_dir, 'predictions.jsonl')
    with open(jsonl_file, 'a') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
        f.flush()


def load_predictions_from_jsonl(output_dir):
    """Load all predictions from JSONL file."""
    jsonl_file = os.path.join(output_dir, 'predictions.jsonl')
    if not os.path.exists(jsonl_file):
        return []

    predictions = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line.strip()))
    return predictions


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


def build_category_batches(categories, tokenizer, batch_size=60, max_text_len=256):
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

def run_inference_batched_optimized(model, img_tensor, batch_info, text_cache, orig_size, box_threshold=0.1, num_select=None):
    """Optimized inference using cached projection features and precomputed text embeddings.

    Returns all predictions above threshold. The num_select parameter is deprecated and ignored.
    """
    orig_w, orig_h = orig_size
    all_predictions = []

    # Extract projection features once for this image (backbone + projection only)
    with jt.no_grad():
        projected_features = model.encode_image_projection(img_tensor)

    # Run encoder + decoder with real text for each batch
    for batch_idx, batch in enumerate(batch_info):
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']

        # Encode text fresh for this batch (required for proper text-vision fusion)
        with jt.no_grad():
            if text_cache is None:
                # Fresh text encoding for each batch
                text_dict = model.encode_text([batch['caption']])
            else:
                # Use cached text (legacy mode)
                text_dict = text_cache[batch_idx]

            outputs = model(
                img_tensor,
                text_dict=text_dict,
                vision_features=projected_features
            )

        # Vectorized computation: Convert logits to probabilities and map to labels (GPU sigmoid + bmm):
        prob_to_token_gpu = jt.sigmoid(outputs['pred_logits'][0])  # GPU sigmoid
        positive_map_gpu = jt.array(positive_map_np.T).unsqueeze(0)  # (1, 95, num_cats)
        prob_to_label = jt.nn.bmm(
            prob_to_token_gpu.unsqueeze(0),  # (1, 900, 95)
            positive_map_gpu                  # (1, 95, num_cats)
        ).squeeze(0).numpy()                  # (900, num_cats) → numpy
        pred_boxes_gpu = outputs['pred_boxes'][0]

        # Vectorized filtering and box conversion
        threshold = box_threshold
        q_idxs, c_idxs = np.where(prob_to_label >= threshold)
        scores = prob_to_label[q_idxs, c_idxs]

        if len(scores) == 0:
            continue

        # Vectorized box conversion
        boxes = pred_boxes_gpu[q_idxs].numpy()
        cx, cy, w, h = boxes.T

        # Convert to absolute coordinates (vectorized)
        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Convert to [x, y, w, h] format
        bw = x2 - x1
        bh = y2 - y1

        # Filter valid boxes
        valid = (bw > 0) & (bh > 0)
        q_idxs = q_idxs[valid]
        c_idxs = c_idxs[valid]
        scores = scores[valid]
        x1 = x1[valid]
        y1 = y1[valid]
        bw = bw[valid]
        bh = bh[valid]

        cat_ids_arr = np.array(batch_cat_ids)[c_idxs]
        batch_preds = [
            {'category_id': cid, 'bbox': box, 'score': sc}
            for cid, box, sc in zip(
                cat_ids_arr.tolist(),
                np.column_stack([x1, y1, bw, bh]).tolist(),
                scores.tolist()
            )
        ]

        all_predictions.extend(batch_preds)

    # Return all predictions above threshold (num_select parameter is deprecated)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    return all_predictions

def run_inference_batched_ultra_optimized(model, img_tensor, batch_info, text_cache, orig_size, box_threshold=0.1, num_select=None):
    """Ultra-optimized inference: minimize GPU-CPU syncs by keeping everything on GPU until final sync.

    Returns all predictions above threshold. The num_select parameter is deprecated and ignored.
    """
    orig_w, orig_h = orig_size

    # Accumulators (stay on GPU)
    all_scores = []
    all_boxes = []
    all_cat_ids = []

    with jt.no_grad():
        projected_features = model.encode_image_projection(img_tensor)

    for batch_idx, batch in enumerate(batch_info):
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']

        with jt.no_grad():
            text_dict = model.encode_text([batch['caption']])
            outputs = model(
                img_tensor,
                text_dict=text_dict,
                vision_features=projected_features
            )

        # GPU: sigmoid + bmm
        prob_to_token_gpu = jt.sigmoid(outputs['pred_logits'][0])
        positive_map_gpu = jt.array(positive_map_np.T).unsqueeze(0)
        prob_to_label_gpu = jt.nn.bmm(
            prob_to_token_gpu.unsqueeze(0),
            positive_map_gpu
        ).squeeze(0)  # (900, num_cats) - STAY ON GPU

        pred_boxes_gpu = outputs['pred_boxes'][0]  # (900, 4)

        # GPU: threshold filtering
        mask = prob_to_label_gpu >= box_threshold
        indices = jt.where(mask)
        q_idxs = indices[0]
        c_idxs = indices[1]

        if q_idxs.shape[0] == 0:
            continue

        scores = prob_to_label_gpu[q_idxs, c_idxs]
        boxes = pred_boxes_gpu[q_idxs]  # (N, 4) - STAY ON GPU

        # GPU: coordinate conversion
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = jt.clamp((cx - w/2) * orig_w, 0, orig_w)
        y1 = jt.clamp((cy - h/2) * orig_h, 0, orig_h)
        x2 = jt.clamp((cx + w/2) * orig_w, 0, orig_w)
        y2 = jt.clamp((cy + h/2) * orig_h, 0, orig_h)
        bw = x2 - x1
        bh = y2 - y1

        # GPU: valid filter
        valid = (bw > 0) & (bh > 0)
        scores = scores[valid]
        x1 = x1[valid]
        y1 = y1[valid]
        bw = bw[valid]
        bh = bh[valid]
        c_idxs = c_idxs[valid]

        # Map category indices to IDs (on GPU)
        cat_ids_gpu = jt.array(batch_cat_ids)[c_idxs]

        # Accumulate (still GPU tensors)
        all_scores.append(scores)
        all_boxes.append(jt.stack([x1, y1, bw, bh], dim=1))
        all_cat_ids.append(cat_ids_gpu)

    if not all_scores:
        return []

    # Concatenate on GPU
    all_scores = jt.concat(all_scores)
    all_boxes = jt.concat(all_boxes)
    all_cat_ids = jt.concat(all_cat_ids)

    # Return all predictions above threshold (num_select parameter is deprecated)
    # SINGLE SYNC: transfer all results to CPU
    scores_np = all_scores.numpy()
    boxes_np = all_boxes.numpy()
    cat_ids_np = all_cat_ids.numpy().astype(int)

    # Build output dicts (CPU)
    return [
        {'category_id': cid, 'bbox': box, 'score': sc}
        for cid, box, sc in zip(
            cat_ids_np.tolist(),
            boxes_np.tolist(),
            scores_np.tolist()
        )
    ]


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

    # Build annotations list FIRST
    annotations_list = []
    ann_id = 1
    for img_id in image_ids:
        for ann in img_to_anns.get(img_id, []):
            annotations_list.append({
                'id': ann_id,
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['bbox'][2] * ann['bbox'][3],
                'iscrowd': 0
            })
            ann_id += 1

    # Get categories with GT in this subset
    gt_cat_ids = set(ann['category_id'] for ann in annotations_list)

    # Build GT dict with ONLY categories that have annotations
    coco_gt_dict = {
        'info': {'description': 'LVIS v1 validation subset', 'date_created': datetime.now().isoformat()},
        'licenses': [],
        'images': [img for img in lvis_data['images'] if img['id'] in image_ids],
        'annotations': annotations_list,
        'categories': [cat for cat in categories if cat['id'] in gt_cat_ids]  # ← FIXED
    }
    
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
        coco_eval_sub.summarize()
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

def coordinator_main(args):
    """Coordinator function that spawns workers for multi-GPU evaluation."""
    import subprocess

    print(f"Multi-GPU evaluation with {args.n_gpus} GPUs")

    # Archive previous run (unless resuming)
    if not args.resume:
        archive_previous_run(args.output_dir)

    # Load LVIS annotations to get total image count
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)

    images = {img['id']: img for img in lvis_data['images']}
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Select images with annotations
    image_ids = [img_id for img_id in images.keys() if img_id in img_to_anns]

    if args.full:
        args.num_images = len(image_ids)
    elif args.num_images > 0:
        image_ids = image_ids[:args.num_images]

    total_images = len(image_ids)
    chunk_size = total_images // args.n_gpus

    print(f"Total images to evaluate: {total_images}")
    print(f"Images per GPU: {chunk_size}")

    # Spawn workers
    procs = []
    for gpu_id in range(args.n_gpus):
        start_idx = gpu_id * chunk_size
        end_idx = start_idx + chunk_size if gpu_id < args.n_gpus - 1 else total_images

        print(f"Spawning GPU {gpu_id}: images [{start_idx}:{end_idx}] ({end_idx - start_idx} images)")

        # Build command for worker
        cmd = [
            sys.executable,  # Python executable
            __file__,        # This script
            '--gpu', str(gpu_id),
            '--start_idx', str(start_idx),
            '--end_idx', str(end_idx),
            '--worker_id', str(gpu_id),
            '--output_dir', args.output_dir + f'/gpu{gpu_id}',
            '--checkpoint', args.checkpoint,
            '--lvis_ann', args.lvis_ann,
            '--image_dir', args.image_dir,
            '--batch_size', str(args.batch_size),
            '--num_select', str(args.num_select),
            '--checkpoint_interval', str(args.checkpoint_interval),
        ]

        if args.ultra_optimized:
            cmd.append('--ultra_optimized')

        if args.full:
            cmd.append('--full')
        if args.resume:
            cmd.append('--resume')

        # Spawn worker process (non-blocking)
        proc = subprocess.Popen(
            cmd,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
        )
        procs.append(proc)

    print(f"Running {args.n_gpus} workers in parallel...")
    for p in procs:
        p.wait()

    print("All workers complete. Merging results...")

    # Merge results from all workers (concatenate JSONL files)
    merged_jsonl_file = os.path.join(args.output_dir, 'predictions.jsonl')
    total_predictions = 0

    with open(merged_jsonl_file, 'w') as outfile:
        for gpu_id in range(args.n_gpus):
            worker_output_dir = args.output_dir + f'/gpu{gpu_id}'
            worker_jsonl_file = os.path.join(worker_output_dir, 'predictions.jsonl')

            if os.path.exists(worker_jsonl_file):
                worker_predictions = 0
                with open(worker_jsonl_file, 'r') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            worker_predictions += 1
                total_predictions += worker_predictions
                print(f"  GPU {gpu_id}: {worker_predictions} predictions")
            else:
                print(f"  Warning: No predictions file found for GPU {gpu_id}")

    # Convert JSONL to JSON array for evaluation
    all_predictions = load_predictions_from_jsonl(args.output_dir)
    print(f"  Total predictions: {total_predictions}")

    # Save merged results in JSON format too
    merged_pred_file = os.path.join(args.output_dir, 'lvis_predictions.json')
    with open(merged_pred_file, 'w') as f:
        json.dump(all_predictions, f)

    # Run final evaluation on merged results
    print(f"\nRunning final evaluation on merged predictions...")
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

    results = evaluate_with_pycocotools(
        all_predictions, lvis_data, set(image_ids), categories, args.output_dir
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (True Zero-Shot) - Multi-GPU")
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
    results['n_gpus'] = args.n_gpus
    results['timestamp'] = datetime.now().isoformat()

    results_file = os.path.join(args.output_dir, 'lvis_zeroshot_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to: {results_file}")


def main():
    args = parse_args()

    # Check if we should run as coordinator (multi-GPU mode)
    if args.n_gpus > 1:
        coordinator_main(args)
        return

    # Single GPU mode (original behavior)
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Archive previous run (unless resuming)
    if not args.resume:
        archive_previous_run(args.output_dir)

    # Setup logging to both console and file
    setup_logging(args.output_dir)

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

    # Apply range selection (start_idx/end_idx takes priority over num_images)
    if args.start_idx > 0 or args.end_idx is not None:
        # Explicit range overrides --num_images
        end_idx = args.end_idx if args.end_idx is not None else len(image_ids)
        image_ids = image_ids[args.start_idx:end_idx]
        start_idx = args.start_idx
    elif args.full:
        start_idx = 0
    elif args.num_images > 0:
        image_ids = image_ids[:args.num_images]
        start_idx = 0
    else:
        start_idx = 0

    print(f"  Total images in LVIS val: {len(images)}")
    print(f"  Images with annotations: {len(img_to_anns)}")
    print(f"  Evaluating images [{start_idx}:{start_idx + len(image_ids)}] (subset: {len(image_ids)} images)")
    print(f"  Total categories: {len(categories)}")
    
    # Load tokenizer and build category batches
    print(f"\n[2/5] Building category batches (batch_size={args.batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_info = build_category_batches(categories, tokenizer, args.batch_size)
    
    print(f"  Number of batches: {len(batch_info)}")
    print(f"  Max tokens per batch: {max(b['num_tokens'] for b in batch_info)}")
    # Log positive_map sum for reproducibility verification
    print(f"  batch_info[0]['positive_map'].sum() = {batch_info[0]['positive_map'].sum()}")
    
    # Load model
    print(f"\n[3/5] Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    # Check for existing checkpoint
    print(f"\n[3.5/5] Checking for checkpoints in {args.output_dir}...")
    resume_start_idx = 0
    if args.resume or os.path.exists(os.path.join(args.output_dir, 'progress.json')):
        resume_start_idx = load_checkpoint(args.output_dir)
        print(f"  Resuming from image index {resume_start_idx}")
    else:
        print("  No checkpoint found, starting fresh")

    # Text embeddings will be encoded fresh for each batch (required for proper text-vision fusion)

    # Run inference
    if args.ultra_optimized:
        inference_func = run_inference_batched_ultra_optimized
        print(f"\n[4/5] Running ULTRA-OPTIMIZED inference on {len(image_ids)} images...")
        print(f"  (GPU-CPU syncs reduced from 26 to 2-3 per image)")
        print(f"  (All operations kept on GPU until final sync)")
        print(f"  (Expected 15-25% speedup)")
    else:
        inference_func = run_inference_batched_optimized
        print(f"\n[4/5] Running OPTIMIZED inference on {len(image_ids)} images...")
        print(f"  (Vision features cached per image, text encoded fresh per batch)")
    print(f"  (Each image now requires only {len(batch_info)} encoder+decoder passes)")
    print(f"  (Checkpointing every {args.checkpoint_interval} images)")

    start_time = time.time()
    processed_count = 0

    # Resume from checkpoint if available
    remaining_image_ids = image_ids[resume_start_idx:]
    print(f"  Processing {len(remaining_image_ids)} images (starting from index {resume_start_idx})")

    # Pre-filter to only include images that actually exist
    print(f"  Pre-filtering images to check which files exist...")
    existing_image_ids = []
    for img_id in tqdm(remaining_image_ids, desc="Checking images"):
        img_info = images[img_id]

        # Get filename
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"

        img_path = os.path.join(args.image_dir, file_name)

        if os.path.exists(img_path):
            existing_image_ids.append(img_id)

    print(f"  Found {len(existing_image_ids)} existing images out of {len(remaining_image_ids)}")
    remaining_image_ids = existing_image_ids

    if len(remaining_image_ids) == 0:
        print("  ERROR: No valid images found in specified range!")
        print(f"  Check that image directory '{args.image_dir}' contains the correct images")
        print(f"  For LVIS val, you may need to download images to LVIS/val/ directory")
        sys.exit(1)

    for img_idx, img_id in enumerate(tqdm(
        remaining_image_ids,
        initial=resume_start_idx,
        total=len(image_ids),
        desc=f"GPU {args.gpu}",
        position=args.worker_id,
        leave=True
    ), start=resume_start_idx):
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

        # Run batched inference with selected optimization level
        img_preds = inference_func(
            model, img_tensor, batch_info, None,  # Text encoded fresh per batch
            (orig_w, orig_h), args.box_threshold, args.num_select
        )

        # Add image_id to predictions
        for pred in img_preds:
            pred['image_id'] = int(img_id)

        # Write predictions to JSONL file
        write_predictions_to_jsonl(args.output_dir, img_preds)
        processed_count += 1

        # Save checkpoint every N images
        if processed_count % args.checkpoint_interval == 0:
            save_checkpoint(args.output_dir, img_idx)
            print(f"  Checkpoint saved at image {img_idx + 1}")

        # Real-time AP evaluation (optional, expensive)
        if args.eval_interval > 0 and processed_count % args.eval_interval == 0 and processed_count > 0:
            try:
                # Load current predictions
                current_predictions = load_predictions_from_jsonl(args.output_dir)
                # Evaluate on current subset of images processed so far
                current_image_ids = set(pred['image_id'] for pred in current_predictions)

                # Run evaluation on current predictions
                partial_results = evaluate_with_pycocotools(
                    current_predictions, lvis_data, current_image_ids, categories, args.output_dir
                )

                # Log partial results
                logging.info(f"[Partial @ {processed_count}/{len(remaining_image_ids)} images] "
                           f"AP: {partial_results['AP']:.2f}%, "
                           f"AP50: {partial_results['AP50']:.2f}%, "
                           f"AP75: {partial_results['AP75']:.2f}%")

                # Save partial results
                partial_results_file = os.path.join(args.output_dir, f'results_partial_{processed_count:06d}.json')
                with open(partial_results_file, 'w') as f:
                    json.dump({
                        'images_processed': processed_count,
                        'total_images': len(remaining_image_ids),
                        'results': partial_results,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                print(f"  [Partial AP: {partial_results['AP']:.2f}% at {processed_count} images]")

            except Exception as e:
                logging.warning(f"Failed to compute partial evaluation: {e}")

        # Cleanup tensor
        del img_tensor

    # Final checkpoint
    if processed_count > 0:
        save_checkpoint(args.output_dir, len(image_ids) - 1)
        print(f"  Final checkpoint saved")

    # Final cleanup
    # jt.sync_all()  # Not available in Jittor
    # jt.gc()  # Not available in Jittor

    elapsed = time.time() - start_time
    print(f"\n  Inference completed in {elapsed/60:.1f} minutes")
    if processed_count > 0:
        print(f"  Average time per image: {elapsed/processed_count:.1f} seconds")
    else:
        print("  Warning: No images were processed")

    # Load all predictions from JSONL and convert to JSON array
    print(f"\n  Loading predictions from JSONL file...")
    all_predictions = load_predictions_from_jsonl(args.output_dir)
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
    results['ultra_optimized'] = args.ultra_optimized
    results['timestamp'] = datetime.now().isoformat()
    
    results_file = os.path.join(args.output_dir, 'lvis_zeroshot_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {os.path.join(args.output_dir, 'lvis_predictions.json')}")


if __name__ == '__main__':
    main()
