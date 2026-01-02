#!/usr/bin/env python
"""
OWL-ViT LVIS Zero-Shot Evaluation Script

This script evaluates OWL-ViT (Open-World Localization with Vision Transformer) on the LVIS dataset
using true zero-shot evaluation (all 1203 categories processed in batches).

Uses HuggingFace transformers for OWL-ViT model loading and inference.

Usage:
    python scripts/eval_owlvit_lvis.py --num_images 100 --batch_size 25
    python scripts/eval_owlvit_lvis.py --full  # Full validation set (~17K images)
"""
import sys
import os
import json
import argparse
import time
import logging
from datetime import datetime

# Force HuggingFace offline mode (skip network checks for cached models)
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Import required libraries
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    import torch
    import torch.nn.functional as F
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install: pip install transformers torch torchvision")
    sys.exit(1)


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
    parser = argparse.ArgumentParser(description='OWL-ViT LVIS Zero-Shot Evaluation')
    parser.add_argument('--model_name', type=str,
                        default='google/owlvit-base-patch32',
                        help='HuggingFace OWL-ViT model name')
    parser.add_argument('--use_full_val', action='store_true',
                        help='Use full LVIS val set instead of minival')
    parser.add_argument('--lvis_ann', type=str,
                        default=None,
                        help='Path to LVIS annotation file (auto-detected based on --use_full_val)')
    parser.add_argument('--image_dir', type=str,
                        default=None,
                        help='Path to LVIS validation images (auto-detected based on --use_full_val)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to evaluate (0 for all)')
    parser.add_argument('--full', action='store_true',
                        help='Evaluate on full validation set')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Number of categories per batch (OWL-ViT has smaller limits than GroundingDINO)')
    parser.add_argument('--box_threshold', type=float, default=0.1,
                        help='Box score threshold (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='outputs/owlvit',
                        help='Output directory for results')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index for image subset')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for image subset')
    parser.add_argument('--checkpoint_interval', type=int, default=250,
                        help='Save checkpoint every N images (default: 250)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint if available')
    args = parser.parse_args()

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


def build_category_batches(categories, batch_size=25):
    """Build batches of categories for OWL-ViT processing."""
    num_batches = (len(categories) + batch_size - 1) // batch_size
    batch_info = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(categories))
        batch_cats = categories[start:end]
        batch_cat_names = [cat['name'].lower().replace('_', ' ') for cat in batch_cats]
        batch_cat_ids = [cat['id'] for cat in batch_cats]

        batch_info.append({
            'cat_ids': batch_cat_ids,
            'cat_names': batch_cat_names,
            'num_cats': len(batch_cat_ids)
        })

    return batch_info


def predict_owlvit_batch(model, processor, image_path, batch_cat_ids, text_queries, box_threshold=0.1):
    """
    Run OWL-ViT inference on a single image with batched text queries.

    Args:
        model: OWL-ViT model
        processor: OWL-ViT processor
        image_path: Path to image
        batch_cat_ids: List of category IDs corresponding to text_queries
        text_queries: List of text queries (category names)
        box_threshold: Confidence threshold for detections

    Returns:
        List of predictions in LVIS format
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    # Process inputs
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_object_detection(
        outputs, threshold=box_threshold, target_sizes=[(orig_h, orig_w)]
    )

    # Convert to LVIS format predictions
    all_predictions = []

    for batch_idx, result in enumerate(results):
        boxes = result['boxes']  # [x1, y1, x2, y2] format
        scores = result['scores']
        labels = result['labels']  # 0-indexed within this batch

        for box, score, label in zip(boxes, scores, labels):
            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            # Map batch-local label index to global category ID
            category_id = batch_cat_ids[label]

            prediction = {
                'category_id': category_id,
                'bbox': [x1, y1, w, h],
                'score': score.item()
            }
            all_predictions.append(prediction)

    return all_predictions


def evaluate_with_lvis(predictions, lvis_data, image_ids, categories, output_dir):
    """Run official LVIS evaluation using LVISEval."""
    try:
        from lvis import LVIS, LVISResults, LVISEval
    except ImportError:
        print("LVIS package not found. Please install: pip install lvis")
        print("Falling back to simple evaluation...")
        return evaluate_simple(predictions, lvis_data, image_ids, categories)

    # Build ground truth in LVIS format
    subset_images = [img for img in lvis_data['images'] if img['id'] in image_ids]
    subset_annotations = [ann for ann in lvis_data['annotations'] if ann['image_id'] in image_ids]

    # Create subset GT dict
    lvis_gt_subset = {
        'info': {'description': 'LVIS v1 validation subset', 'date_created': datetime.now().isoformat()},
        'licenses': lvis_data.get('licenses', []),
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': categories
    }

    # Save subset GT and predictions
    gt_file = os.path.join(output_dir, 'lvis_gt_subset.json')
    pred_file = os.path.join(output_dir, 'lvis_predictions.json')

    with open(gt_file, 'w') as f:
        json.dump(lvis_gt_subset, f)
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)

    # Load with LVIS API
    lvis_gt = LVIS(gt_file)
    lvis_dt = LVISResults(lvis_gt, pred_file)

    # Run evaluation
    lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
    lvis_eval.run()
    lvis_eval.print_results()

    # Extract results
    results = {}

    # Get overall metrics
    if hasattr(lvis_eval, 'results'):
        overall_results = lvis_eval.results
        results['AP'] = overall_results.get('AP', 0) * 100
        results['AP50'] = overall_results.get('AP50', 0) * 100
        results['AP75'] = overall_results.get('AP75', 0) * 100
        results['APs'] = overall_results.get('APs', 0) * 100
        results['APm'] = overall_results.get('APm', 0) * 100
        results['APl'] = overall_results.get('APl', 0) * 100
        results['APr'] = overall_results.get('APr', 0) * 100
        results['APc'] = overall_results.get('APc', 0) * 100
        results['APf'] = overall_results.get('APf', 0) * 100
    else:
        # Fallback: try to access eval's internal state
        try:
            results['AP'] = lvis_eval.ap * 100 if hasattr(lvis_eval, 'ap') else 0
            results['AP50'] = lvis_eval.ap50 * 100 if hasattr(lvis_eval, 'ap50') else 0
            results['AP75'] = lvis_eval.ap75 * 100 if hasattr(lvis_eval, 'ap75') else 0
            results['APs'] = lvis_eval.aps * 100 if hasattr(lvis_eval, 'aps') else 0
            results['APm'] = lvis_eval.apm * 100 if hasattr(lvis_eval, 'apm') else 0
            results['APl'] = lvis_eval.apl * 100 if hasattr(lvis_eval, 'apl') else 0
            results['APr'] = lvis_eval.ap_rare * 100 if hasattr(lvis_eval, 'ap_rare') else 0
            results['APc'] = lvis_eval.ap_common * 100 if hasattr(lvis_eval, 'ap_common') else 0
            results['APf'] = lvis_eval.ap_freq * 100 if hasattr(lvis_eval, 'ap_freq') else 0
        except:
            print("Warning: Could not extract detailed metrics from LVISEval, using fallback values")
            results['AP'] = 0
            results['AP50'] = 0
            results['AP75'] = 0
            results['APs'] = 0
            results['APm'] = 0
            results['APl'] = 0
            results['APr'] = 0
            results['APc'] = 0
            results['APf'] = 0

    # Count categories with GT in this subset
    gt_cat_ids = set(ann['category_id'] for ann in subset_annotations)
    results['n_rare_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'r'])
    results['n_common_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'c'])
    results['n_freq_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'f'])

    # Cleanup
    os.remove(gt_file)

    return results


def evaluate_simple(predictions, lvis_data, image_ids, categories):
    """Simple evaluation fallback when LVIS package is not available."""
    print("Using simple evaluation (no LVIS package)")

    # Group predictions by image
    img_to_preds = defaultdict(list)
    for pred in predictions:
        img_to_preds[pred['image_id']].append(pred)

    # Group GT by image
    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        if ann['image_id'] in image_ids:
            img_to_anns[ann['image_id']].append(ann)

    total_tp = total_fp = total_fn = 0

    for img_id in image_ids:
        gt_anns = img_to_anns[img_id]
        pred_anns = img_to_preds[img_id]

        # Simple IoU matching (not official LVIS evaluation)
        tp = min(len(gt_anns), len(pred_anns))  # Approximation
        fp = max(0, len(pred_anns) - len(gt_anns))
        fn = max(0, len(gt_anns) - len(pred_anns))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        'AP': f1 * 100,  # Approximation
        'precision': precision * 100,
        'recall': recall * 100,
        'n_rare_cats': len([c for c in categories if c.get('frequency') == 'r']),
        'n_common_cats': len([c for c in categories if c.get('frequency') == 'c']),
        'n_freq_cats': len([c for c in categories if c.get('frequency') == 'f'])
    }


def main():
    args = parse_args()

    # Auto-detect paths based on --use_full_val flag
    if args.use_full_val:
        if args.lvis_ann is None:
            args.lvis_ann = '../LVIS/lvis_v1_val.json'
        if args.image_dir is None:
            args.image_dir = '../LVIS/val'
    else:
        if args.lvis_ann is None:
            args.lvis_ann = '../LVIS/minival/lvis_v1_minival.json'
        if args.image_dir is None:
            args.image_dir = '../LVIS/minival'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Archive previous run (unless resuming)
    if not args.resume:
        archive_previous_run(args.output_dir)

    # Setup logging
    setup_logging(args.output_dir)

    print("=" * 70)
    print("OWL-ViT LVIS Zero-Shot Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Output dir: {args.output_dir}")

    # Load LVIS annotations
    print(f"\n[1/4] Loading LVIS annotations from {args.lvis_ann}...")
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)

    images = {img['id']: img for img in lvis_data['images']}
    categories = lvis_data['categories']

    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Select images with annotations
    image_ids = [img_id for img_id in images.keys() if img_id in img_to_anns]

    # Apply range selection
    if args.start_idx > 0 or args.end_idx is not None:
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

    print(f"  Total images in LVIS: {len(images)}")
    print(f"  Images with annotations: {len(img_to_anns)}")
    print(f"  Evaluating images [{start_idx}:{start_idx + len(image_ids)}] (subset: {len(image_ids)} images)")
    print(f"  Total categories: {len(categories)}")

    # Build category batches
    print(f"\n[2/4] Building category batches (batch_size={args.batch_size})...")
    batch_info = build_category_batches(categories, args.batch_size)
    print(f"  Number of batches: {len(batch_info)}")
    print(f"  Categories per batch: {[b['num_cats'] for b in batch_info]}")

    # Load OWL-ViT model
    print(f"\n[3/4] Loading OWL-ViT model: {args.model_name}...")
    try:
        processor = OwlViTProcessor.from_pretrained(args.model_name)
        model = OwlViTForObjectDetection.from_pretrained(args.model_name)
        model.cuda().eval()
        print("  Model loaded successfully")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Make sure you have run: pip install transformers")
        sys.exit(1)

    # Check for existing checkpoint
    print(f"\n[3.5/4] Checking for checkpoints in {args.output_dir}...")
    resume_start_idx = 0
    if args.resume or os.path.exists(os.path.join(args.output_dir, 'progress.json')):
        resume_start_idx = load_checkpoint(args.output_dir)
        print(f"  Resuming from image index {resume_start_idx}")
    else:
        print("  No checkpoint found, starting fresh")

    # Run inference
    print(f"\n[4/4] Running OWL-ViT inference on {len(image_ids)} images...")
    print(f"  (Processing {len(batch_info)} category batches per image)")
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
        print(f"  Check that image directory '{args.image_dir}' contains the correct lvis/minival images")
        sys.exit(1)

    for img_idx, img_id in enumerate(tqdm(
        remaining_image_ids,
        initial=resume_start_idx,
        total=len(image_ids),
        desc="OWL-ViT",
        position=0,
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

        # Run inference on all category batches
        img_preds = []
        for batch in batch_info:
            batch_preds = predict_owlvit_batch(
                model, processor, img_path, batch['cat_ids'], batch['cat_names'], args.box_threshold
            )
            img_preds.extend(batch_preds)

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

    # Final checkpoint
    if processed_count > 0:
        save_checkpoint(args.output_dir, len(image_ids) - 1)
        print(f"  Final checkpoint saved")

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

    # Save merged results in JSON format
    merged_pred_file = os.path.join(args.output_dir, 'lvis_predictions.json')
    with open(merged_pred_file, 'w') as f:
        json.dump(all_predictions, f)

    # Evaluate
    print(f"\nRunning LVIS evaluation...")
    results = evaluate_with_lvis(
        all_predictions, lvis_data, set(image_ids), categories, args.output_dir
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (OWL-ViT Zero-Shot)")
    print("=" * 70)
    print(f"  Images evaluated: {len(image_ids)}")
    print(f"  Categories: {len(categories)} (r:{results.get('n_rare_cats', 0)}, c:{results.get('n_common_cats', 0)}, f:{results.get('n_freq_cats', 0)} with GT)")
    print("-" * 70)
    print(f"  AP   (IoU=0.50:0.95): {results.get('AP', 0):.1f}%")
    print(f"  AP50 (IoU=0.50):      {results.get('AP50', 0):.1f}%")
    print(f"  AP75 (IoU=0.75):      {results.get('AP75', 0):.1f}%")
    print("-" * 70)
    print(f"  APr  (rare):          {results.get('APr', 0):.1f}%")
    print(f"  APc  (common):        {results.get('APc', 0):.1f}%")
    print(f"  APf  (frequent):      {results.get('APf', 0):.1f}%")
    print("-" * 70)
    print(f"  APs  (small):         {results.get('APs', 0):.1f}%")
    print(f"  APm  (medium):        {results.get('APm', 0):.1f}%")
    print(f"  APl  (large):         {results.get('APl', 0):.1f}%")
    print("=" * 70)

    # Save results
    results['num_images'] = len(image_ids)
    results['inference_time_seconds'] = elapsed
    results['model_name'] = args.model_name
    results['timestamp'] = datetime.now().isoformat()

    results_file = os.path.join(args.output_dir, 'lvis_zeroshot_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {os.path.join(args.output_dir, 'lvis_predictions.json')}")


if __name__ == '__main__':
    main()
