#!/usr/bin/env python3
"""
Build LVIS Training Dataset Cache

Preprocess LVIS training data for efficient loading during training.
Creates compressed .npz files with normalized images and annotations.

Usage:
    python scripts/build_lvis_train_dataset.py \
        --lvis_ann lvis/lvis_v1_train.json \
        --train_dir ./train2017 \
        --val_dir ./val2017 \
        --output_dir data/lvis_finetune_preload_cache \
        --image_size 640

Output:
    data/lvis_finetune_preload_cache/
    ├── index.pkl          # Sample metadata list
    ├── categories.pkl     # Category ID → name mapping
    └── {image_id}.npz     # ~100k preprocessed samples
"""

import os
import json
import argparse
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def find_image_file(img_info, train_dir, val_dir):
    """
    Find image file in train2017 or val2017 directories.

    Args:
        img_info (dict): Image info from LVIS annotation
        train_dir (str): Path to train2017 directory
        val_dir (str): Path to val2017 directory

    Returns:
        str: Full path to image file, or None if not found
    """
    # Extract filename from coco_url
    if 'coco_url' in img_info:
        filename = img_info['coco_url'].split('/')[-1]
    elif 'file_name' in img_info:
        filename = img_info['file_name']
    else:
        # Fallback: create filename from ID
        filename = f"{img_info['id']:012d}.jpg"

    # Try train2017 first, then val2017
    train_path = os.path.join(train_dir, filename)
    if os.path.exists(train_path):
        return train_path

    val_path = os.path.join(val_dir, filename)
    if os.path.exists(val_path):
        return val_path

    return None


def resize_and_normalize(image, target_size):
    """
    Resize image to square target_size and apply ImageNet normalization.

    Args:
        image (PIL.Image): Input image
        target_size (int): Target size for both dimensions (square resize)

    Returns:
        tuple: (normalized_image_array, new_size)
               normalized_image: (H, W, 3) float32 array, values in [0,1]
               new_size: (new_h, new_w) tuple
    """
    orig_w, orig_h = image.size

    # Force square resize - this may distort aspect ratio slightly
    new_w = target_size
    new_h = target_size

    # Resize image to square
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    # Convert to numpy array and normalize to [0,1]
    image_array = np.array(resized_image, dtype=np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalize each channel
    image_array = (image_array - mean) / std

    # Convert to (H, W, C) format as expected by the spec
    # Note: transforms.py uses (C, H, W) but spec says (3, H, W), so we'll use (3, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))  # (C, H, W)

    return image_array.astype(np.float16), (new_h, new_w)


def process_annotations(anns, categories, orig_w, orig_h):
    """
    Process LVIS annotations into normalized cxcywh format.

    Args:
        anns (list): List of annotations for this image
        categories (dict): Category ID to info mapping
        orig_w, orig_h: Original image dimensions

    Returns:
        tuple: (boxes, labels, cat_names)
               boxes: (N, 4) normalized cxcywh coordinates
               labels: (N,) category IDs
               cat_names: set of category names in this image
    """
    boxes = []
    labels = []
    cat_names = set()

    for ann in anns:
        x, y, w, h = ann['bbox']

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # Convert xywh to normalized cxcywh
        cx = (x + w / 2) / orig_w
        cy = (y + h / 2) / orig_h
        nw = w / orig_w
        nh = h / orig_h

        # Ensure coordinates are in [0, 1]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
            continue

        boxes.append([cx, cy, nw, nh])
        labels.append(ann['category_id'])
        cat_names.add(categories[ann['category_id']]['name'])

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), set()

    return (np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            cat_names)


def save_sample(output_path, image, boxes, labels, orig_size, new_size):
    """
    Save preprocessed sample to compressed .npz file.

    Args:
        output_path (str): Path to save .npz file
        image: (3, H, W) float16 normalized image
        boxes: (N, 4) float32 normalized cxcywh boxes
        labels: (N,) int32 category IDs
        orig_size: (orig_h, orig_w) tuple
        new_size: (new_h, new_w) tuple
    """
    np.savez_compressed(output_path,
                       image=image,
                       boxes=boxes,
                       labels=labels,
                       orig_size=np.array(orig_size, dtype=np.int32),
                       new_size=np.array(new_size, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description="Build LVIS training dataset cache")
    parser.add_argument('--lvis_ann', default = 'LVIS/lvis_v1_train.json', help="Path to LVIS annotation JSON")
    parser.add_argument('--train_dir', default='./train2017', help="Path to COCO train2017 images")
    parser.add_argument('--val_dir', default='./val2017', help="Path to COCO val2017 images")
    parser.add_argument('--output_dir', default='data/lvis_finetune_preload_cache',
                       help="Output directory for cache files")
    parser.add_argument('--image_size', type=int, default=640,
                       help="Target size for longest image side")
    parser.add_argument('--max_samples', type=int, default=None,
                       help="Limit number of samples for testing (optional)")

    args = parser.parse_args()

    print("Loading LVIS annotations...")
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)

    # Build data structures
    images = {img['id']: img for img in lvis_data['images']}
    categories = {cat['id']: cat for cat in lvis_data['categories']}
    img_to_anns = defaultdict(list)

    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build index and process images
    index = []
    missing_images = 0
    empty_images = 0

    print(f"Processing {len(images)} images...")

    # Limit samples for testing if specified
    image_ids = list(images.keys())
    if args.max_samples:
        image_ids = image_ids[:args.max_samples]

    for img_id in tqdm(image_ids):
        img_info = images[img_id]
        anns = img_to_anns[img_id]

        # Find image file
        image_path = find_image_file(img_info, args.train_dir, args.val_dir)
        if image_path is None:
            missing_images += 1
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size

            # Resize and normalize
            processed_image, new_size = resize_and_normalize(image, args.image_size)

            # Process annotations
            boxes, labels, cat_names = process_annotations(anns, categories, orig_w, orig_h)

            # Skip images with no valid annotations
            if len(boxes) == 0:
                empty_images += 1
                continue

            # Save sample
            npz_path = os.path.join(args.output_dir, f"{img_id}.npz")
            save_sample(npz_path, processed_image, boxes, labels,
                       (orig_h, orig_w), new_size)

            # Add to index
            index.append({
                'id': img_id,
                'path': f"{args.output_dir}/{img_id}.npz",
                'cat_names': list(cat_names),
                'num_boxes': len(boxes)
            })

        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            continue

    # Save index and categories
    print(f"Saving index.pkl ({len(index)} samples)...")
    with open(os.path.join(args.output_dir, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)

    print(f"Saving categories.pkl ({len(categories)} categories)...")
    cat_mapping = {cat_id: cat_info['name'] for cat_id, cat_info in categories.items()}
    with open(os.path.join(args.output_dir, 'categories.pkl'), 'wb') as f:
        pickle.dump(cat_mapping, f)

    # Summary
    print("\nProcessing complete!")
    print(f"Total images processed: {len(index)}")
    print(f"Missing images: {missing_images}")
    print(f"Empty images (no valid annotations): {empty_images}")
    print(f"Categories: {len(categories)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
