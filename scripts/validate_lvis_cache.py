#!/usr/bin/env python3
"""
Validate LVIS Training Dataset Cache

Verify the integrity and structure of the preprocessed LVIS dataset cache.

Usage:
    python scripts/validate_lvis_cache.py --cache_dir data/lvis_finetune_preload_cache
"""

import os
import pickle
import numpy as np
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def validate_cache_structure(cache_dir):
    """Validate the cache directory structure and file counts."""
    print(f"Validating cache in: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("ERROR: Cache directory does not exist!")
        return False

    # Check required files
    index_path = os.path.join(cache_dir, 'index.pkl')
    categories_path = os.path.join(cache_dir, 'categories.pkl')

    if not os.path.exists(index_path):
        print("ERROR: index.pkl not found!")
        return False

    if not os.path.exists(categories_path):
        print("ERROR: categories.pkl not found!")
        return False

    # Load index
    with open(index_path, 'rb') as f:
        index = pickle.load(f)

    # Load categories
    with open(categories_path, 'rb') as f:
        categories = pickle.load(f)

    print(f"✓ Found index.pkl with {len(index)} samples")
    print(f"✓ Found categories.pkl with {len(categories)} categories")

    # Check npz files exist
    missing_npz = 0
    for entry in index:
        npz_path = os.path.join(cache_dir, f"{entry['id']}.npz")
        if not os.path.exists(npz_path):
            missing_npz += 1

    if missing_npz > 0:
        print(f"ERROR: {missing_npz} npz files missing!")
        return False

    print(f"✓ All {len(index)} npz files present")

    return True, index, categories


def validate_sample_data(index, categories, cache_dir, sample_idx=0):
    """Validate the structure and content of a sample."""
    if sample_idx >= len(index):
        print(f"ERROR: Sample index {sample_idx} out of range!")
        return False

    entry = index[sample_idx]
    npz_path = os.path.join(cache_dir, f"{entry['id']}.npz")

    print(f"\nValidating sample {sample_idx} (ID: {entry['id']}):")

    # Load npz data
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"ERROR loading npz: {e}")
        return False

    # Check required keys
    required_keys = ['image', 'boxes', 'labels', 'orig_size', 'new_size']
    for key in required_keys:
        if key not in data:
            print(f"ERROR: Missing key '{key}' in npz file")
            return False

    print("✓ All required keys present")

    # Validate data types and shapes
    image = data['image']
    boxes = data['boxes']
    labels = data['labels']
    orig_size = data['orig_size']
    new_size = data['new_size']

    # Check image
    if image.dtype != np.float16:
        print(f"ERROR: Image dtype {image.dtype}, expected float16")
        return False
    if image.shape[0] != 3:
        print(f"ERROR: Image shape {image.shape}, expected (3, H, W)")
        return False
    print(f"✓ Image shape: {image.shape}, dtype: {image.dtype}")

    # Check boxes
    if boxes.dtype != np.float32:
        print(f"ERROR: Boxes dtype {boxes.dtype}, expected float32")
        return False
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        print(f"ERROR: Boxes shape {boxes.shape}, expected (N, 4)")
        return False
    if len(boxes) != entry['num_boxes']:
        print(f"ERROR: Box count mismatch: {len(boxes)} vs {entry['num_boxes']}")
        return False
    print(f"✓ Boxes shape: {boxes.shape}, dtype: {boxes.dtype}")

    # Check labels
    if labels.dtype != np.int32:
        print(f"ERROR: Labels dtype {labels.dtype}, expected int32")
        return False
    if len(labels) != len(boxes):
        print(f"ERROR: Labels length {len(labels)} != boxes length {len(boxes)}")
        return False
    print(f"✓ Labels shape: {labels.shape}, dtype: {labels.dtype}")

    # Check sizes
    if orig_size.dtype != np.int32 or new_size.dtype != np.int32:
        print("ERROR: Size arrays should be int32")
        return False
    print(f"✓ Original size: {orig_size}, New size: {new_size}")

    # Validate box coordinates are in [0, 1]
    if len(boxes) > 0:
        min_vals = np.min(boxes, axis=0)
        max_vals = np.max(boxes, axis=0)
        if np.any(min_vals < 0) or np.any(max_vals > 1):
            print(f"ERROR: Box coordinates out of [0,1] range: min={min_vals}, max={max_vals}")
            return False
        print("✓ Box coordinates are normalized [0,1]")

    # Validate category names match
    unique_labels = set(labels)
    expected_names = set(entry['cat_names'])
    actual_names = set(categories[label] for label in unique_labels)

    if expected_names != actual_names:
        print(f"ERROR: Category names mismatch: expected {expected_names}, got {actual_names}")
        return False
    print("✓ Category names match")

    return True


def visualize_sample(index, categories, cache_dir, sample_idx=0, save_path=None):
    """Create a visualization of a sample with bounding boxes."""
    entry = index[sample_idx]
    npz_path = os.path.join(cache_dir, f"{entry['id']}.npz")

    # Load data
    data = np.load(npz_path)
    image = data['image']  # (3, H, W) float16
    boxes = data['boxes']   # (N, 4) cxcywh normalized
    labels = data['labels'] # (N,) int32

    # Convert back to HWC for visualization
    image_vis = np.transpose(image, (1, 2, 0))  # (H, W, 3)

    # Denormalize image for visualization (approximate)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_vis = image_vis * std + mean
    image_vis = np.clip(image_vis, 0, 1)

    # Convert to PIL Image
    pil_image = Image.fromarray((image_vis * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)

    # Draw boxes
    h, w = pil_image.size[::-1]  # PIL size is (W, H)
    for box, label in zip(boxes, labels):
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h

        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        cat_name = categories[label]
        draw.text((x1, y1-10), cat_name, fill='red')

    if save_path:
        pil_image.save(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.imshow(pil_image)
        plt.title(f"Sample {sample_idx} (ID: {entry['id']})")
        plt.axis('off')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Validate LVIS dataset cache")
    parser.add_argument('--cache_dir', default='data/lvis_finetune_preload_cache',
                       help="Path to cache directory")
    parser.add_argument('--visualize', type=int, nargs='?', const=0,
                       help="Visualize sample (optional: specify sample index)")
    parser.add_argument('--save_vis', help="Save visualization to file")

    args = parser.parse_args()

    # Validate structure
    success, index, categories = validate_cache_structure(args.cache_dir)
    if not success:
        return

    # Validate sample data
    if not validate_sample_data(index, categories, args.cache_dir, sample_idx=0):
        return

    print("\n✓ Cache validation passed!")

    # Optional visualization
    if args.visualize is not None:
        sample_idx = args.visualize
        if sample_idx >= len(index):
            print(f"ERROR: Sample index {sample_idx} out of range (max: {len(index)-1})")
            return

        print(f"\nGenerating visualization for sample {sample_idx}...")
        visualize_sample(index, categories, args.cache_dir,
                        sample_idx=sample_idx, save_path=args.save_vis)


if __name__ == "__main__":
    main()
