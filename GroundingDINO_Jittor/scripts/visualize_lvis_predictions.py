#!/usr/bin/env python
"""
Visualize LVIS Zero-Shot Detection Results

This script visualizes zero-shot detection predictions on images by drawing bounding boxes
with category names and confidence scores.

Usage:
    python scripts/visualize_lvis_predictions.py \
        --predictions outputs/test_10_images/lvis_predictions.json \
        --lvis_ann ../data/lvis_v1_val.json \
        --image_dir ../data/val2017 \
        --output_dir outputs/test_10_images/visualized \
        --score_threshold 0.3 \
        --max_boxes 50
"""

import os
import json
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize LVIS Detection Results')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--lvis_ann', type=str, required=True,
                        help='Path to LVIS annotation file (for category names)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualized images')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Minimum confidence score to display')
    parser.add_argument('--max_boxes', type=int, default=50,
                        help='Maximum number of boxes to draw per image')
    parser.add_argument('--font_size', type=int, default=12,
                        help='Font size for labels')
    return parser.parse_args()


def load_predictions(predictions_file):
    """Load predictions from JSON file."""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    return predictions


def load_lvis_categories(lvis_ann_file):
    """Load category names from LVIS annotations."""
    with open(lvis_ann_file, 'r') as f:
        lvis_data = json.load(f)

    categories = lvis_data['categories']
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    return cat_id_to_name


def group_predictions_by_image(predictions):
    """Group predictions by image_id."""
    img_predictions = defaultdict(list)
    for pred in predictions:
        img_predictions[pred['image_id']].append(pred)
    return img_predictions


def get_box_color(score):
    """Get color based on confidence score."""
    if score >= 0.7:
        return (255, 0, 0)  # Red for high confidence
    elif score >= 0.4:
        return (255, 255, 0)  # Yellow for medium confidence
    else:
        return (0, 255, 0)  # Green for lower confidence


def visualize_image(img_path, predictions, cat_id_to_name, score_threshold, max_boxes, font_size, output_path):
    """Visualize predictions on a single image."""
    # Load image
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return False

    # Create drawing context
    draw = ImageDraw.Draw(image)

    # Try to load font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Filter and sort predictions
    filtered_preds = [p for p in predictions if p['score'] >= score_threshold]
    filtered_preds.sort(key=lambda x: x['score'], reverse=True)
    filtered_preds = filtered_preds[:max_boxes]

    # Draw each prediction
    for pred in filtered_preds:
        bbox = pred['bbox']
        score = pred['score']
        cat_id = pred['category_id']
        cat_name = cat_id_to_name.get(cat_id, f'cat_{cat_id}')

        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        # Get color based on score
        color = get_box_color(score)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        label = f"{cat_name}: {score:.2f}"
        text_bbox = draw.textbbox((x1, y1 - font_size - 2), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle for text
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width, y1], fill=color)

        # Draw text (use black for visibility)
        draw.text((x1, y1 - text_height - 2), label, fill=(0, 0, 0), font=font)

    # Save image
    image.save(output_path)
    return True


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions and annotations...")

    # Load data
    predictions = load_predictions(args.predictions)
    cat_id_to_name = load_lvis_categories(args.lvis_ann)
    img_predictions = group_predictions_by_image(predictions)

    print(f"Loaded {len(predictions)} predictions for {len(img_predictions)} images")
    print(f"Loaded {len(cat_id_to_name)} categories")

    # Process each image
    success_count = 0
    total_count = 0

    for img_id, preds in img_predictions.items():
        # Get image filename
        img_filename = f"{img_id:012d}.jpg"
        img_path = os.path.join(args.image_dir, img_filename)

        # Output path
        output_filename = f"{img_id:012d}_annotated.jpg"
        output_path = os.path.join(args.output_dir, output_filename)

        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Visualize
        success = visualize_image(
            img_path, preds, cat_id_to_name,
            args.score_threshold, args.max_boxes, args.font_size, output_path
        )

        if success:
            success_count += 1
            print(f"Visualized {img_filename} → {output_filename} ({len(preds)} predictions)")
        else:
            print(f"Failed to visualize {img_filename}")

        total_count += 1

    print("\nVisualization complete!")
    print(f"Processed: {success_count}/{total_count} images")
    print(f"Output directory: {args.output_dir}")

    # Show some statistics
    all_scores = [p['score'] for p in predictions]
    if all_scores:
        print("\nPrediction statistics:")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Average score: {np.mean(all_scores):.3f}")
        print(f"  Max score: {np.max(all_scores):.3f}")
        print(f"  Min score: {np.min(all_scores):.3f}")
        print(f"  Score threshold used: {args.score_threshold}")

        # Count predictions above threshold
        above_threshold = sum(1 for s in all_scores if s >= args.score_threshold)
        print(f"  Predictions shown (>= {args.score_threshold}): {above_threshold}")


if __name__ == '__main__':
    main()
