# COCO to ODVG Script (Member B)
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse


def coco_to_odvg(coco_anno_path: str, output_path: str, image_dir: str = None) -> None:
    """
    Convert COCO format to ODVG (Object Detection + Visual Grounding) format
    
    Args:
        coco_anno_path: Path to COCO annotation file
        output_path: Path to save the converted ODVG format
        image_dir: Directory containing images (optional)
    """
    # Load COCO annotation file
    with open(coco_anno_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Create category_id to category name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image_id
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)
    
    # Convert to ODVG format
    odvg_data = []
    
    for img_id, img_info in image_id_to_info.items():
        # Get annotations for this image
        annotations = image_id_to_annotations.get(img_id, [])
        
        # Extract bounding boxes and labels
        bboxes = []
        labels = []
        categories = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_id_to_name[category_id]
            
            bboxes.append(bbox)
            labels.append(category_name)
            categories.append(category_name)
        
        # Create ODVG entry
        odvg_entry = {
            'image_id': img_id,
            'file_name': img_info['file_name'],
            'height': img_info['height'],
            'width': img_info['width'],
            'bboxes': bboxes,
            'labels': labels,
            'categories': list(set(categories)),  # Unique categories
            'text': ' . '.join(list(set(categories)))  # Text prompt for grounding
        }
        
        # Add image path if image_dir is provided
        if image_dir:
            odvg_entry['image_path'] = os.path.join(image_dir, img_info['file_name'])
        
        odvg_data.append(odvg_entry)
    
    # Save ODVG format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(odvg_data, f, indent=2)
    
    print(f"Converted {len(odvg_data)} images from COCO to ODVG format")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to ODVG format')
    parser.add_argument('--coco_path', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save ODVG format')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing images')
    
    args = parser.parse_args()
    
    coco_to_odvg(args.coco_path, args.output_path, args.image_dir)


if __name__ == '__main__':
    main()

