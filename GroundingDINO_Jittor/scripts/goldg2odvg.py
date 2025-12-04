# GoldG to ODVG Script (Member B)
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse


def goldg_to_odvg(goldg_anno_path: str, output_path: str, image_dir: str = None) -> None:
    """
    Convert GoldG (Grounded Image Captioning) format to ODVG (Object Detection + Visual Grounding) format
    
    Args:
        goldg_anno_path: Path to GoldG annotation file
        output_path: Path to save the converted ODVG format
        image_dir: Directory containing images (optional)
    """
    # Load GoldG annotation file
    with open(goldg_anno_path, 'r') as f:
        goldg_data = json.load(f)
    
    # Convert to ODVG format
    odvg_data = []
    
    for item in goldg_data:
        # Extract image information
        img_info = item.get('image', {})
        img_id = img_info.get('id', 0)
        file_name = img_info.get('file_name', '')
        height = img_info.get('height', 0)
        width = img_info.get('width', 0)
        
        # Extract caption and entities
        caption = item.get('caption', '')
        entities = item.get('entities', [])
        
        # Extract bounding boxes and labels from entities
        bboxes = []
        labels = []
        categories = []
        
        for entity in entities:
            # GoldG entity bbox format: [x, y, width, height]
            bbox = entity.get('bbox', [])
            label = entity.get('name', '')
            
            if bbox and label:
                bboxes.append(bbox)
                labels.append(label)
                categories.append(label)
        
        # Create ODVG entry
        odvg_entry = {
            'image_id': img_id,
            'file_name': file_name,
            'height': height,
            'width': width,
            'bboxes': bboxes,
            'labels': labels,
            'categories': list(set(categories)),  # Unique categories
            'text': caption if caption else ' . '.join(list(set(categories)))  # Use caption or create from categories
        }
        
        # Add image path if image_dir is provided
        if image_dir:
            odvg_entry['image_path'] = os.path.join(image_dir, file_name)
        
        odvg_data.append(odvg_entry)
    
    # Save ODVG format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(odvg_data, f, indent=2)
    
    print(f"Converted {len(odvg_data)} images from GoldG to ODVG format")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert GoldG format to ODVG format')
    parser.add_argument('--goldg_path', type=str, required=True, help='Path to GoldG annotation file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save ODVG format')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing images')
    
    args = parser.parse_args()
    
    goldg_to_odvg(args.goldg_path, args.output_path, args.image_dir)


if __name__ == '__main__':
    main()

