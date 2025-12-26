#!/usr/bin/env python3
"""
Test Data Loading for LVIS/val
"""

import os
import sys
import json

# Add the project root to path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

from jittor_implementation.data.dataset import LVISDataset

def main():
    # Path to LVIS val annotations
    ann_file = '/root/shared-nvme/GroundingDINO-Light/LVIS/lvis_v1_val.json'
    image_dir = '/root/shared-nvme/GroundingDINO-Light/LVIS/val'
    
    # Create dataset
    dataset = LVISDataset(ann_file, image_dir)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Annotations: {len(sample['annotations'])}")
    
    # Print some annotation info
    for ann in sample['annotations'][:5]:
        print(f"Category: {ann['category_id']}, Bbox: {ann['bbox']}")

if __name__ == "__main__":
    main()