#!/usr/bin/env python
"""
Quick test script for OWL-ViT LVIS evaluation

This script runs a quick test of the OWL-ViT evaluation on a few images
to verify the setup works correctly.

Usage:
    python scripts/test_owlvit_quick.py
"""
import os
import sys

# Add the project root to the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

def main():
    print("Testing OWL-ViT LVIS Evaluation Setup")
    print("=" * 50)

    # Test imports
    print("1. Testing imports...")
    try:
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        import torch
        print("   ✓ Required packages imported successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        print("   Please install: pip install transformers torch torchvision")
        return

    # Test model loading
    print("\n2. Testing model loading...")
    try:
        model_name = 'google/owlvit-base-patch32'
        processor = OwlViTProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(model_name)
        print("   ✓ OWL-ViT model loaded successfully")
    except Exception as e:
        print(f"   ✗ Model loading error: {e}")
        print("   Make sure you have internet connection for first download")
        return

    # Test LVIS data access
    print("\n3. Testing LVIS data access...")
    lvis_ann_path = '../LVIS/minival/lvis_v1_minival.json'
    if os.path.exists(lvis_ann_path):
        print("   ✓ LVIS minival annotations found")
    else:
        print(f"   ✗ LVIS annotations not found at {lvis_ann_path}")
        print("   Make sure LVIS dataset is downloaded and paths are correct")
        return

    # Test image access
    print("\n4. Testing image access...")
    image_dir = '../LVIS/minival'
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if images:
            print(f"   ✓ Found {len(images)} images in {image_dir}")
        else:
            print(f"   ✗ No .jpg images found in {image_dir}")
            return
    else:
        print(f"   ✗ Image directory not found at {image_dir}")
        return

    print("\n5. Setup verification complete!")
    print("\nTo run OWL-ViT evaluation:")
    print("  Quick test (100 images):")
    print("    python scripts/eval_owlvit_lvis.py --num_images 100 --batch_size 25")
    print("\n  Full evaluation:")
    print("    python scripts/eval_owlvit_lvis.py --full --batch_size 25")
    print("\n  Custom output directory:")
    print("    python scripts/eval_owlvit_lvis.py --num_images 500 --output_dir outputs/owlvit_test")

if __name__ == '__main__':
    main()
