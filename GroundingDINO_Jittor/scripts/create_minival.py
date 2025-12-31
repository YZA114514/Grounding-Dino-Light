import json
import os
from pathlib import Path

def create_minival(lvis_val_path, coco_train_path, output_path):
    """
    Create LVIS minival set by removing images that appear in COCO training set.

    Minival definition: LVIS val images - COCO train images
    This ensures fair evaluation for models pre-trained on COCO.
    """
    print("Loading LVIS val annotations...")
    with open(lvis_val_path, 'r') as f:
        lvis_val = json.load(f)

    print("Loading COCO train annotations...")
    with open(coco_train_path, 'r') as f:
        coco_train = json.load(f)

    # Extract image IDs from both datasets
    lvis_image_ids = {img['id'] for img in lvis_val['images']}
    coco_image_ids = {img['id'] for img in coco_train['images']}

    # Compute minival IDs: LVIS val images that are NOT in COCO train
    minival_image_ids = lvis_image_ids - coco_image_ids

    print(f"LVIS val images: {len(lvis_image_ids)}")
    print(f"COCO train images: {len(coco_image_ids)}")
    print(f"Minival images: {len(minival_image_ids)}")
    print(f"Removed images: {len(lvis_image_ids) - len(minival_image_ids)}")

    # Filter images for minival
    minival_images = [img for img in lvis_val['images'] if img['id'] in minival_image_ids]

    # Filter annotations for minival images
    minival_annotations = [ann for ann in lvis_val['annotations'] if ann['image_id'] in minival_image_ids]

    # Create minival dataset
    minival_data = {
        'info': lvis_val.get('info', {}),
        'licenses': lvis_val.get('licenses', []),
        'categories': lvis_val['categories'],  # Keep all categories
        'images': minival_images,
        'annotations': minival_annotations
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save minival dataset
    print(f"Saving minival to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(minival_data, f)

    print("Minival creation complete!")
    print(f"Minival contains {len(minival_images)} images and {len(minival_annotations)} annotations")

    return minival_data

def verify_minival(minival_data, coco_train_path):
    """Verify minival has no overlap with COCO training set."""
    print("\nVerifying minival...")

    # Load COCO train again for verification
    with open(coco_train_path, 'r') as f:
        coco_train = json.load(f)

    coco_image_ids = {img['id'] for img in coco_train['images']}
    minival_image_ids = {img['id'] for img in minival_data['images']}

    overlap = minival_image_ids & coco_image_ids
    if len(overlap) == 0:
        print("✅ SUCCESS: No overlap with COCO training set!")
    else:
        print(f"❌ ERROR: Found {len(overlap)} overlapping images!")
        print(f"Overlapping IDs: {list(overlap)[:10]}...")

    print(f"Minival size: {len(minival_image_ids)} images")

if __name__ == "__main__":
    # File paths
    lvis_val_path = Path("../../LVIS/lvis_v1_val.json")
    coco_train_path = Path("../../annotations/instances_train2017.json")
    output_path = Path("../../LVIS/minival/lvis_v1_minival.json")

    # Create minival
    minival_data = create_minival(lvis_val_path, coco_train_path, output_path)

    # Verify
    verify_minival(minival_data, coco_train_path)
