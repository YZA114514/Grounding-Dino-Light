import json
import os
import argparse

def create_subset(input_file, output_file, fraction=50):
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Sample 1 in N images
    subset_images = []
    for img in images[::fraction]:
        img_copy = dict(img)
        if 'file_name' not in img_copy or not img_copy['file_name']:
            if 'coco_url' in img_copy and img_copy['coco_url']:
                img_copy['file_name'] = img_copy['coco_url'].split('/')[-1]
            else:
                img_copy['file_name'] = f"{img_copy['id']:012d}.jpg"
        subset_images.append(img_copy)

    subset_img_ids = set(img['id'] for img in subset_images)
    
    # Filter annotations
    subset_anns = [ann for ann in annotations if ann['image_id'] in subset_img_ids]
    
    subset_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': categories,
        'images': subset_images,
        'annotations': subset_anns
    }
    
    print(f"Original images: {len(images)}")
    print(f"Subset images: {len(subset_images)}")
    print(f"Subset annotations: {len(subset_anns)}")
    print(f"Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(subset_data, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input LVIS json")
    parser.add_argument('--output', required=True, help="Output subset json")
    parser.add_argument('--fraction', type=int, default=50, help="1/N fraction to keep")
    args = parser.parse_args()
    
    create_subset(args.input, args.output, args.fraction)
