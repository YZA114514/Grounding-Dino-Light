import json
import os
import argparse
import random

def split_val2017(input_file, output_train, num_train=100):
    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Filter for val2017 images only
    val2017_images = []
    for img in images:
        # Check if image belongs to val2017 based on coco_url
        if 'val2017' in img.get('coco_url', ''):
            img_copy = dict(img)
            # Ensure file_name is set correctly
            if 'file_name' not in img_copy or not img_copy['file_name']:
                img_copy['file_name'] = img_copy['coco_url'].split('/')[-1]
            val2017_images.append(img_copy)
            
    print(f"Total images in original file: {len(images)}")
    print(f"Images belonging to val2017: {len(val2017_images)}")
    
    if len(val2017_images) == 0:
        print("Error: No val2017 images found in the input file.")
        return

    # Shuffle and split
    # random.seed(42) # Optional: for reproducibility
    # random.shuffle(val2017_images)
    
    # Take the first N images for training
    train_images = val2017_images[:num_train]
    
    train_img_ids = set(img['id'] for img in train_images)
    
    # Filter annotations for train set
    train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
    
    train_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': categories,
        'images': train_images,
        'annotations': train_anns
    }
    
    print(f"Created training subset with {len(train_images)} images and {len(train_anns)} annotations.")
    print(f"Saving to {output_train}...")
    
    with open(output_train, 'w') as f:
        json.dump(train_data, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Input LVIS json")
    parser.add_argument('--output_train', required=True, help="Output training subset json")
    parser.add_argument('--num_train', type=int, default=100, help="Number of images for training subset")
    
    args = parser.parse_args()
    
    split_val2017(args.input, args.output_train, args.num_train)
