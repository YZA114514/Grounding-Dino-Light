import json
import os
import sys
import subprocess
import time

if len(sys.argv) != 2:
    print("Usage: python LVIS_downloader_new.py <split>")
    print("Splits: train, val, test_challenge, test_dev")
    sys.exit(1)

split = sys.argv[1]

if split in ['train', 'val']:
    json_file = f'LVIS/lvis_v1_{split}.json'
elif split == 'test_challenge':
    json_file = 'LVIS/lvis_v1_image_info_test_challenge.json'
elif split == 'test_dev':
    json_file = 'LVIS/lvis_v1_image_info_test_dev.json'
else:
    print("Invalid split. Use: train, val, test_challenge, test_dev")
    sys.exit(1)

with open(json_file, 'r') as f:
    data = json.load(f)

if 'images' in data:
    urls = [img['coco_url'] for img in data['images'] if 'coco_url' in img]
else:
    urls = []

with open(f'LVIS/{split}_urls.txt', 'w') as f:
    f.write('\n'.join(urls))

# Create the directory if it doesn't exist
os.makedirs(f'LVIS/{split}/', exist_ok=True)

# Download each image using wget
total = len(urls)
if total > 0:
    start_time = time.time()
    for i, url in enumerate(urls, 1):
        filename = os.path.basename(url)
        filepath = os.path.join(f'LVIS/{split}/', filename)
        if os.path.exists(filepath):
            print(f"Skipping {i}/{total}: {filename} (already exists)")
            continue
        download_start = time.time()
        print(f"Downloading {i}/{total}: {filename}")
        result = subprocess.run([
            'wget', url, '-O', filepath, '-nc', '--quiet'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            download_time = time.time() - download_start
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total - i) * avg_time
            eta_hours = int(remaining // 3600)
            eta_minutes = int((remaining % 3600) // 60)
            eta_seconds = int(remaining % 60)
            print(f"  Time for this image: {download_time:.2f}s, ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}")
        else:
            print(f"  Failed to download {filename}: {result.stderr}")
    total_time = time.time() - start_time
    print(f"All downloads completed in {total_time:.2f}s.")
else:
    print(f"No URLs found for split '{split}'.")

# aria2c -x 16 -s 16 --no-proxy "http://images.cocodataset.org/zips/val2017.zip"

# aria2c -x 16 -s 16 --no-proxy "http://images.cocodataset.org/zips/train2017.zip"

# cd /root/shared-nvme/GroundingDINO-Light && aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2017.zip