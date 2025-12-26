import sys
import jittor as jt
import os

# Add path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

# Load weights
weights_path = 'weights/groundingdino_swint_jittor.pkl'
print(f"Loading weights from {weights_path}")
state_dict = jt.load(weights_path)

print("\n=== class_embed in state_dict? ===")
found_class = False
for k in state_dict.keys():
    if 'class_embed' in k:
        print(f"{k}: {state_dict[k].shape}")
        found_class = True
if not found_class:
    print("No class_embed keys found in state_dict")

print("\n=== bbox_embed in state_dict? ===")
found_bbox = False
for k in state_dict.keys():
    if 'bbox_embed' in k:
        print(f"{k}: {state_dict[k].shape}")
        found_bbox = True
if not found_bbox:
    print("No bbox_embed keys found in state_dict")
