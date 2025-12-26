
import pickle
import sys
import os

# Add path for model import
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

import jittor as jt
jt.flags.use_cuda = 0

from jittor_implementation.models import GroundingDINO
from jittor_implementation.train.config import TrainingConfig

def check_label_enc():
    print("Checking label_enc.weight...")
    
    # 1. Check pickle file
    try:
        with open('weights/groundingdino_swint_jittor.pkl', 'rb') as f:
            data = pickle.load(f)
            if 'label_enc.weight' in data:
                print(f"Pickle file 'label_enc.weight' shape: {data['label_enc.weight'].shape}")
            else:
                print("Pickle file does NOT contain 'label_enc.weight'")
    except Exception as e:
        print(f"Error reading pickle: {e}")

    # 2. Check Model expectation
    try:
        # Need to mock backbone since we don't want to load it fully if not needed, 
        # but GroundingDINO init might need it. 
        # Let's try to init with minimal config.
        
        # We need to match the config used in test_inference.py
        # model = GroundingDINO(
        #     backbone=backbone,
        #     num_queries=900,
        #     hidden_dim=256,
        #     num_feature_levels=4,
        #     nheads=8,
        #     two_stage_type="standard", # Match config
        # )
        
        # We can pass None for backbone if we just want to check parameters created in __init__
        # But GroundingDINO init checks backbone.num_channels if provided.
        
        class MockBackbone:
            num_channels = [192, 384, 768]
            
        model = GroundingDINO(
            backbone=MockBackbone(),
            num_queries=900,
            hidden_dim=256,
            num_feature_levels=4,
            nheads=8,
            two_stage_type="standard",
        )
        
        found = False
        for name, param in model.named_parameters():
            if 'label_enc' in name:
                print(f"Model parameter '{name}' shape: {param.shape}")
                found = True
        
        if not found:
            print("Model does not have any parameter named 'label_enc'")
            
    except Exception as e:
        print(f"Error initializing model: {e}")

if __name__ == "__main__":
    check_label_enc()
