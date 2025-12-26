import sys
import os
import pickle
import jittor as jt

# Add path
sys.path.append(os.getcwd())

from jittor_implementation.models.groundingdino import GroundingDINO

def check_backbone():
    # Load weights
    weights_path = 'weights/groundingdino_swint_jittor.pkl'
    print(f"Loading weights from {weights_path}")
    with open(weights_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Clean keys
    cleaned_weights = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        cleaned_weights[k] = v
        
    from jittor_implementation.models.backbone.swin_transformer import build_swin_transformer
    
    # Build backbone
    backbone = build_swin_transformer("swin_T_224_1k", 224)
    
    # Create model
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
    )
    
    model_keys = set(model.state_dict().keys())
    weight_keys = set(cleaned_weights.keys())
    
    print(f"Model keys: {len(model_keys)}")
    print("First 20 model keys:")
    for k in sorted(list(model_keys))[:20]:
        print(k)
    
    print(f"Weight keys: {len(weight_keys)}")
    
    # Check backbone
    backbone_keys = [k for k in model_keys if 'backbone' in k]
    print(f"Backbone keys in model: {len(backbone_keys)}")
    
    missing_backbone = [k for k in backbone_keys if k not in weight_keys]
    print(f"Missing backbone keys: {len(missing_backbone)}")
    
    if missing_backbone:
        print("First 10 missing backbone keys:")
        for k in missing_backbone[:10]:
            print(k)
            
    # Check intersection
    common = model_keys.intersection(weight_keys)
    print(f"Common keys: {len(common)}")
    
    # Check what IS in weights that looks like backbone
    weight_backbone = [k for k in weight_keys if 'backbone' in k]
    print(f"Backbone keys in weights: {len(weight_backbone)}")
    
    if weight_backbone and missing_backbone:
        print("First 10 backbone keys in weights:")
        for k in weight_backbone[:10]:
            print(k)

if __name__ == "__main__":
    check_backbone()
