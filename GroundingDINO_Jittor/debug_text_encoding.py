#!/usr/bin/env python3
"""
Debug script to understand text encoding and token-to-category mapping
"""

import os
import sys
import json

# Add the project root to path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

import jittor as jt
from jittor_implementation.models.groundingdino import GroundingDINO
from jittor_implementation.models.backbone.swin_transformer import build_swin_transformer

def debug_text_encoding():
    print("Debugging text encoding and token-to-category mapping...")
    
    # Set device
    jt.flags.use_cuda = 1
    print("Using GPU")
    
    # Create a simple model to get text encoder
    backbone = build_swin_transformer(
        modelname="swin_T_224_1k",
        pretrain_img_size=224,
        out_indices=(1, 2, 3),
        dilation=False,
    )
    
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        max_text_len=256,
        two_stage_type="standard",
        dec_pred_bbox_embed_share=False,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=False,
    )
    
    # Load weights
    weights_path = '/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor/weights/groundingdino_swint_jittor.pkl'
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        import pickle
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        # Simplified weight loading for text encoder only
        bert_weights = {}
        for k, v in weights.items():
            if k.startswith('bert.'):
                bert_weights[k] = v
        
        # Load BERT weights
        if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'bert'):
            bert_state = model.text_encoder.bert.state_dict()
            bert_loaded = 0
            for k, v in bert_weights.items():
                bert_key = k[5:] if k.startswith('bert.') else k
                if bert_key in bert_state:
                    if bert_state[bert_key].shape == tuple(v.shape):
                        bert_state[bert_key] = jt.array(v)
                        bert_loaded += 1
            model.text_encoder.bert.load_state_dict(bert_state)
            print(f"Loaded {bert_loaded} BERT weights")
    
    model.eval()
    
    # Test with simple category names
    test_categories = ["aerosol_can", "air_conditioner", "airplane", "alarm_clock"]
    
    print("\n" + "="*60)
    print("TEXT ENCODING DEBUG")
    print("="*60)
    
    for i, category in enumerate(test_categories):
        print(f"\n--- Category {i}: '{category}' ---")
        
        # Encode text
        with jt.no_grad():
            text_features = model.text_encoder(category)
        
        print(f"Encoded text shape: {text_features['encoded_text'].shape}")
        print(f"Text token mask shape: {text_features['text_token_mask'].shape}")
        
        # Get the actual tokens if available
        if 'text_tokens' in text_features:
            tokens = text_features['text_tokens']
            print(f"Tokens: {tokens}")
            print(f"Token shape: {tokens.shape}")
        
        # Check token mask
        mask = text_features['text_token_mask']
        print(f"Token mask shape: {mask.shape}")
        mask_np = mask.numpy()
        if mask_np.ndim > 1:
            mask_np = mask_np[0]  # Take first element if batched
        print(f"Valid token positions: {[i for i, m in enumerate(mask_np) if m]}")
    
    # Test with the actual prompt used in evaluation
    print(f"\n--- Testing with actual prompt ---")
    prompt = "aerosol_can. air_conditioner. airplane. alarm_clock. alcohol. alligator. almond. ambulance. amplifier. anchor. antenna. ape. apple. apple_ipad."
    print(f"Prompt: {prompt}")
    
    with jt.no_grad():
        text_features = model.text_encoder(prompt)
    
    print(f"Prompt encoded text shape: {text_features['encoded_text'].shape}")
    print(f"Prompt token mask shape: {text_features['text_token_mask'].shape}")
    
    mask = text_features['text_token_mask']
    mask_np = mask.numpy()
    if mask_np.ndim > 1:
        mask_np = mask_np[0]  # Take first element if batched
    valid_positions = [i for i, m in enumerate(mask_np) if m]
    print(f"Valid token positions: {valid_positions}")
    print(f"Number of valid tokens: {len(valid_positions)}")
    
    # Try to map token positions back to category names
    # This is the key issue - we need to understand which token position corresponds to which category
    words_in_prompt = prompt.replace('.', '').split()
    print(f"\nWords in prompt: {words_in_prompt}")
    
    # Check if we can map position to word
    print(f"\nToken position to word mapping attempt:")
    current_pos = 0
    for word in words_in_prompt:
        print(f"  Position {current_pos}: '{word}'")
        # In reality, BERT might use multiple tokens per word, so this is simplified
        current_pos += 1

if __name__ == "__main__":
    debug_text_encoding()
