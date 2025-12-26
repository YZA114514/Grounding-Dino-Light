#!/usr/bin/env python3
"""
Test Inference Script for Jittor GroundingDINO on LVIS/val
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the project root to path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

import jittor as jt
from jittor_implementation.train.config import TrainingConfig
from jittor_implementation.models import GroundingDINO, BERTWrapper
from jittor_implementation.data.transforms import build_transforms

def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    transform = build_transforms(is_train=False)
    image_tensor, _ = transform(image, {})
    image_tensor = jt.Var(image_tensor).float32()
    return image_tensor.unsqueeze(0), image

def main():
    # Set device
    import jittor as jt
    jt.flags.use_cuda = 1  # Use GPU if available
    
    # Create config
    config = TrainingConfig()
    
    # Build Backbone (Swin-T)
    from jittor_implementation.models.backbone.swin_transformer import SwinTransformer
    backbone = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(1, 2, 3), # Output from stage 2, 3, 4 (indices 1, 2, 3) -> [192, 384, 768]
    )
    # Manually set num_channels for GroundingDINO init
    backbone.num_channels = [192, 384, 768]
    
    # Load model
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        two_stage_type="standard", # Match config
        two_stage_bbox_embed_share=False,  # Don't share encoder and decoder bbox_embed
        dec_pred_bbox_embed_share=False,   # Don't share bbox_embed across decoder layers
    )
    text_encoder = model.text_encoder  # Use the model's text encoder
    
    # Print model structure to debug
    print("Model structure:")
    # print(model)
    # Check specific path
    try:
        print(f"Check layer 0 attn: {model.transformer.encoder.layers[0].self_attn}")
        print(f"Check sampling_offsets: {model.transformer.encoder.layers[0].self_attn.sampling_offsets}")
    except Exception as e:
        print(f"Error checking model structure: {e}")

    # Load weights
    import pickle
    weights_path = '/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor/weights/groundingdino_swint_jittor.pkl'
    with open(weights_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Remap keys
    new_state_dict = {}
    bert_state_dict = {}
    for k, v in state_dict.items():
        # Ensure float32
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            v = v.astype(np.float32)
        
        if k.startswith('bert.'):
            bert_k = k.replace('bert.', '')
            if isinstance(v, np.ndarray):
                bert_state_dict[bert_k] = torch.from_numpy(v)
            else:
                bert_state_dict[bert_k] = v
            continue 
        
        new_k = k
        if k.startswith('module.'):
            new_k = k.replace('module.', '')
        if k.startswith('backbone.0.'):
            new_k = k.replace('backbone.0.', 'backbone.')
        elif k.startswith('transformer.level_embed'):
            new_k = k.replace('transformer.level_embed', 'level_embed')
        elif k.startswith('transformer.enc_output'):
            new_k = k.replace('transformer.enc_output', 'enc_output')
        elif k.startswith('transformer.tgt_embed'):
            new_k = k.replace('transformer.tgt_embed', 'tgt_embed')
        elif k.startswith('transformer.decoder.bbox_embed.'):
            new_k = k  # Keep as is
        elif k.startswith('transformer.enc_out_bbox_embed'):
            new_k = k  # Keep as is
        elif k.startswith('bbox_embed.'):
            # Skip model-level bbox_embed for now, as Jittor might not have it
            continue
            
        if 'in_proj_weight' in new_k:
            # Split into q, k, v
            weight = v
            chunk_size = weight.shape[0] // 3
            new_state_dict[new_k.replace('in_proj_weight', 'q_proj.weight')] = weight[:chunk_size]
            new_state_dict[new_k.replace('in_proj_weight', 'k_proj.weight')] = weight[chunk_size:2*chunk_size]
            new_state_dict[new_k.replace('in_proj_weight', 'v_proj.weight')] = weight[2*chunk_size:]
            continue
        elif 'in_proj_bias' in new_k:
            bias = v
            chunk_size = bias.shape[0] // 3
            new_state_dict[new_k.replace('in_proj_bias', 'q_proj.bias')] = bias[:chunk_size]
            new_state_dict[new_k.replace('in_proj_bias', 'k_proj.bias')] = bias[chunk_size:2*chunk_size]
            new_state_dict[new_k.replace('in_proj_bias', 'v_proj.bias')] = bias[2*chunk_size:]
            continue
            
        new_state_dict[new_k] = v
        
    # Filter state_dict to only include keys that exist in the model
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_keys}
    
    model.load_state_dict(filtered_state_dict)
    print("Weights loaded from groundingdino_swint_jittor.pkl (with key remapping and filtering)")

    # Diagnostic: Check for missing/unexpected keys
    weight_keys = set(filtered_state_dict.keys())

    missing = model_keys - weight_keys
    unexpected = weight_keys - model_keys

    print(f"Missing from weights: {len(missing)}")
    for k in list(missing)[:10]:
        print(f"  {k}")

    print(f"\nUnexpected in weights: {len(unexpected)}")
    for k in list(unexpected)[:10]:
        print(f"  {k}")
    
    # Load BERT weights
    if bert_state_dict:
        print(f"Loading {len(bert_state_dict)} BERT weights...")
        msg = text_encoder.bert.load_state_dict(bert_state_dict, strict=False)
        print(f"BERT weights loaded: {msg}")
    
    print("Setting model to eval mode...")
    model.eval()
    print("Model in eval mode.")
    
    print("Setting text_encoder to eval mode...")
    text_encoder.eval()
    print("Text encoder in eval mode.")
    
    # Check feat_map weights
    print(f"feat_map.weight stats: min={model.feat_map.weight.min().item()}, max={model.feat_map.weight.max().item()}, mean={model.feat_map.weight.mean().item()}")
    
    # Load 10 sample images from LVIS/val
    image_dir = '/root/shared-nvme/GroundingDINO-Light/LVIS/val'
    image_files = sorted(os.listdir(image_dir))[:10]  # First 10 images
    
    for i, image_file in enumerate(image_files):
        sample_image_path = os.path.join(image_dir, image_file)
        image_tensor, original_image = load_image(sample_image_path)
        print(f"image_tensor dtype: {image_tensor.dtype}")
        
        print(f"\n--- Image {i+1}: {sample_image_path} ---")
        
        # Text prompt
        text_prompt = "person, car, dog, cat, chair, bottle, table"
        
        print("Encoding text...")
        # Encode text
        text_features = text_encoder(text_prompt)
        print(f"Text encoded. encoded_text dtype: {text_features['encoded_text'].dtype}")
        
        # Run inference
        print("Running inference...")
        with jt.no_grad():
            outputs = model(image_tensor, text_dict=text_features)
        print("Inference finished.")
        
        # DEBUG CODE
        print(f"Output keys: {outputs.keys() if isinstance(outputs, dict) else type(outputs)}")
        if isinstance(outputs, dict):
            print(f"pred_logits shape: {outputs['pred_logits'].shape}")
            print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
            print(f"pred_logits dtype: {outputs['pred_logits'].dtype}")
            print(f"pred_boxes dtype: {outputs['pred_boxes'].dtype}")
            
            # Check sigmoid outputs
            probas_debug = jt.sigmoid(outputs['pred_logits'])
        
        # Process outputs
        pred_logits = outputs['pred_logits'][0] # [nq, 256]
        pred_boxes = outputs['pred_boxes'][0]   # [nq, 4]

        # Stats
        print(f"pred_logits shape: {pred_logits.shape}")
        print(f"pred_boxes shape: {pred_boxes.shape}")
        
        print(f"pred_boxes stats: min={pred_boxes.min().item():.5f}, max={pred_boxes.max().item():.5f}, mean={pred_boxes.mean().item():.5f}")
        
        cx = pred_boxes[:, 0]
        cy = pred_boxes[:, 1]
        w = pred_boxes[:, 2]
        h = pred_boxes[:, 3]
        
        print(f"cx stats: min={cx.min().item():.5f}, max={cx.max().item():.5f}, mean={cx.mean().item():.5f}")
        print(f"cy stats: min={cy.min().item():.5f}, max={cy.max().item():.5f}, mean={cy.mean().item():.5f}")
        print(f"w stats: min={w.min().item():.5f}, max={w.max().item():.5f}, mean={w.mean().item():.5f}")
        print(f"h stats: min={h.min().item():.5f}, max={h.max().item():.5f}, mean={h.mean().item():.5f}")

        # Top detection
        logits = jt.sigmoid(pred_logits)  # (nq, 256)
        max_scores = logits.max(dim=1) # (nq)
        top_score = max_scores.max(dim=0)
        top_idx = max_scores.argmax(dim=0)
        
        top_box = pred_boxes[top_idx]
        print(f"Top Detection: Score {top_score.item():.3f}, Box {top_box.numpy()}")
        
        # Get predictions above threshold
        # probas = pred_logits.softmax(-1)[0, :, :-1]
        # keep = probas.max(-1) > 0.35 # Lower threshold slightly for testing
        
        # Convert boolean mask to indices
        # keep_indices = jt.nonzero(keep)[0]  # Get indices where keep is True
        
        # Convert boxes from [0, 1] to image scale
        # Always process detections for now (avoid .item()/.numpy() issues)
        # bboxes_scaled = pred_boxes[0, keep_indices].numpy()
        # bboxes_scaled[:, 0::2] *= original_image.width
        # bboxes_scaled[:, 1::2] *= original_image.height
        
        # print(f"Text prompt: {text_prompt}")
        # print(f"Number of detections: {len(bboxes_scaled)}")
        
        # For each detection, print the class and box
        # for j, (box, logit) in enumerate(zip(bboxes_scaled, probas[keep_indices])):
        #     class_id = logit.argmax(0)
        #     if isinstance(class_id, tuple):
        #         class_id = class_id[0]
        #     score = logit[class_id.item()]
        #     print(f"Detection {j+1}: Class {class_id.item()}, Score {score.item():.3f}, Box {box}")

if __name__ == "__main__":
    main()