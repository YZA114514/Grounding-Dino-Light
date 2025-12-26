#!/usr/bin/env python3
"""
Debug script for AP=0 issue
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the project root to path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

import jittor as jt
from jittor_implementation.models.groundingdino import GroundingDINO
from jittor_implementation.models.backbone.swin_transformer import build_swin_transformer

def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    # Simple preprocessing - resize and normalize
    image = image.resize((800, 800))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1))
    
    # Normalize
    mean = jt.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = jt.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0), image

def main():
    print("Debugging AP=0 issue...")
    
    # Set device
    jt.flags.use_cuda = 1
    print("Using GPU")
    
    # Create backbone
    print("Building backbone...")
    backbone = build_swin_transformer(
        modelname="swin_T_224_1k",
        pretrain_img_size=224,
        out_indices=(1, 2, 3),
        dilation=False,
    )
    
    # Create model
    print("Creating model...")
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
        
        # Clean and map weights (simplified version)
        cleaned = {}
        for k, v in weights.items():
            new_k = k
            if new_k.startswith('module.'):
                new_k = new_k[7:]
            
            if new_k.startswith('backbone.0.'):
                new_k = 'backbone.' + new_k[11:]
            
            if new_k == 'transformer.level_embed':
                new_k = 'level_embed'
            
            if new_k == 'transformer.tgt_embed.weight':
                new_k = 'tgt_embed.weight'
            
            if new_k.startswith('transformer.enc_output'):
                new_k = new_k.replace('transformer.enc_output', 'enc_output')
            
            if new_k.startswith('bbox_embed.'):
                new_k = 'transformer.decoder.' + new_k
            
            cleaned[new_k] = v
        
        # Handle BERT weights separately
        bert_weights = {}
        other_weights = {}
        for k, v in cleaned.items():
            if k.startswith('bert.'):
                bert_weights[k] = v
            else:
                other_weights[k] = v
        
        # Handle in_proj conversion
        converted_weights = {}
        for k, v in other_weights.items():
            if '.in_proj_weight' in k:
                d = v.shape[0] // 3
                base_key = k.replace('.in_proj_weight', '.')
                converted_weights[base_key + 'q_proj.weight'] = v[:d, :]
                converted_weights[base_key + 'k_proj.weight'] = v[d:2*d, :]
                converted_weights[base_key + 'v_proj.weight'] = v[2*d:, :]
            elif '.in_proj_bias' in k:
                d = v.shape[0] // 3
                base_key = k.replace('.in_proj_bias', '.')
                converted_weights[base_key + 'q_proj.bias'] = v[:d]
                converted_weights[base_key + 'k_proj.bias'] = v[d:2*d]
                converted_weights[base_key + 'v_proj.bias'] = v[2*d:]
            else:
                converted_weights[k] = v
        
        # Load weights
        model_state = model.state_dict()
        loaded = 0
        for k, v in converted_weights.items():
            if k in model_state:
                if model_state[k].shape == tuple(v.shape):
                    model_state[k] = jt.array(v)
                    loaded += 1
        
        model.load_state_dict(model_state)
        print(f"Loaded {loaded} weights")
        
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
    print("Model loaded and set to eval mode")
    
    # Test with a sample image
    image_dir = '/root/shared-nvme/GroundingDINO-Light/LVIS/val'
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:3]
        
        for i, image_file in enumerate(image_files):
            print(f"\n--- Testing Image {i+1}: {image_file} ---")
            
            image_path = os.path.join(image_dir, image_file)
            image_tensor, original_image = load_image(image_path)
            
            # Text prompt
            text_prompt = "person. car. dog. cat. chair. bottle. table."
            
            print("Running inference...")
            with jt.no_grad():
                outputs = model(image_tensor, captions=[text_prompt])
            
            print(f"Output keys: {outputs.keys()}")
            print(f"pred_logits shape: {outputs['pred_logits'].shape}")
            print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
            
            # Analyze outputs
            pred_logits = outputs['pred_logits'][0]  # [nq, max_text_len]
            pred_boxes = outputs['pred_boxes'][0]    # [nq, 4]
            
            print(f"pred_logits range: [{pred_logits.min().item():.4f}, {pred_logits.max().item():.4f}]")
            print(f"pred_boxes range: [{pred_boxes.min().item():.4f}, {pred_boxes.max().item():.4f}]")
            
            # Apply sigmoid to get probabilities - with fix for extreme values
            pred_logits_clamped = jt.clamp(pred_logits, -10.0, 10.0)
            pred_probs = jt.sigmoid(pred_logits_clamped)
            max_probs, pred_labels = jt.argmax(pred_probs, dim=-1, keepdims=False)
            max_probs = max_probs[0] if len(max_probs.shape) > 0 else max_probs
            pred_labels = pred_labels[0] if len(pred_labels.shape) > 0 else pred_labels
            
            print(f"Max probabilities range: [{max_probs.min().item():.4f}, {max_probs.max().item():.4f}]")
            print(f"Mean max probability: {max_probs.mean().item():.4f}")
            
            # Count predictions above threshold
            threshold = 0.25
            high_conf_mask = max_probs > threshold
            high_conf_count = jt.sum(high_conf_mask).item()
            print(f"Predictions above {threshold}: {high_conf_count}/{len(max_probs)}")
            
            if high_conf_count > 0:
                print("Top 5 predictions:")
                top_indices = jt.argsort(max_probs, descending=True)[:5]
                for idx in top_indices:
                    score = max_probs[idx].item()
                    label = pred_labels[idx].item()
                    box = pred_boxes[idx].numpy()
                    print(f"  Score: {score:.3f}, Label: {label}, Box: {box}")
            else:
                print("NO HIGH CONFIDENCE PREDICTIONS - This is likely the AP=0 issue!")
                
                # Debug: check if logits are all negative
                print(f"Percentage of negative logits: {jt.sum(pred_logits < 0).item() / pred_logits.numel() * 100:.1f}%")
                
                # Debug: check if boxes are reasonable
                cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
                print(f"Box stats - cx: [{cx.min().item():.3f}, {cx.max().item():.3f}], "
                      f"cy: [{cy.min().item():.3f}, {cy.max().item():.3f}], "
                      f"w: [{w.min().item():.3f}, {w.max().item():.3f}], "
                      f"h: [{h.min().item():.3f}, {h.max().item():.3f}]")
    
    else:
        print(f"Image directory {image_dir} not found")

if __name__ == "__main__":
    main()
