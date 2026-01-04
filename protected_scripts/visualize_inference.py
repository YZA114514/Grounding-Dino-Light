#!/usr/bin/env python3
"""
Visualization Script for Jittor GroundingDINO with LVIS Prompts
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import jittor as jt


# Add the project root to path
sys.path.append('/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor')

from jittor_implementation.train.config import TrainingConfig
from jittor_implementation.models import GroundingDINO, BERTWrapper
from jittor_implementation.data.transforms import build_transforms
from jittor_implementation.models.backbone.swin_transformer import SwinTransformer

def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    transform = build_transforms(is_train=False)
    image_tensor, _ = transform(image, {})
    image_tensor = jt.Var(image_tensor).float32()
    return image_tensor.unsqueeze(0), image

def draw_boxes(image, boxes, labels, scores):
    draw = ImageDraw.Draw(image)
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        # GroundingDINO outputs [cx, cy, w, h]
        cx, cy, w, h = box
        
        # Convert to [x0, y0, x1, y1] for PIL
        x0 = cx - w / 2
        y0 = cy - h / 2
        x1 = cx + w / 2
        y1 = cy + h / 2
        
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text = f"{label}: {score:.2f}"
        
        # Draw text background
        text_bbox = draw.textbbox((x0, y0), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x0, y0), text, fill="white", font=font)
        
    return image

def load_lvis_annotations(json_path):
    print(f"Loading LVIS annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Map category_id to name
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Map image filename to image_id
    # LVIS images have 'coco_url' like 'http://images.cocodataset.org/val2017/000000397133.jpg'
    # We'll map the filename (e.g., '000000397133.jpg') to image_id
    filename_to_id = {}
    for img in data['images']:
        filename = os.path.basename(img['coco_url'])
        filename_to_id[filename] = img['id']
        
    # Map image_id to list of category names present
    image_id_to_categories = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        cat_name = categories[cat_id]
        
        if img_id not in image_id_to_categories:
            image_id_to_categories[img_id] = set()
        image_id_to_categories[img_id].add(cat_name)
        
    return filename_to_id, image_id_to_categories

def main():
    # Set device
    jt.flags.use_cuda = 1
    
    output_dir = "inference_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = TrainingConfig()
    
    # Build Backbone (Swin-T)
    backbone = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(1, 2, 3),
    )
    backbone.num_channels = [192, 384, 768]
    
    # Load model
    model = GroundingDINO(
        backbone=backbone,
        num_queries=900,
        hidden_dim=256,
        num_feature_levels=4,
        nheads=8,
        two_stage_type="standard",
        two_stage_bbox_embed_share=False,  # Don't share encoder and decoder bbox_embed
        dec_pred_bbox_embed_share=False,   # Don't share bbox_embed across decoder layers
    )
    text_encoder = BERTWrapper(config.text_encoder_type)
    
    # Load weights
    import pickle
    weights_path = '/root/shared-nvme/GroundingDINO-Light/Grounding-Dino-Light/GroundingDINO_Jittor/weights/groundingdino_swint_jittor.pkl'
    print(f"Loading weights from {weights_path}...")
    with open(weights_path, 'rb') as f:
        state_dict = pickle.load(f)
        
    # Remap keys
    new_state_dict = {}
    bert_state_dict = {}
    for k, v in state_dict.items():
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
        if k.startswith('backbone.0.'):
            new_k = k.replace('backbone.0.', 'backbone.')
        elif k.startswith('transformer.level_embed'):
            new_k = k.replace('transformer.level_embed', 'level_embed')
        elif k.startswith('transformer.enc_output'):
            new_k = k.replace('transformer.enc_output', 'enc_output')
        elif k.startswith('transformer.tgt_embed'):
            new_k = k.replace('transformer.tgt_embed', 'tgt_embed')
        if 'in_proj_weight' in new_k:
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
        
    model.load_state_dict(new_state_dict)
    text_encoder.bert.load_state_dict(bert_state_dict, strict=False)
    
    model.eval()
    text_encoder.eval()
    
    # Load LVIS Data
    lvis_json_path = '/root/shared-nvme/GroundingDINO-Light/LVIS/lvis_v1_val.json'
    filename_to_id, image_id_to_categories = load_lvis_annotations(lvis_json_path)
    
    # Images
    image_dir = '/root/shared-nvme/GroundingDINO-Light/LVIS/val'
    image_files = sorted(os.listdir(image_dir))
    
    results_data = []
    
    # Process first 20 images
    count = 0
    for image_file in image_files:
        if count >= 20:
            break
            
        sample_image_path = os.path.join(image_dir, image_file)
        
        # Get prompt for this image
        img_id = filename_to_id.get(image_file)
        if img_id is None:
            print(f"Skipping {image_file}: ID not found in JSON")
            continue
            
        categories = image_id_to_categories.get(img_id)
        if not categories:
            print(f"Skipping {image_file}: No categories found")
            continue
            
        # Construct prompt
        # GroundingDINO typically uses dot-separated prompts
        # e.g. "cat . dog . chair ."
        # We replace underscores with spaces in category names just in case
        prompt_categories = [c.replace('_', ' ') for c in categories]
        text_prompt = " . ".join(prompt_categories) + " ."
        labels_list = prompt_categories # For mapping back
        
        print(f"Processing {image_file} with prompt: '{text_prompt}'")
        
        image_tensor, original_image = load_image(sample_image_path)
        
        # Encode text for this specific image
        with jt.no_grad():
            text_features = text_encoder(text_prompt)
            input_ids = text_encoder.tokenizer.encode(text_prompt, add_special_tokens=False)
            attention_mask = jt.array([1] * len(input_ids) + [0] * (256 - len(input_ids)))
            outputs = model(image_tensor, text_dict=text_features)
            
        # Get category mapping
        # text_features['cate_to_token_mask_list'] is a list of tensors (one per batch)
        # We have batch size 1
        cate_mask = text_features['cate_to_token_mask_list'][0] # [num_cats, num_tokens]
        if isinstance(cate_mask, jt.Var):
            cate_mask = cate_mask.numpy()
            
        # Create token_idx -> cat_idx mapping
        token_to_cat = {}
        num_cats, num_tokens = cate_mask.shape
        for cat_idx in range(num_cats):
            for token_idx in range(num_tokens):
                if cate_mask[cat_idx, token_idx]:
                    token_to_cat[token_idx] = cat_idx

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Debug: Check logit distribution
        print(f"pred_logits min: {pred_logits.min().item()}, max: {pred_logits.max().item()}")
        print(f"pred_logits after sigmoid: {jt.sigmoid(pred_logits).max().item()}")

        # Check if any query has high confidence
        max_scores = jt.sigmoid(pred_logits).max(-1)[0]  # [bs, 900]
        top10_scores = jt.topk(max_scores[0], 10)[0]
        print(f"Top 10 confidence scores: {top10_scores.numpy()}")

        # Mask out padding tokens
        pred_logits_masked = pred_logits[0].clone()
        pred_logits_masked[:, attention_mask == 0] = -1e9

        probas = jt.sigmoid(pred_logits_masked)[:, :-1]  # Use sigmoid for multi-label classification
        keep = probas.max(-1) > 0.25  # Lower threshold from 0.35 to 0.25
        keep_indices = jt.nonzero(keep)
        
        if keep_indices.shape[0] == 0:
            print(f"No detections for {image_file}")
            # Still save the image to show we tried
            # continue 
            keep_indices = []
        else:
            keep_indices = keep_indices[:, 0]
        
        if len(keep_indices) == 0:
            # Handle empty detections
            bboxes_scaled = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            detected_labels = []
        else:
            bboxes_scaled = pred_boxes[0, keep_indices].numpy()
            bboxes_scaled[:, 0::2] *= original_image.width
            bboxes_scaled[:, 1::2] *= original_image.height
            
            scores = probas[keep_indices].max(-1).numpy()
            class_ids = probas[keep_indices].argmax(-1)
            if isinstance(class_ids, tuple):
                class_ids = class_ids[0]  # indices
            class_ids = class_ids.numpy()
            
            detected_labels = []
            for i in range(len(class_ids)):
                cid = class_ids[i]
                if cid in token_to_cat:
                    cat_idx = token_to_cat[cid]
                    if cat_idx < len(labels_list):
                        detected_labels.append(labels_list[cat_idx])
                    else:
                        detected_labels.append(f"Cat_{cat_idx}")
                else:
                    # Decode the token
                    if cid < len(input_ids):
                        token_id = input_ids[cid]
                        word = text_encoder.tokenizer.decode(token_id)
                    else:
                        word = "[PAD]"
                    detected_labels.append(word)
        
        # Draw
        vis_image = draw_boxes(original_image.copy(), bboxes_scaled, detected_labels, scores)
        vis_path = os.path.join(output_dir, f"vis_{image_file}")
        vis_image.save(vis_path)
        
        # Save data
        img_data = {
            "image_file": image_file,
            "prompt": text_prompt,
            "detections": []
        }
        for box, label, score in zip(bboxes_scaled.tolist(), detected_labels, scores.tolist()):
            img_data["detections"].append({
                "box": box, # [cx, cy, w, h]
                "label": label,
                "score": score
            })
        results_data.append(img_data)
        count += 1
        
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
