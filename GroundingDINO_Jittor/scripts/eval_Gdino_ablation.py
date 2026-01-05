#!/usr/bin/env python
"""
Grounding DINO Ablation Experiments

Supports multiple ablation experiments through command-line arguments:

1. --ablation no_text_cross_attn: Remove text cross-attention in decoder (novel ablation)
2. --ablation single_scale --scale_idx N: Use only single scale features (N=0,1,2,3)
3. --ablation random_text: Replace text embeddings with random noise
4. --ablation fewer_layers --num_layers N: Use only N decoder layers (default 3)
5. --ablation prompt_style --style MODE: Change text prompt format
   - original: "traffic light"
   - with_article: "a traffic light"
   - with_photo: "a photo of a traffic light"

Usage:
    python scripts/eval_Gdino_ablation.py --ablation no_text_cross_attn --num_images 100
    python scripts/eval_Gdino_ablation.py --ablation single_scale --scale_idx 0 --num_images 100
    python scripts/eval_Gdino_ablation.py --ablation random_text --num_images 100
    python scripts/eval_Gdino_ablation.py --ablation fewer_layers --num_layers 3 --num_images 100
    python scripts/eval_Gdino_ablation.py --ablation prompt_style --style with_photo --num_images 100
"""
import sys
import numpy.core.numeric
sys.modules['numpy._core.numeric'] = numpy.core.numeric

import os
import sys
import json
import argparse
import time
import logging
from datetime import datetime

# Force HuggingFace offline mode (skip network checks for cached models)
os.environ['HF_HUB_OFFLINE'] = '1'

# Set GPU before importing jittor (check if GPU is available)
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# Check if CUDA_VISIBLE_DEVICES is set to empty (CPU mode) or invalid GPU
if cuda_visible == '' or not cuda_visible.strip():
    cuda_visible = ''  # Force CPU mode
else:
    # Check if the specified GPU exists
    try:
        gpu_id = int(cuda_visible)
        # Simple check: try to get GPU info
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
        available_gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        if gpu_id not in available_gpus:
            print(f"Warning: GPU {gpu_id} not available. Available GPUs: {available_gpus}")
            cuda_visible = ''  # Fall back to CPU
    except:
        print("Warning: Could not check GPU availability, falling back to CPU")
        cuda_visible = ''

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'jittor_implementation'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

PT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'GroundingDINO-main')
sys.path.insert(0, PT_DIR)

import numpy as np
import jittor as jt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

from quick_test_zeroshot import load_model, preprocess_image, Config
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from transformers import AutoTokenizer

# Set CUDA flag based on GPU availability
if cuda_visible == '':
    jt.flags.use_cuda = 0
    print("Using CPU mode")
else:
    jt.flags.use_cuda = 1
    print(f"Using GPU mode (GPU {cuda_visible})")


def archive_previous_run(output_dir):
    """Archive previous run outputs to timestamped folder if they exist."""
    predictions_file = os.path.join(output_dir, 'predictions.jsonl')
    progress_file = os.path.join(output_dir, 'progress.json')

    # Check if there's a previous run to archive
    if os.path.exists(predictions_file) or os.path.exists(progress_file):
        # Create archive folder with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(output_dir, f'archive_{timestamp}')
        os.makedirs(archive_dir, exist_ok=True)

        # Move files to archive
        files_to_archive = [
            'predictions.jsonl',
            'progress.json',
            'lvis_predictions.json',
            'lvis_zeroshot_results.json'
        ]
        moved = []
        for fname in files_to_archive:
            src = os.path.join(output_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(archive_dir, fname)
                os.rename(src, dst)
                moved.append(fname)

        if moved:
            print(f"  Archived previous run to: {archive_dir}")
            print(f"  Files moved: {', '.join(moved)}")


def setup_logging(output_dir):
    """Set up logging to both console and file."""
    log_file = os.path.join(output_dir, 'eval.log')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    print(f"Logging to: {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Grounding DINO Ablation Experiments')

    # Basic evaluation arguments
    parser.add_argument('--checkpoint', type=str,
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Path to Jittor checkpoint')
    parser.add_argument('--use_full_val', action='store_true',
                        help='Use full LVIS val set instead of minival')
    parser.add_argument('--lvis_ann', type=str,
                        default=None,
                        help='Path to LVIS annotation file (auto-detected based on --use_full_val)')
    parser.add_argument('--image_dir', type=str,
                        default=None,
                        help='Path to LVIS validation images (auto-detected based on --use_full_val)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to evaluate (0 for all)')
    parser.add_argument('--full', action='store_true',
                        help='Evaluate on full validation set')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='Number of categories per batch (to fit BERT 512 token limit)')
    parser.add_argument('--box_threshold', type=float, default=0.1,
                        help='Box score threshold (original: 0.1)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index for image subset')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for image subset')
    parser.add_argument('--checkpoint_interval', type=int, default=250,
                        help='Save checkpoint every N images')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint if available')
    parser.add_argument('--eval_interval', type=int, default=0,
                        help='Evaluate AP every N images (0 = disabled)')
    parser.add_argument('--ultra_optimized', action='store_true',
                        help='Use ultra-optimized inference')

    # Ablation experiment arguments
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_text_cross_attn', 'single_scale', 'random_text', 'fewer_layers', 'prompt_style'],
                        help='Type of ablation experiment to run')
    parser.add_argument('--scale_idx', type=int, default=2,
                        help='Scale index for single_scale ablation (0=high-res, 3=low-res)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of decoder layers for fewer_layers ablation')
    parser.add_argument('--style', type=str, default='original',
                        choices=['original', 'with_article', 'with_photo'],
                        help='Text prompt style for prompt_style ablation')

    args = parser.parse_args()

    # Validation
    if args.ablation == 'single_scale' and args.scale_idx not in [0, 1, 2, 3]:
        parser.error("--scale_idx must be 0, 1, 2, or 3")

    if args.ablation == 'fewer_layers' and args.num_layers not in range(1, 7):
        parser.error("--num_layers must be between 1 and 6")

    return args


def load_checkpoint(output_dir):
    """Load checkpoint and return starting index."""
    progress_file = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_idx = progress.get('last_idx', 0) + 1
        timestamp = progress.get('timestamp', 'unknown')
        print(f"  Found checkpoint: last processed image index {progress.get('last_idx', 0)} (at {timestamp})")
        print(f"  Resuming from image index {start_idx}")
        return start_idx
    return 0


def save_checkpoint(output_dir, last_idx):
    """Save progress checkpoint with atomic write."""
    progress_file = os.path.join(output_dir, 'progress.json')
    tmp_file = progress_file + '.tmp'

    progress = {
        'last_idx': last_idx,
        'timestamp': datetime.now().isoformat(),
        'total_processed': last_idx + 1
    }

    # Atomic write: write to .tmp then rename
    with open(tmp_file, 'w') as f:
        json.dump(progress, f, indent=2)
    os.rename(tmp_file, progress_file)


def write_predictions_to_jsonl(output_dir, predictions):
    """Append predictions to JSONL file."""
    jsonl_file = os.path.join(output_dir, 'predictions.jsonl')
    with open(jsonl_file, 'a') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
        f.flush()


def load_predictions_from_jsonl(output_dir):
    """Load all predictions from JSONL file."""
    jsonl_file = os.path.join(output_dir, 'predictions.jsonl')
    if not os.path.exists(jsonl_file):
        return []

    predictions = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line.strip()))
    return predictions


def apply_ablation_direct(model, ablation_type, ablation_params):
    """Apply ablation experiments directly to the model object (in-memory only)."""

    if ablation_type == 'no_text_cross_attn':
        # Remove text cross-attention in decoder
        apply_no_text_cross_attn_ablation(model)

    elif ablation_type == 'single_scale':
        # Use only single scale features
        apply_single_scale_ablation(model, ablation_params.get('scale_idx', 2))

    elif ablation_type == 'random_text':
        # Replace text embeddings with random noise
        apply_random_text_ablation(model)

    elif ablation_type == 'fewer_layers':
        # Use fewer decoder layers
        apply_fewer_layers_ablation(model, ablation_params.get('num_layers', 3))

    # prompt_style ablation is applied at the data level, not model level

    # Store ablation type for later use in encoder
    model._ablation_type = ablation_type


def apply_no_enhancer_ablation(model):
    """Skip Feature Enhancer by bypassing fusion layers in encoder."""
    original_encoder_execute = model.transformer.encoder.execute

    def ablated_encoder_execute(src, pos, spatial_shapes, level_start_index, valid_ratios,
                              key_padding_mask, memory_text=None, text_attention_mask=None,
                              pos_text=None, text_self_attention_masks=None, position_ids=None):
        """Modified encoder that skips fusion layers."""
        output = src

        # Generate reference points (same as original)
        if model.transformer.encoder.num_layers > 0:
            reference_points = model.transformer.encoder.get_reference_points(
                spatial_shapes, valid_ratios, device=None
            )

        # Generate text position encoding (same as original)
        if model.transformer.encoder.text_layers:
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = jt.arange(n_text).float().unsqueeze(0).unsqueeze(-1)
                pos_text = pos_text.repeat(bs, 1, 1)
                pos_text = jt.zeros(bs, n_text, model.hidden_dim)
            if position_ids is not None:
                pos_text = jt.zeros(bs, n_text, model.hidden_dim)

        # Main processing loop WITHOUT fusion layers (skip BiAttentionBlock)
        for layer_id, layer in enumerate(model.transformer.encoder.layers):
            # SKIP: Feature fusion (BiAttentionBlock) - this is the "enhancer" we want to ablate

            # Apply text enhancement layers (keep text processing)
            if model.transformer.encoder.text_layers:
                memory_text = model.transformer.encoder.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=jt.logical_not(text_self_attention_masks) if text_self_attention_masks is not None else None,
                    src_key_padding_mask=text_attention_mask,
                    pos=pos_text.transpose(0, 1) if pos_text is not None else None,
                ).transpose(0, 1)

            # Apply visual encoding (keep deformable attention)
            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )

        return output, memory_text

    # Replace encoder execute method
    model.transformer.encoder.execute = ablated_encoder_execute


def apply_single_scale_ablation(model, scale_idx):
    """Use only a single scale of features by zeroing out other scales."""
    original_encode_image_projection = model.encode_image_projection

    def single_scale_encode_image_projection(samples):
        """Modified image projection that zeros out all scales except the specified one."""
        # Call original method
        projected_features = original_encode_image_projection(samples)

        # Extract features
        src_flatten = projected_features['src_flatten']
        mask_flatten = projected_features['mask_flatten']
        lvl_pos_embed_flatten = projected_features['lvl_pos_embed_flatten']
        spatial_shapes = projected_features['spatial_shapes']
        level_start_index = projected_features['level_start_index']
        valid_ratios = projected_features['valid_ratios']

        # Zero out features from all scales except the specified one
        total_features = 0
        for i, (h, w) in enumerate(spatial_shapes):
            scale_size = int(h * w)
            start_idx = int(level_start_index[i])
            end_idx = start_idx + scale_size

            if i != scale_idx:
                # Zero out this scale's features (Jittor uses direct assignment)
                src_flatten[:, start_idx:end_idx, :] = 0
                mask_flatten[:, start_idx:end_idx] = 0
                lvl_pos_embed_flatten[:, start_idx:end_idx, :] = 0

        return {
            'src_flatten': src_flatten,
            'mask_flatten': mask_flatten,
            'lvl_pos_embed_flatten': lvl_pos_embed_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
        }

    # Replace method
    model.encode_image_projection = single_scale_encode_image_projection


def apply_random_text_ablation(model):
    """Replace text embeddings with random noise."""
    # Store original method for shape reference
    original_encode_text = model.encode_text

    def random_text_encode_text(captions):
        """Return random text embeddings instead of real ones."""
        # Get a sample text_dict to know the shape
        real_text_dict = original_encode_text(captions)

        # Replace with random noise of same shape
        bs = len(captions)
        random_text = jt.randn_like(real_text_dict['encoded_text'])

        # Keep other fields the same
        return {
            'encoded_text': random_text,
            'text_token_mask': real_text_dict['text_token_mask'],
            'position_ids': real_text_dict['position_ids'],
            'text_self_attention_masks': real_text_dict['text_self_attention_masks'],
        }

    # Replace method
    model.encode_text = random_text_encode_text


def apply_no_text_cross_attn_ablation(model):
    """Remove text cross-attention in decoder layers."""
    from jittor_implementation.models.transformer.decoder import gen_sineembed_for_position, inverse_sigmoid

    original_decoder_execute = model.transformer.decoder.execute

    def no_text_cross_attn_decoder_execute(tgt, memory, tgt_mask=None, memory_mask=None,
                                         tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                         pos=None, refpoints_unsigmoid=None, level_start_index=None,
                                         spatial_shapes=None, valid_ratios=None, memory_text=None,
                                         text_attention_mask=None):
        """Modified decoder that skips text cross-attention in all layers."""
        output = tgt

        intermediate = []
        reference_points = jt.sigmoid(refpoints_unsigmoid)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(model.transformer.decoder.layers):
            # Prepare reference points input (same as original)
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points.unsqueeze(2)
                    * jt.concat([valid_ratios, valid_ratios], dim=-1).unsqueeze(0)
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points.unsqueeze(2) * valid_ratios.unsqueeze(0)

            # Generate query sine embed (same as original)
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )

            # Conditional query position encoding (same as original)
            raw_query_pos = model.transformer.decoder.ref_point_head(query_sine_embed)
            pos_scale = model.transformer.decoder.query_scale(output) if model.transformer.decoder.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # Call layer with text cross-attention DISABLED
            # We'll modify the layer call to skip text cross-attention
            output = layer_no_text_cross_attn(
                layer=layer,
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                # Skip text inputs
                memory_text=None,  # Pass None to disable text cross-attention
                text_attention_mask=None,
                # Keep vision inputs
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )

            # Iterative bbox refinement (same as original)
            if model.transformer.decoder.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = model.transformer.decoder.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = jt.sigmoid(outputs_unsig)

                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(model.transformer.decoder.norm(output))

        # Return format: transpose to [bs, nq, d_model]
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]

    # Helper function to call layer without text cross-attention
    def layer_no_text_cross_attn(layer, tgt, tgt_query_pos=None, tgt_query_sine_embed=None,
                                tgt_key_padding_mask=None, tgt_reference_points=None,
                                memory_text=None, text_attention_mask=None,  # These are ignored
                                memory=None, memory_key_padding_mask=None,
                                memory_level_start_index=None, memory_spatial_shapes=None,
                                memory_pos=None, self_attn_mask=None, cross_attn_mask=None):

        # 1. Self-attention (same as original)
        if layer.self_attn is not None:
            q = k = layer.with_pos_embed(tgt, tgt_query_pos)
            tgt2, _ = layer.self_attn(q, k, tgt, attn_mask=self_attn_mask)
            tgt = tgt + layer.dropout2(tgt2)
            tgt = layer.norm2(tgt)

        # SKIP: Text cross-attention (this is the ablation!)

        # 3. Deformable cross-attention (with vision features - same as original)
        tgt2 = layer.cross_attn(
            query=layer.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)

        tgt = tgt + layer.dropout1(tgt2)
        tgt = layer.norm1(tgt)

        # 4. FFN (same as original)
        tgt = layer.forward_ffn(tgt)

        return tgt

    # Replace decoder execute method
    model.transformer.decoder.execute = no_text_cross_attn_decoder_execute


def apply_fewer_layers_ablation(model, num_layers):
    """Use fewer decoder layers, but return 6 intermediate layers (pad with last layer)."""
    original_decoder_execute = model.transformer.decoder.execute

    def fewer_layers_decoder_execute(tgt, memory, memory_key_padding_mask, pos, refpoints_unsigmoid,
                                   level_start_index, spatial_shapes, valid_ratios, memory_text,
                                   text_attention_mask):
        """Modified decoder that uses only specified number of layers but returns 6 intermediates."""
        # Use only the first num_layers layers
        layers_to_use = model.transformer.decoder.layers[:num_layers]

        # Initialize outputs
        output = tgt
        reference_points = refpoints_unsigmoid

        # Keep track of intermediate outputs for bbox/class heads (need exactly 6)
        intermediate = []
        intermediate_reference_points = []

        # Compute actual layers
        for layer_idx, layer in enumerate(layers_to_use):
            if model.transformer.decoder.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

            output, reference_points = layer(
                output, memory, memory_key_padding_mask, pos,
                reference_points, level_start_index, spatial_shapes, valid_ratios,
                memory_text, text_attention_mask
            )

        # Pad to exactly 6 intermediate layers (required by main model)
        if model.transformer.decoder.return_intermediate:
            # Add final computed output
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

            # Pad remaining layers with the last computed output (inactive layers)
            while len(intermediate) < 6:
                intermediate.append(output.clone())
                intermediate_reference_points.append(reference_points.clone())

        if model.transformer.decoder.return_intermediate:
            return intermediate, intermediate_reference_points
        else:
            return output, reference_points

    # Replace decoder execute method
    model.transformer.decoder.execute = fewer_layers_decoder_execute


def build_category_batches_ablation(categories, tokenizer, batch_size=60, max_text_len=256, ablation_type=None, ablation_params=None):
    """Build batches of categories with optional ablation for prompt style."""
    if ablation_type == 'prompt_style':
        style = ablation_params.get('style', 'original')
        categories = modify_category_names(categories, style)

    num_batches = (len(categories) + batch_size - 1) // batch_size
    batch_info = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(categories))
        batch_cats = categories[start:end]
        batch_cat_names = [cat['name'] for cat in batch_cats]  # Already modified if prompt_style
        batch_cat_ids = [cat['id'] for cat in batch_cats]

        # Build caption and token spans
        caption, cat2tokenspan = build_captions_and_token_span(batch_cat_names, force_lowercase=True)
        tokenized = tokenizer(caption, return_tensors="pt")
        tokenspanlist = [cat2tokenspan.get(name, []) for name in batch_cat_names]
        positive_map = create_positive_map_from_span(tokenized, tokenspanlist, max_text_len=max_text_len)

        batch_info.append({
            'caption': caption,
            'cat_ids': batch_cat_ids,
            'cat_names': batch_cat_names,
            'positive_map': positive_map.numpy(),
            'num_tokens': tokenized['input_ids'].shape[1]
        })

    return batch_info


def modify_category_names(categories, style):
    """Modify category names based on prompt style ablation."""
    modified_cats = []

    for cat in categories:
        name = cat['name'].replace('_', ' ')
        if style == 'original':
            new_name = name
        elif style == 'with_article':
            new_name = f"a {name}"
        elif style == 'with_photo':
            new_name = f"a photo of a {name}"
        else:
            new_name = name

        modified_cat = cat.copy()
        modified_cat['name'] = new_name
        modified_cats.append(modified_cat)

    return modified_cats


def run_inference_batched_ultra_optimized(model, img_tensor, batch_info, text_cache, orig_size, box_threshold=0.1, num_select=None):
    """Ultra-optimized inference: minimize GPU-CPU syncs by keeping everything on GPU until final sync.

    Returns all predictions above threshold. The num_select parameter is deprecated and ignored.
    """
    orig_w, orig_h = orig_size

    # Accumulators (stay on GPU)
    all_scores = []
    all_boxes = []
    all_cat_ids = []

    with jt.no_grad():
        projected_features = model.encode_image_projection(img_tensor)

    for batch_idx, batch in enumerate(batch_info):
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']

        with jt.no_grad():
            text_dict = model.encode_text([batch['caption']])
            outputs = model(
                img_tensor,
                text_dict=text_dict,
                vision_features=projected_features
            )

        # GPU: sigmoid + bmm
        prob_to_token_gpu = jt.sigmoid(outputs['pred_logits'][0])
        positive_map_gpu = jt.array(positive_map_np.T).unsqueeze(0)
        prob_to_label_gpu = jt.nn.bmm(
            prob_to_token_gpu.unsqueeze(0),
            positive_map_gpu
        ).squeeze(0)  # (900, num_cats) - STAY ON GPU

        pred_boxes_gpu = outputs['pred_boxes'][0]  # (900, 4)

        # GPU: threshold filtering
        mask = prob_to_label_gpu >= box_threshold
        indices = jt.where(mask)
        q_idxs = indices[0]
        c_idxs = indices[1]

        if q_idxs.shape[0] == 0:
            continue

        scores = prob_to_label_gpu[q_idxs, c_idxs]
        boxes = pred_boxes_gpu[q_idxs]  # (N, 4) - STAY ON GPU

        # GPU: coordinate conversion
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = jt.clamp((cx - w/2) * orig_w, 0, orig_w)
        y1 = jt.clamp((cy - h/2) * orig_h, 0, orig_h)
        x2 = jt.clamp((cx + w/2) * orig_w, 0, orig_w)
        y2 = jt.clamp((cy + h/2) * orig_h, 0, orig_h)
        bw = x2 - x1
        bh = y2 - y1

        # GPU: valid filter
        valid = (bw > 0) & (bh > 0)
        scores = scores[valid]
        x1 = x1[valid]
        y1 = y1[valid]
        bw = bw[valid]
        bh = bh[valid]
        c_idxs = c_idxs[valid]

        # Map category indices to IDs (on GPU)
        cat_ids_gpu = jt.array(batch_cat_ids)[c_idxs]

        # Accumulate (still GPU tensors)
        all_scores.append(scores)
        all_boxes.append(jt.stack([x1, y1, bw, bh], dim=1))
        all_cat_ids.append(cat_ids_gpu)

    if not all_scores:
        return []

    # Concatenate on GPU
    all_scores = jt.concat(all_scores)
    all_boxes = jt.concat(all_boxes)
    all_cat_ids = jt.concat(all_cat_ids)

    # Return all predictions above threshold (num_select parameter is deprecated)
    # SINGLE SYNC: transfer all results to CPU
    scores_np = all_scores.numpy()
    boxes_np = all_boxes.numpy()
    cat_ids_np = all_cat_ids.numpy().astype(int)

    # Build output dicts (CPU)
    return [
        {'category_id': cid, 'bbox': box, 'score': sc}
        for cid, box, sc in zip(
            cat_ids_np.tolist(),
            boxes_np.tolist(),
            scores_np.tolist()
        )
    ]


def run_inference_batched_optimized(model, img_tensor, batch_info, text_cache, orig_size, box_threshold=0.1, num_select=None):
    """Optimized inference using cached projection features and precomputed text embeddings.

    Returns all predictions above threshold. The num_select parameter is deprecated and ignored.
    """
    orig_w, orig_h = orig_size
    all_predictions = []

    # Extract projection features once for this image (backbone + projection only)
    with jt.no_grad():
        projected_features = model.encode_image_projection(img_tensor)

    # Run encoder + decoder with real text for each batch
    for batch_idx, batch in enumerate(batch_info):
        positive_map_np = batch['positive_map']
        batch_cat_ids = batch['cat_ids']

        # Encode text fresh for this batch (required for proper text-vision fusion)
        with jt.no_grad():
            if text_cache is None:
                # Fresh text encoding for each batch
                text_dict = model.encode_text([batch['caption']])
            else:
                # Use cached text (legacy mode)
                text_dict = text_cache[batch_idx]

            outputs = model(
                img_tensor,
                text_dict=text_dict,
                vision_features=projected_features
            )

        # Vectorized computation: Convert logits to probabilities and map to labels (GPU sigmoid + bmm):
        prob_to_token_gpu = jt.sigmoid(outputs['pred_logits'][0])  # GPU sigmoid
        positive_map_gpu = jt.array(positive_map_np.T).unsqueeze(0)  # (1, 95, num_cats)
        prob_to_label = jt.nn.bmm(
            prob_to_token_gpu.unsqueeze(0),  # (1, 900, 95)
            positive_map_gpu                  # (1, 95, num_cats)
        ).squeeze(0).numpy()                  # (900, num_cats) → numpy
        pred_boxes_gpu = outputs['pred_boxes'][0]

        # Vectorized filtering and box conversion
        threshold = box_threshold
        q_idxs, c_idxs = np.where(prob_to_label >= threshold)
        scores = prob_to_label[q_idxs, c_idxs]

        if len(scores) == 0:
            continue

        # Vectorized box conversion
        boxes = pred_boxes_gpu[q_idxs].numpy()
        cx, cy, w, h = boxes.T

        # Convert to absolute coordinates (vectorized)
        x1 = (cx - w/2) * orig_w
        y1 = (cy - h/2) * orig_h
        x2 = (cx + w/2) * orig_w
        y2 = (cy + h/2) * orig_h

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # Convert to [x, y, w, h] format
        bw = x2 - x1
        bh = y2 - y1

        # Filter valid boxes
        valid = (bw > 0) & (bh > 0)
        q_idxs = q_idxs[valid]
        c_idxs = c_idxs[valid]
        scores = scores[valid]
        x1 = x1[valid]
        y1 = y1[valid]
        bw = bw[valid]
        bh = bh[valid]

        cat_ids_arr = np.array(batch_cat_ids)[c_idxs]
        batch_preds = [
            {'category_id': cid, 'bbox': box, 'score': sc}
            for cid, box, sc in zip(
                cat_ids_arr.tolist(),
                np.column_stack([x1, y1, bw, bh]).tolist(),
                scores.tolist()
            )
        ]

        all_predictions.extend(batch_preds)

    # Return all predictions above threshold (num_select parameter is deprecated)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    return all_predictions


def evaluate_with_lvis(predictions, lvis_data, image_ids, categories, output_dir):
    """Run official LVIS evaluation using LVISEval (handles federated annotations correctly)."""
    from lvis import LVIS, LVISResults, LVISEval

    # Handle zero predictions case (ablation experiments may produce no predictions)
    if not predictions:
        print("Warning: No predictions generated, skipping LVIS evaluation")
        # Return zero metrics
        gt_cat_ids = set(ann['category_id'] for ann in lvis_data['annotations'] if ann['image_id'] in image_ids)
        return {
            'AP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0,
            'APr': 0.0,
            'APc': 0.0,
            'APf': 0.0,
            'n_rare_cats': len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'r']),
            'n_common_cats': len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'c']),
            'n_freq_cats': len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'f']),
        }

    # Build ground truth in LVIS format (use original full annotations)
    lvis_gt_dict = lvis_data  # Use the original full LVIS data

    # Filter images to our subset
    subset_images = [img for img in lvis_data['images'] if img['id'] in image_ids]

    # Filter annotations to our subset
    subset_annotations = [ann for ann in lvis_data['annotations'] if ann['image_id'] in image_ids]

    # Create subset GT dict
    lvis_gt_subset = {
        'info': {'description': 'LVIS v1 validation subset', 'date_created': datetime.now().isoformat()},
        'licenses': lvis_data.get('licenses', []),
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': categories  # Use all categories (federated annotations)
    }

    # Save subset GT and predictions
    gt_file = os.path.join(output_dir, 'lvis_gt_subset.json')
    pred_file = os.path.join(output_dir, 'lvis_predictions.json')

    with open(gt_file, 'w') as f:
        json.dump(lvis_gt_subset, f)
    with open(pred_file, 'w') as f:
        json.dump(predictions, f)

    # Load with LVIS API
    lvis_gt = LVIS(gt_file)
    lvis_dt = LVISResults(lvis_gt, pred_file)

    # Run evaluation
    lvis_eval = LVISEval(lvis_gt, lvis_dt, 'bbox')
    lvis_eval.run()
    lvis_eval.print_results()

    # Extract results (LVISEval doesn't provide stats array like COCOeval)
    # We'll parse the printed results or access internal metrics
    results = {}

    # Get overall metrics
    if hasattr(lvis_eval, 'results'):
        overall_results = lvis_eval.results
        results['AP'] = overall_results.get('AP', 0) * 100
        results['AP50'] = overall_results.get('AP50', 0) * 100
        results['AP75'] = overall_results.get('AP75', 0) * 100
        results['APs'] = overall_results.get('APs', 0) * 100
        results['APm'] = overall_results.get('APm', 0) * 100
        results['APl'] = overall_results.get('APl', 0) * 100
        results['APr'] = overall_results.get('APr', 0) * 100
        results['APc'] = overall_results.get('APc', 0) * 100
        results['APf'] = overall_results.get('APf', 0) * 100
    else:
        # Fallback: try to access eval's internal state
        try:
            results['AP'] = lvis_eval.ap * 100 if hasattr(lvis_eval, 'ap') else 0
            results['AP50'] = lvis_eval.ap50 * 100 if hasattr(lvis_eval, 'ap50') else 0
            results['AP75'] = lvis_eval.ap75 * 100 if hasattr(lvis_eval, 'ap75') else 0
            results['APs'] = lvis_eval.aps * 100 if hasattr(lvis_eval, 'aps') else 0
            results['APm'] = lvis_eval.apm * 100 if hasattr(lvis_eval, 'apm') else 0
            results['APl'] = lvis_eval.apl * 100 if hasattr(lvis_eval, 'apl') else 0
            results['APr'] = lvis_eval.ap_rare * 100 if hasattr(lvis_eval, 'ap_rare') else 0
            results['APc'] = lvis_eval.ap_common * 100 if hasattr(lvis_eval, 'ap_common') else 0
            results['APf'] = lvis_eval.ap_freq * 100 if hasattr(lvis_eval, 'ap_freq') else 0
        except:
            # Last resort: set to 0 and warn
            print("Warning: Could not extract detailed metrics from LVISEval, using fallback values")
            results['AP'] = 0
            results['AP50'] = 0
            results['AP75'] = 0
            results['APs'] = 0
            results['APm'] = 0
            results['APl'] = 0
            results['APr'] = 0
            results['APc'] = 0
            results['APf'] = 0

    # Count categories with GT in this subset
    gt_cat_ids = set(ann['category_id'] for ann in subset_annotations)
    results['n_rare_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'r'])
    results['n_common_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'c'])
    results['n_freq_cats'] = len([c for c in categories if c['id'] in gt_cat_ids and c.get('frequency') == 'f'])

    # Cleanup
    os.remove(gt_file)

    return results


def main():
    args = parse_args()

    # Auto-detect paths based on --use_full_val flag
    if args.use_full_val:
        # Use full LVIS val set
        if args.lvis_ann is None:
            args.lvis_ann = '../LVIS/lvis_v1_val.json'
        if args.image_dir is None:
            args.image_dir = '../LVIS/val'
    else:
        # Use minival set (default, fair evaluation)
        if args.lvis_ann is None:
            args.lvis_ann = '../LVIS/minival/lvis_v1_minival.json'
        if args.image_dir is None:
            args.image_dir = '../LVIS/minival'

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create ablation-specific output directory
    if args.ablation:
        ablation_suffix = f"_{args.ablation}"
        if args.ablation == 'single_scale':
            ablation_suffix += f"_scale{args.scale_idx}"
        elif args.ablation == 'fewer_layers':
            ablation_suffix += f"_layers{args.num_layers}"
        elif args.ablation == 'prompt_style':
            ablation_suffix += f"_{args.style}"

        args.output_dir = args.output_dir + ablation_suffix

    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Archive previous run (unless resuming)
    if not args.resume:
        archive_previous_run(args.output_dir)

    # Setup logging to both console and file
    setup_logging(args.output_dir)

    config = Config()

    print("=" * 80)
    print("Grounding DINO Ablation Experiments")
    print("=" * 80)

    if args.ablation:
        print(f"Ablation: {args.ablation}")
        if args.ablation == 'single_scale':
            print(f"  Scale index: {args.scale_idx}")
        elif args.ablation == 'fewer_layers':
            print(f"  Number of layers: {args.num_layers}")
        elif args.ablation == 'prompt_style':
            print(f"  Prompt style: {args.style}")
    else:
        print("No ablation (baseline evaluation)")

    print("=" * 80)

    # Load LVIS annotations
    print(f"\n[1/6] Loading LVIS annotations from {args.lvis_ann}...")
    with open(args.lvis_ann, 'r') as f:
        lvis_data = json.load(f)

    images = {img['id']: img for img in lvis_data['images']}
    categories = lvis_data['categories']

    img_to_anns = defaultdict(list)
    for ann in lvis_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # Select images with annotations
    image_ids = [img_id for img_id in images.keys() if img_id in img_to_anns]

    # Apply range selection (start_idx/end_idx takes priority over num_images)
    if args.start_idx > 0 or args.end_idx is not None:
        # Explicit range overrides --num_images
        end_idx = args.end_idx if args.end_idx is not None else len(image_ids)
        image_ids = image_ids[args.start_idx:end_idx]
        start_idx = args.start_idx
    elif args.full:
        start_idx = 0
    elif args.num_images > 0:
        image_ids = image_ids[:args.num_images]
        start_idx = 0
    else:
        start_idx = 0

    print(f"  Total images in LVIS minival: {len(images)}")
    print(f"  Images with annotations: {len(img_to_anns)}")
    print(f"  Evaluating images [{start_idx}:{start_idx + len(image_ids)}] (subset: {len(image_ids)} images)")
    print(f"  Total categories: {len(categories)}")

    # Load tokenizer and build category batches
    print(f"\n[2/6] Building category batches (batch_size={args.batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    batch_info = build_category_batches_ablation(
        categories, tokenizer, args.batch_size,
        ablation_type=args.ablation, ablation_params=vars(args)
    )

    print(f"  Number of batches: {len(batch_info)}")
    print(f"  Max tokens per batch: {max(b['num_tokens'] for b in batch_info)}")
    # Log positive_map sum for reproducibility verification
    print(f"  batch_info[0]['positive_map'].sum() = {batch_info[0]['positive_map'].sum()}")

    # Load model
    print(f"\n[3/6] Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    # Apply ablation directly to model (in-memory only)
    if args.ablation:
        print(f"\n[3.5/6] Applying ablation: {args.ablation}")
        ablation_params = {}
        if args.ablation == 'single_scale':
            ablation_params['scale_idx'] = args.scale_idx
        elif args.ablation == 'fewer_layers':
            ablation_params['num_layers'] = args.num_layers
        elif args.ablation == 'prompt_style':
            ablation_params['style'] = args.style

        apply_ablation_direct(model, args.ablation, ablation_params)
        print("  Ablation applied successfully")

    # Check for existing checkpoint
    print(f"\n[4/6] Checking for checkpoints in {args.output_dir}...")
    resume_start_idx = 0
    if args.resume or os.path.exists(os.path.join(args.output_dir, 'progress.json')):
        resume_start_idx = load_checkpoint(args.output_dir)
        print(f"  Resuming from image index {resume_start_idx}")
    else:
        print("  No checkpoint found, starting fresh")

    # Build ablation description for output
    ablation_desc = ""
    if args.ablation:
        ablation_desc = f" (ABLATION: {args.ablation}"
        if args.ablation == 'single_scale':
            ablation_desc += f", scale_idx={args.scale_idx}"
        elif args.ablation == 'fewer_layers':
            ablation_desc += f", num_layers={args.num_layers}"
        elif args.ablation == 'prompt_style':
            ablation_desc += f", style={args.style}"
        ablation_desc += ")"

    # Run inference
    if args.ultra_optimized:
        inference_func = run_inference_batched_ultra_optimized
        print(f"\n[5/6] Running ULTRA-OPTIMIZED inference{ablation_desc} on {len(image_ids)} images...")
        print(f"  (GPU-CPU syncs reduced from 26 to 2-3 per image)")
        print(f"  (All operations kept on GPU until final sync)")
        print(f"  (Expected 15-25% speedup)")
    else:
        inference_func = run_inference_batched_optimized
        print(f"\n[5/6] Running OPTIMIZED inference{ablation_desc} on {len(image_ids)} images...")
        print(f"  (Vision features cached per image, text encoded fresh per batch)")
    print(f"  (Each image now requires only {len(batch_info)} encoder+decoder passes)")
    print(f"  (Checkpointing every {args.checkpoint_interval} images)")

    start_time = time.time()
    processed_count = 0

    # Resume from checkpoint if available
    remaining_image_ids = image_ids[resume_start_idx:]
    print(f"  Processing {len(remaining_image_ids)} images (starting from index {resume_start_idx})")

    # Pre-filter to only include images that actually exist
    print(f"  Pre-filtering images to check which files exist...")
    existing_image_ids = []
    for img_id in tqdm(remaining_image_ids, desc="Checking images"):
        img_info = images[img_id]

        # Get filename
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"

        img_path = os.path.join(args.image_dir, file_name)

        if os.path.exists(img_path):
            existing_image_ids.append(img_id)

    print(f"  Found {len(existing_image_ids)} existing images out of {len(remaining_image_ids)}")
    remaining_image_ids = existing_image_ids

    if len(remaining_image_ids) == 0:
        print("  ERROR: No valid images found in specified range!")
        print(f"  Check that image directory '{args.image_dir}' contains the correct lvis/minival images")
        sys.exit(1)

    for img_idx, img_id in enumerate(tqdm(
        remaining_image_ids,
        initial=resume_start_idx,
        total=len(image_ids),
        desc=f"GPU {args.gpu}",
        position=0,
        leave=True
    ), start=resume_start_idx):
        img_info = images[img_id]

        # Get filename
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = img_info['coco_url'].split('/')[-1]
        else:
            file_name = f"{img_id:012d}.jpg"

        img_path = os.path.join(args.image_dir, file_name)

        if not os.path.exists(img_path):
            continue

        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        img_tensor, _ = preprocess_image(image, config)

        # Run batched inference with selected optimization level
        img_preds = inference_func(
            model, img_tensor, batch_info, None,  # Text encoded fresh per batch
            (orig_w, orig_h), args.box_threshold
        )

        # DEBUG: Print prediction count for first image
        if img_idx == 0:
            print(f"  DEBUG: Image {img_id} produced {len(img_preds)} predictions")
            if len(img_preds) > 0:
                print(f"  DEBUG: Sample prediction: {img_preds[0]}")

        # Add image_id to predictions
        for pred in img_preds:
            pred['image_id'] = int(img_id)

        # Write predictions to JSONL file
        write_predictions_to_jsonl(args.output_dir, img_preds)
        processed_count += 1

        # Save checkpoint every N images
        if processed_count % args.checkpoint_interval == 0:
            save_checkpoint(args.output_dir, img_idx)
            print(f"  Checkpoint saved at image {img_idx + 1}")

        # Real-time AP evaluation (optional, expensive)
        if args.eval_interval > 0 and processed_count % args.eval_interval == 0 and processed_count > 0:
            try:
                # Load current predictions
                current_predictions = load_predictions_from_jsonl(args.output_dir)
                # Evaluate on current subset of images processed so far
                current_image_ids = set(pred['image_id'] for pred in current_predictions)

                # Run evaluation on current predictions
                partial_results = evaluate_with_lvis(
                    current_predictions, lvis_data, current_image_ids, categories, args.output_dir
                )

                # Build ablation description for logging
                ablation_log_desc = ""
                if args.ablation:
                    ablation_log_desc = f" [{args.ablation}"
                    if args.ablation == 'single_scale':
                        ablation_log_desc += f", scale_idx={args.scale_idx}"
                    elif args.ablation == 'fewer_layers':
                        ablation_log_desc += f", num_layers={args.num_layers}"
                    elif args.ablation == 'prompt_style':
                        ablation_log_desc += f", style={args.style}"
                    ablation_log_desc += "]"

                # Log partial results
                logging.info(f"[Partial{ablation_log_desc} @ {processed_count}/{len(remaining_image_ids)} images] "
                           f"AP: {partial_results['AP']:.2f}%, "
                           f"AP50: {partial_results['AP50']:.2f}%, "
                           f"AP75: {partial_results['AP75']:.2f}%")

                # Save partial results
                partial_results_file = os.path.join(args.output_dir, f'results_partial_{processed_count:06d}.json')
                with open(partial_results_file, 'w') as f:
                    json.dump({
                        'images_processed': processed_count,
                        'total_images': len(remaining_image_ids),
                        'results': partial_results,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                print(f"  [Partial AP: {partial_results['AP']:.2f}% at {processed_count} images]")

            except Exception as e:
                logging.warning(f"Failed to compute partial evaluation: {e}")

        # Cleanup tensor
        del img_tensor

    # Final checkpoint
    if processed_count > 0:
        save_checkpoint(args.output_dir, len(image_ids) - 1)
        print(f"  Final checkpoint saved")

    elapsed = time.time() - start_time
    print(f"\n  Inference completed in {elapsed/60:.1f} minutes")
    if processed_count > 0:
        print(f"  Average time per image: {elapsed/processed_count:.1f} seconds")
    else:
        print("  Warning: No images were processed")

    # Load all predictions from JSONL and convert to JSON array
    print(f"\n  Loading predictions from JSONL file...")
    all_predictions = load_predictions_from_jsonl(args.output_dir)
    print(f"  Total predictions: {len(all_predictions)}")

    # Evaluate
    print(f"\n[6/6] Running LVIS evaluation...")
    results = evaluate_with_lvis(
        all_predictions, lvis_data, set(image_ids), categories, args.output_dir
    )

    # Print results
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 80)
    if args.ablation:
        print(f"Ablation: {args.ablation}")
        if args.ablation == 'single_scale':
            print(f"Scale: {args.scale_idx} (0=high-res, 3=low-res)")
        elif args.ablation == 'fewer_layers':
            print(f"Layers: {args.num_layers}/6")
        elif args.ablation == 'prompt_style':
            print(f"Style: {args.style}")
    print("-" * 80)
    print(f"  Images evaluated: {len(image_ids)}")
    print(f"  Categories: {len(categories)} (r:{results['n_rare_cats']}, c:{results['n_common_cats']}, f:{results['n_freq_cats']} with GT)")
    print("-" * 80)
    print(f"  AP   (IoU=0.50:0.95): {results['AP']:.1f}%")
    print(f"  AP50 (IoU=0.50):      {results['AP50']:.1f}%")
    print(f"  AP75 (IoU=0.75):      {results['AP75']:.1f}%")
    print("-" * 80)
    print(f"  APr  (rare):          {results['APr']:.1f}%")
    print(f"  APc  (common):        {results['APc']:.1f}%")
    print(f"  APf  (frequent):      {results['APf']:.1f}%")
    print("-" * 80)
    print(f"  APs  (small):         {results['APs']:.1f}%")
    print(f"  APm  (medium):        {results['APm']:.1f}%")
    print(f"  APl  (large):         {results['APl']:.1f}%")
    print("=" * 80)

    # Save results
    results['num_images'] = len(image_ids)
    results['inference_time_seconds'] = elapsed
    results['ultra_optimized'] = args.ultra_optimized
    results['timestamp'] = datetime.now().isoformat()

    # Add ablation info
    if args.ablation:
        results['ablation'] = args.ablation
        if args.ablation == 'single_scale':
            results['scale_idx'] = args.scale_idx
        elif args.ablation == 'fewer_layers':
            results['num_layers'] = args.num_layers
        elif args.ablation == 'prompt_style':
            results['prompt_style'] = args.style

    results_file = os.path.join(args.output_dir, 'lvis_zeroshot_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {os.path.join(args.output_dir, 'lvis_predictions.json')}")


if __name__ == '__main__':
    main()
