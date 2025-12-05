# Module Interfaces (All Members)
"""
This file defines the interfaces that need to be implemented by different modules
to ensure compatibility between components implemented by different team members.
"""

import jittor as jt
from typing import Dict, List, Optional, Tuple, Union

# Model input/output interfaces
class ModelInput:
    """Input interface for the complete model"""
    images: jt.Var  # (B, 3, H, W) - Image tensor
    text_features: jt.Var  # (B, L, D) - Text features from BERT
    text_token_mask: jt.Var  # (B, L) - Text token mask
    position_ids: jt.Var  # (B, L) - Position IDs for text tokens
    text_self_attention_masks: jt.Var  # (B, L, L) - Text self-attention masks
    cate_to_token_mask_list: List[jt.Var]  # List of category-to-token masks


class ModelOutput:
    """Output interface for the complete model"""
    pred_logits: jt.Var  # (B, N, num_classes) - Classification logits
    pred_boxes: jt.Var  # (B, N, 4) - Bounding box predictions (cx, cy, w, h)

# Data interfaces
class BatchData:
    """Interface for data returned by dataloader"""
    images: jt.Var  # (B, 3, H, W) - Image tensor
    texts: List[str]  # List of text strings
    annotations: Dict  # {'boxes': ..., 'labels': ..., 'categories': ...}

# Text encoder interfaces
class TextFeatures:
    """Interface for text encoder output"""
    features: jt.Var  # (B, L, D) - Text features
    attention_mask: jt.Var  # (B, L) - Attention mask
    position_ids: jt.Var  # (B, L) - Position IDs
    self_attention_masks: jt.Var  # (B, L, L) - Self-attention masks
    cate_to_token_mask_list: List[jt.Var]  # Category to token masks

# Feature fusion interfaces
class FusionInput:
    """Input interface for feature fusion module"""
    visual_features: jt.Var  # (B, H, W, D) - Visual features
    text_features: jt.Var  # (B, L, D) - Text features

class FusionOutput:
    """Output interface for feature fusion module"""
    fused_features: jt.Var  # (B, H, W, D) - Fused features

# Query generation interfaces
class QueryInput:
    """Input interface for language-guided query generation"""
    text_features: jt.Var  # (B, L, D) - Text features
    num_queries: int  # Number of queries to generate

class QueryOutput:
    """Output interface for query generation"""
    query_embeds: jt.Var  # (num_queries, D) - Query embeddings
    query_positions: jt.Var  # (num_queries, 2) - Query positions

# Training interfaces
class TrainingState:
    """Interface for training state"""
    epoch: int  # Current epoch
    step: int  # Current step
    loss: float  # Current loss
    metrics: Dict  # Dictionary of metrics

class TrainingConfig:
    """Interface for training configuration"""
    learning_rate: float  # Learning rate
    batch_size: int  # Batch size
    epochs: int  # Number of epochs
    save_freq: int  # Save frequency
    eval_freq: int  # Evaluation frequency
    checkpoint_dir: str  # Checkpoint directory
    device: str  # Device to use ('cuda' or 'cpu')

# Evaluation interfaces
class EvalResult:
    """Interface for evaluation results"""
    AP: float  # Average Precision
    AP50: float  # AP at IoU=0.5
    AP75: float  # AP at IoU=0.75
    APs: float  # AP for small objects
    APm: float  # AP for medium objects
    APl: float  # AP for large objects
    AR1: float  # AR with 1 detection per image
    AR10: float  # AR with 10 detections per image
    AR100: float  # AR with 100 detections per image
    ARs: float  # AR for small objects
    ARm: float  # AR for medium objects
    ARl: float  # AR for large objects