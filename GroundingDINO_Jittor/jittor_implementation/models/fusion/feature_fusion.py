# Feature Fusion (Member C)
import jittor as jt
from jittor import nn
from typing import Dict, List, Optional, Tuple


class FeatureFusion(nn.Module):
    """Visual-Language feature fusion module for GroundingDINO"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Self-attention for text features
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Self-attention for visual features
        self.vis_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)  # For cross-attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # For FFN
        self.norm3 = nn.LayerNorm(hidden_dim)  # For self-attention
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def execute(
        self, 
        visual_features: jt.Var, 
        text_features: jt.Var,
        text_self_attention_masks: Optional[jt.Var] = None,
        text_token_mask: Optional[jt.Var] = None
    ):
        """
        Fuse visual and text features
        
        Args:
            visual_features: (B, H, W, D) or (B, N, D) visual features
            text_features: (B, L, D) text features
            text_self_attention_masks: (B, L, L) text self-attention masks
            text_token_mask: (B, L) text token mask
            
        Returns:
            Fused features with same shape as visual_features
        """
        batch_size = visual_features.shape[0]
        
        # Reshape visual features if needed
        if len(visual_features.shape) == 4:  # (B, H, W, D)
            vis_shape = visual_features.shape
            h, w, d = vis_shape[1], vis_shape[2], vis_shape[3]
            visual_features_flat = visual_features.reshape(batch_size, h * w, d)
        else:  # (B, N, D)
            visual_features_flat = visual_features
            h = w = None
        
        # Normalize features
        visual_features_norm = self.norm3(visual_features_flat)
        text_features_norm = text_features
        
        # Apply self-attention to text features with masks
        if text_self_attention_masks is not None:
            text_features_sa = self.text_self_attention(
                text_features_norm.transpose(0, 1),  # (L, B, D)
                text_features_norm.transpose(0, 1),  # (L, B, D)
                text_features_norm.transpose(0, 1),  # (L, B, D)
                key_padding_mask=~text_token_mask if text_token_mask is not None else None  # (B, L)
            )[0].transpose(0, 1)  # (B, L, D)
            text_features_norm = text_features_norm + self.dropout(text_features_sa)
            text_features_norm = self.norm1(text_features_norm)
        
        # Cross-modal attention: visual features query, text features key/value
        visual_query = visual_features_norm.transpose(0, 1)  # (N, B, D)
        text_key_value = text_features_norm.transpose(0, 1)  # (L, B, D)
        
        cross_attn_out = self.cross_attention(
            visual_query,  # Query from visual
            text_key_value,  # Key from text
            text_key_value,  # Value from text
            key_padding_mask=~text_token_mask if text_token_mask is not None else None  # (B, L)
        )[0].transpose(0, 1)  # (B, N, D)
        
        # Residual connection and normalization
        visual_features_fused = visual_features_norm + self.dropout(cross_attn_out)
        visual_features_fused = self.norm1(visual_features_fused)
        
        # Feed-forward network
        ffn_out = self.ffn(visual_features_fused)
        
        # Residual connection and normalization
        visual_features_fused = visual_features_fused + self.dropout(ffn_out)
        visual_features_fused = self.norm2(visual_features_fused)
        
        # Reshape back to original format if needed
        if h is not None and w is not None:
            visual_features_fused = visual_features_fused.reshape(batch_size, h, w, self.hidden_dim)
        
        return visual_features_fused


class ContrastiveEmbed(nn.Module):
    """Contrastive embedding for classification"""
    
    def __init__(self, logit_scale_init=-2.5):
        super().__init__()
        self.logit_scale = jt.array([logit_scale_init])
    
    def execute(self, visual_features, text_features, text_token_mask=None):
        """
        Compute contrastive similarity between visual and text features
        
        Args:
            visual_features: (B, N, D) visual features
            text_features: (B, L, D) text features
            text_token_mask: (B, L) text token mask
            
        Returns:
            Similarity matrix (B, N, L)
        """
        # Normalize features
        visual_features_norm = nn.normalize(visual_features, dim=-1)
        text_features_norm = nn.normalize(text_features, dim=-1)
        
        # Compute similarity
        similarity = jt.matmul(visual_features_norm, text_features_norm.transpose(0, 2, 1))
        
        # Apply temperature scaling
        logit_scale = jt.exp(self.logit_scale)
        similarity = similarity * logit_scale
        
        # Apply token mask if provided
        if text_token_mask is not None:
            # Set similarity for masked tokens to very negative value
            mask_expanded = text_token_mask.unsqueeze(1)  # (B, 1, L)
            similarity = similarity * mask_expanded + -1e6 * ~mask_expanded
        
        return similarity


class LanguageGuidedFusion(nn.Module):
    """Language-guided feature fusion"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Language-guided attention
        self.lang_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def execute(
        self, 
        visual_features: jt.Var, 
        text_features: jt.Var,
        text_token_mask: Optional[jt.Var] = None
    ):
        """
        Language-guided feature fusion
        
        Args:
            visual_features: (B, N, D) visual features
            text_features: (B, L, D) text features
            text_token_mask: (B, L) text token mask
            
        Returns:
            Enhanced visual features (B, N, D)
        """
        batch_size, num_vis_tokens, _ = visual_features.shape
        _, num_text_tokens, _ = text_features.shape
        
        # Visual features as query
        visual_query = visual_features.transpose(0, 1)  # (N, B, D)
        
        # Text features as key and value
        text_key_value = text_features.transpose(0, 1)  # (L, B, D)
        
        # Language-guided attention
        attn_out = self.lang_attention(
            visual_query,  # Query from visual
            text_key_value,  # Key from text
            text_key_value,  # Value from text
            key_padding_mask=~text_token_mask if text_token_mask is not None else None  # (B, L)
        )[0].transpose(0, 1)  # (B, N, D)
        
        # Residual connection
        visual_features_enhanced = visual_features + self.dropout(attn_out)
        
        # Normalization
        visual_features_enhanced = self.norm(visual_features_enhanced)
        
        # Feature enhancement
        enhanced_features = self.feature_enhance(visual_features_enhanced)
        
        # Residual connection
        visual_features_final = visual_features_enhanced + enhanced_features
        
        return visual_features_final


class DynamicFusion(nn.Module):
    """Dynamic fusion module for multi-modal features"""
    
    def __init__(self, hidden_dim, fusion_type="concat"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # Reduce concatenated dimension back to hidden_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif fusion_type == "add":
            # Simple addition, no extra parameters needed
            self.fusion_layer = None
        elif fusion_type == "gate":
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def execute(self, visual_features, text_features):
        """
        Dynamically fuse visual and text features
        
        Args:
            visual_features: (B, N, D) visual features
            text_features: (B, N, D) text features (already projected to same dimension)
            
        Returns:
            Fused features (B, N, D)
        """
        if self.fusion_type == "concat":
            # Concatenate features
            concat_features = jt.concat([visual_features, text_features], dim=-1)
            # Reduce dimension
            fused_features = self.fusion_layer(concat_features)
        
        elif self.fusion_type == "add":
            # Simple addition
            fused_features = visual_features + text_features
        
        elif self.fusion_type == "gate":
            # Compute gate weights
            gate_input = jt.concat([visual_features, text_features], dim=-1)
            gate_weights = self.gate(gate_input)
            # Apply gate
            fused_features = gate_weights * visual_features + (1 - gate_weights) * text_features
            # Additional transformation
            fused_features = self.fusion_layer(fused_features)
        
        return fused_features