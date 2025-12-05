# Language Guided Query (Member C)
import jittor as jt
from jittor import nn
from typing import Dict, List, Optional, Tuple, Union


class LanguageGuidedQuery(nn.Module):
    """Language-guided query generation for GroundingDINO"""
    
    def __init__(self, hidden_dim, num_queries=900, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        
        # Query embedding layer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Linear projection for text features
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Query generation layers
        self.query_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="relu"
            ) for _ in range(num_layers)
        ])
        
        # Position encoding for queries
        self.pos_embed = nn.Parameter(jt.randn(1, num_queries, hidden_dim))
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        nn.init.normal_(self.query_embed.weight)
        nn.init.normal_(self.pos_embed)
    
    def execute(self, text_features, text_token_mask=None):
        """
        Generate queries from text features
        
        Args:
            text_features: (B, L, D) text features from BERT
            text_token_mask: (B, L) text token mask
            
        Returns:
            Generated queries (num_queries, D) or (B, num_queries, D)
        """
        batch_size = text_features.shape[0]
        
        # Project text features
        text_proj = self.text_proj(text_features)  # (B, L, D)
        
        # Get initial query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_queries, D)
        
        # Add position encoding
        query_embed = query_embed + self.pos_embed
        
        # Normalize
        query_embed = self.norm(query_embed)
        
        # Apply query generation layers
        # Reshape for transformer: (num_queries, B, D)
        query_embed_t = query_embed.transpose(0, 1)
        
        # Process each layer
        for layer in self.query_layers:
            # Text as key/value for cross-attention
            text_proj_t = text_proj.transpose(0, 1)  # (L, B, D)
            
            # Cross attention: query_embed as query, text_proj as key/value
            query_with_text = layer.self_attn(
                query_embed_t,  # (num_queries, B, D)
                text_proj_t,    # (L, B, D)
                text_proj_t,    # (L, B, D)
                key_padding_mask=~text_token_mask if text_token_mask is not None else None
            )[0]
            
            # Residual connection
            query_embed_t = query_embed_t + self.dropout(query_with_text)
            
            # Self attention
            query_self = layer.self_attn(
                query_embed_t,  # (num_queries, B, D)
                query_embed_t,  # (num_queries, B, D)
                query_embed_t,  # (num_queries, B, D)
                key_padding_mask=None
            )[0]
            
            # Residual connection
            query_embed_t = query_embed_t + self.dropout(query_self)
            
            # Feed forward
            query_ffn = layer.linear2(layer.dropout(layer.activation(layer.linear1(query_embed_t))))
            query_embed_t = query_embed_t + self.dropout(query_ffn)
            
            # Layer norm
            query_embed_t = layer.norm1(query_embed_t)
            query_embed_t = layer.norm2(query_embed_t)
        
        # Reshape back: (B, num_queries, D)
        query_embed = query_embed_t.transpose(0, 1)
        
        # Final normalization
        query_embed = self.norm(query_embed)
        
        return query_embed


class DynamicQueryGenerator(nn.Module):
    """Dynamic query generator that adapts based on text content"""
    
    def __init__(self, hidden_dim, num_queries=900):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Text feature aggregation
        self.text_aggregator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Query content generator
        self.query_content = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query position generator
        self.query_position = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # (x, y) position
        )
        
        # Base query embeddings
        self.base_query = nn.Parameter(jt.randn(num_queries, hidden_dim))
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.normal_(self.base_query)
    
    def execute(self, text_features, text_token_mask=None):
        """
        Generate dynamic queries based on text content
        
        Args:
            text_features: (B, L, D) text features from BERT
            text_token_mask: (B, L) text token mask
            
        Returns:
            Generated queries (B, num_queries, D) and positions (B, num_queries, 2)
        """
        batch_size = text_features.shape[0]
        
        # Aggregate text features to get text context
        text_features_t = text_features.transpose(0, 1)  # (L, B, D)
        
        # Self attention to aggregate text features
        aggregated_text, _ = self.text_aggregator(
            text_features_t,
            text_features_t,
            text_features_t,
            key_padding_mask=~text_token_mask if text_token_mask is not None else None
        )
        aggregated_text = aggregated_text.transpose(0, 1)  # (B, L, D)
        
        # Global text representation (average pooling)
        text_mask_float = text_token_mask.float().unsqueeze(-1) if text_token_mask is not None else 1.0
        text_sum = jt.sum(aggregated_text * text_mask_float, dim=1)  # (B, D)
        text_count = jt.sum(text_mask_float, dim=1) + 1e-6  # (B, 1)
        text_global = text_sum / text_count  # (B, D)
        
        # Expand base queries for batch
        base_queries = self.base_query.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, num_queries, D)
        
        # Generate query content based on text
        query_content_input = text_global.unsqueeze(1).repeat(1, self.num_queries, 1)  # (B, num_queries, D)
        query_content = self.query_content(query_content_input)  # (B, num_queries, D)
        
        # Generate query positions based on text
        query_pos = self.query_position(query_content_input)  # (B, num_queries, 2)
        
        # Combine base queries with content
        dynamic_queries = base_queries + query_content
        
        # Normalize
        dynamic_queries = self.norm(dynamic_queries)
        
        return dynamic_queries, query_pos


class AdaptiveQueryGenerator(nn.Module):
    """Adaptive query generator that adjusts number of queries based on text complexity"""
    
    def __init__(self, hidden_dim, max_queries=900):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_queries = max_queries
        
        # Text complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Query content generator
        self.query_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position generator
        self.position_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Base query embeddings
        self.base_queries = nn.Parameter(jt.randn(max_queries, hidden_dim))
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.normal_(self.base_queries)
    
    def execute(self, text_features, text_token_mask=None):
        """
        Generate adaptive number of queries based on text complexity
        
        Args:
            text_features: (B, L, D) text features from BERT
            text_token_mask: (B, L) text token mask
            
        Returns:
            Generated queries (B, num_queries, D) and positions (B, num_queries, 2)
        """
        batch_size = text_features.shape[0]
        
        # Get global text representation
        text_mask_float = text_token_mask.float().unsqueeze(-1) if text_token_mask is not None else 1.0
        text_sum = jt.sum(text_features * text_mask_float, dim=1)  # (B, D)
        text_count = jt.sum(text_mask_float, dim=1) + 1e-6  # (B, 1)
        text_global = text_sum / text_count  # (B, D)
        
        # Estimate text complexity
        complexity = self.complexity_estimator(text_global)  # (B, 1)
        num_queries = jt.floor(complexity * self.max_queries).int()  # (B, 1)
        
        # Generate queries for each sample in the batch
        all_queries = []
        all_positions = []
        
        for i in range(batch_size):
            n_q = min(num_queries[i].item(), self.max_queries)
            
            # Get base queries for this sample
            base_q = self.base_queries[:n_q].unsqueeze(0)  # (1, n_q, D)
            
            # Repeat text global for each query
            text_rep = text_global[i:i+1].unsqueeze(1).repeat(1, n_q, 1)  # (1, n_q, D)
            
            # Combine base queries with text
            combined = jt.concat([base_q, text_rep], dim=-1)  # (1, n_q, 2D)
            
            # Generate query content
            query_content = self.query_generator(combined)  # (1, n_q, D)
            
            # Generate positions
            position = self.position_generator(combined)  # (1, n_q, 2)
            
            # Normalize
            query_content = self.norm(query_content)
            
            all_queries.append(query_content)
            all_positions.append(position)
        
        # Pad queries and positions to same size
        max_n = max(q.shape[1] for q in all_queries)
        
        padded_queries = []
        padded_positions = []
        
        for q, p in zip(all_queries, all_positions):
            n_q = q.shape[1]
            if n_q < max_n:
                # Pad with zeros
                q_pad = jt.zeros(1, max_n - n_q, self.hidden_dim)
                p_pad = jt.zeros(1, max_n - n_q, 2)
                
                q = jt.concat([q, q_pad], dim=1)
                p = jt.concat([p, p_pad], dim=1)
            
            padded_queries.append(q.squeeze(0))  # (max_n, D)
            padded_positions.append(p.squeeze(0))  # (max_n, 2)
        
        # Stack across batch
        queries = jt.stack(padded_queries, dim=0)  # (B, max_n, D)
        positions = jt.stack(padded_positions, dim=0)  # (B, max_n, 2)
        
        return queries, positions


class TextConditionalQueryGenerator(nn.Module):
    """Text-conditional query generator that generates different queries based on text content"""
    
    def __init__(self, hidden_dim, num_queries=900, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        
        # Text embedding projection
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Query generation layers
        self.query_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(jt.randn(1, num_queries, hidden_dim))
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.normal_(self.query_tokens)
    
    def execute(self, text_features, text_token_mask=None):
        """
        Generate text-conditional queries
        
        Args:
            text_features: (B, L, D) text features from BERT
            text_token_mask: (B, L) text token mask
            
        Returns:
            Generated queries (B, num_queries, D)
        """
        batch_size = text_features.shape[0]
        
        # Project text features
        text_proj = self.text_proj(text_features)  # (B, L, D)
        
        # Get global text representation
        text_mask_float = text_token_mask.float().unsqueeze(-1) if text_token_mask is not None else 1.0
        text_sum = jt.sum(text_proj * text_mask_float, dim=1)  # (B, D)
        text_count = jt.sum(text_mask_float, dim=1) + 1e-6  # (B, 1)
        text_global = text_sum / text_count  # (B, D)
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.repeat(batch_size, 1, 1)  # (B, num_queries, D)
        
        # Add position encoding
        query_tokens = self.pos_encoding(query_tokens)
        
        # Generate queries based on text
        for layer in self.query_layers:
            # Combine query tokens with text
            text_expanded = text_global.unsqueeze(1).repeat(1, self.num_queries, 1)  # (B, num_queries, D)
            combined = query_tokens + text_expanded
            
            # Apply layer
            query_tokens = layer(combined)
        
        # Normalize
        query_tokens = self.norm(query_tokens)
        
        return query_tokens


class PositionalEncoding(nn.Module):
    """Positional encoding for queries"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = jt.zeros(max_len, d_model)
        position = jt.arange(0, max_len, dtype=jt.float).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * (-jt.log(jt.array(10000.0)) / d_model))
        
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer to avoid being considered as a parameter
        self.register_buffer('pe', pe)
    
    def execute(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if len(x.shape) == 3 and x.shape[0] != x.shape[1]:
            # Assume shape is [batch_size, seq_len, embedding_dim]
            batch_size, seq_len, _ = x.shape
            return x + self.pe[:seq_len, :].transpose(0, 1)
        else:
            # Assume shape is [seq_len, batch_size, embedding_dim]
            seq_len = x.shape[0]
            return x + self.pe[:seq_len, :]