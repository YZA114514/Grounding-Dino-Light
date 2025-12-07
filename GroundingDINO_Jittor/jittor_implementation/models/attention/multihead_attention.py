# Multi-Head Attention Implementation for Jittor
# Jittor 没有内置 MultiheadAttention，这里提供兼容实现

import math
import jittor as jt
from jittor import nn
from typing import Optional, Tuple


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention 模块 (Jittor 实现)
    
    与 PyTorch 的 nn.MultiheadAttention 接口兼容
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # Xavier uniform initialization
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
    
    def execute(
        self,
        query: jt.Var,
        key: jt.Var,
        value: jt.Var,
        key_padding_mask: Optional[jt.Var] = None,
        need_weights: bool = True,
        attn_mask: Optional[jt.Var] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[jt.Var, Optional[jt.Var]]:
        """
        前向传播
        
        Args:
            query: (L, N, E) 或 (N, L, E) if batch_first
            key: (S, N, E) 或 (N, S, E) if batch_first
            value: (S, N, E) 或 (N, S, E) if batch_first
            key_padding_mask: (N, S) - True 表示被屏蔽的位置
            need_weights: 是否返回注意力权重
            attn_mask: (L, S) 或 (N*num_heads, L, S) - 注意力掩码
            
        Returns:
            output: (L, N, E) 或 (N, L, E) if batch_first
            attn_weights: (N, L, S) 如果 need_weights=True
        """
        if self.batch_first:
            # (N, L, E) -> (L, N, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头格式: (L, N, E) -> (L, N, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        q = q.reshape(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # 计算注意力分数: (N, num_heads, L, head_dim) @ (N, num_heads, head_dim, S) = (N, num_heads, L, S)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jt.matmul(q, k.transpose(-2, -1)) * scale
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + attn_mask
        
        # 应用 key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (N, S) -> (N, 1, 1, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = jt.where(
                key_padding_mask,
                jt.full_like(attn_weights, float('-inf')),
                attn_weights
            )
        
        # Softmax
        attn_weights = nn.softmax(attn_weights, dim=-1)
        
        # Dropout
        if self.dropout_layer is not None and self.is_training:
            attn_weights = self.dropout_layer(attn_weights)
        
        # 应用注意力权重: (N, num_heads, L, S) @ (N, num_heads, S, head_dim) = (N, num_heads, L, head_dim)
        output = jt.matmul(attn_weights, v)
        
        # 重塑回原始格式: (N, num_heads, L, head_dim) -> (L, N, num_heads, head_dim) -> (L, N, E)
        output = output.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        if self.batch_first:
            output = output.transpose(0, 1)
        
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # (N, L, S)
            return output, attn_weights
        else:
            return output, None


# 为了兼容性，将 MultiheadAttention 添加到 nn 模块
if not hasattr(nn, 'MultiheadAttention'):
    nn.MultiheadAttention = MultiheadAttention

