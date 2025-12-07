# Attention modules for Grounding DINO Jittor Implementation

from .ms_deform_attn import (
    MSDeformAttn,
    MultiScaleDeformableAttention,
    multi_scale_deformable_attn_jittor,
)

# 导入 MultiheadAttention 并注入到 jittor.nn
from .multihead_attention import MultiheadAttention

# 将 MultiheadAttention 添加到 nn 模块以保持兼容性
from jittor import nn
if not hasattr(nn, 'MultiheadAttention'):
    nn.MultiheadAttention = MultiheadAttention

__all__ = [
    "MSDeformAttn",
    "MultiScaleDeformableAttention",
    "multi_scale_deformable_attn_jittor",
    "MultiheadAttention",
]


