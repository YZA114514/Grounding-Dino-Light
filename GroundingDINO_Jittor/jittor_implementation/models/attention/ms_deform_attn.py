# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Multi-Scale Deformable Attention (Member A)
# ------------------------------------------------------------------------
"""
多尺度可变形注意力模块

核心思想：
- 传统注意力：O(N²) 复杂度，对所有位置计算注意力
- 可变形注意力：O(N×K) 复杂度，只对K个采样点计算注意力

工作原理：
1. 对每个query，学习K个采样偏移量
2. 根据参考点 + 偏移量确定采样位置
3. 在采样位置进行双线性插值获取特征值
4. 加权聚合得到输出

参考：
- Deformable DETR: https://arxiv.org/abs/2010.04159
- 官方实现: https://github.com/fundamentalvision/Deformable-DETR
"""

import math
import warnings
from typing import Optional

import jittor as jt
from jittor import nn
import numpy as np


def _is_power_of_2(n):
    """检查是否是2的幂"""
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


def multi_scale_deformable_attn_jittor(
    value: jt.Var,
    value_spatial_shapes: jt.Var,
    sampling_locations: jt.Var,
    attention_weights: jt.Var,
) -> jt.Var:
    """
    多尺度可变形注意力的纯 Jittor 实现
    
    Args:
        value: [bs, num_value, num_heads, embed_dims]
        value_spatial_shapes: [num_levels, 2] 每个层级的 (H, W)
        sampling_locations: [bs, num_query, num_heads, num_levels, num_points, 2]
        attention_weights: [bs, num_query, num_heads, num_levels, num_points]
        
    Returns:
        output: [bs, num_query, num_heads * embed_dims]
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    
    # 将 value 按层级分割
    # value_spatial_shapes: [[H1, W1], [H2, W2], ...]
    split_sizes = [int(H * W) for H, W in value_spatial_shapes.numpy()]
    value_list = jt.split(value, split_sizes, dim=1)
    
    # 将采样位置从 [0, 1] 转换到 [-1, 1] (grid_sample 的格式)
    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes.numpy()):
        H_, W_ = int(H_), int(W_)
        
        # value_l_: [bs, H_*W_, num_heads, embed_dims]
        # -> [bs, H_*W_, num_heads*embed_dims]
        # -> [bs, num_heads*embed_dims, H_*W_]
        # -> [bs*num_heads, embed_dims, H_, W_]
        value_l_ = value_list[level].flatten(2).transpose(1, 2)
        value_l_ = value_l_.reshape(bs * num_heads, embed_dims, H_, W_)
        
        # sampling_grid_l_: [bs, num_queries, num_heads, num_points, 2]
        # -> [bs, num_heads, num_queries, num_points, 2]
        # -> [bs*num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2)
        sampling_grid_l_ = sampling_grid_l_.flatten(0, 1)
        
        # 使用 grid_sample 进行双线性插值
        # value_l_: [bs*num_heads, embed_dims, H_, W_]
        # sampling_grid_l_: [bs*num_heads, num_queries, num_points, 2]
        # output: [bs*num_heads, embed_dims, num_queries, num_points]
        # 确保数据类型一致 (Jittor grid_sample 要求)
        sampling_grid_l_ = sampling_grid_l_.float()
        value_l_ = value_l_.float()
        sampling_value_l_ = nn.grid_sample(
            value_l_, 
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    
    # attention_weights: [bs, num_queries, num_heads, num_levels, num_points]
    # -> [bs, num_heads, num_queries, num_levels, num_points]
    # -> [bs*num_heads, 1, num_queries, num_levels*num_points]
    attention_weights = attention_weights.transpose(1, 2)
    attention_weights = attention_weights.reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    
    # 堆叠所有层级的采样值并加权求和
    # [bs*num_heads, embed_dims, num_queries, num_points] * num_levels
    # -> [bs*num_heads, embed_dims, num_queries, num_levels, num_points]
    # -> [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
    stacked_values = jt.stack(sampling_value_list, dim=-2).flatten(-2)
    
    # 加权求和
    # [bs*num_heads, embed_dims, num_queries, num_levels*num_points] * [bs*num_heads, 1, num_queries, num_levels*num_points]
    # -> [bs*num_heads, embed_dims, num_queries, num_levels*num_points]
    # -> [bs*num_heads, embed_dims, num_queries] (sum over last dim)
    output = (stacked_values * attention_weights).sum(-1)
    
    # [bs*num_heads, embed_dims, num_queries]
    # -> [bs, num_heads*embed_dims, num_queries]
    # -> [bs, num_queries, num_heads*embed_dims]
    output = output.view(bs, num_heads * embed_dims, num_queries)
    output = output.transpose(1, 2)
    
    return output


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力模块
    
    用于 Deformable DETR 和 Grounding DINO 的核心注意力机制。
    相比标准注意力，具有更高的效率和更好的多尺度特征处理能力。
    
    Args:
        embed_dim: 嵌入维度，默认 256
        num_heads: 注意力头数，默认 8
        num_levels: 特征图层级数，默认 4
        num_points: 每个query在每个层级的采样点数，默认 4
        batch_first: 输入是否是 batch first 格式，默认 True
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = True,
    ):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"but got {embed_dim} and {num_heads}"
            )
        
        head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        if not _is_power_of_2(head_dim):
            warnings.warn(
                "You'd better set embed_dim in MSDeformAttn to make sure that "
                "each dim of the attention head is a power of 2, which is more efficient."
            )
        
        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = head_dim
        
        # 采样偏移量预测
        # 输出: [num_heads * num_levels * num_points * 2]
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        
        # 注意力权重预测
        # 输出: [num_heads * num_levels * num_points]
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        
        # Value 投影
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 采样偏移量初始化为0
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        # 偏置初始化为网格状采样模式
        thetas = jt.arange(self.num_heads, dtype=jt.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = jt.stack([jt.cos(thetas), jt.sin(thetas)], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdims=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        self.sampling_offsets.bias = jt.Var(grid_init.view(-1))
        
        # 注意力权重初始化为0
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        # Value 和 Output 投影使用 xavier 初始化
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def execute(
        self,
        query: jt.Var,
        key: Optional[jt.Var] = None,
        value: Optional[jt.Var] = None,
        query_pos: Optional[jt.Var] = None,
        key_padding_mask: Optional[jt.Var] = None,
        reference_points: Optional[jt.Var] = None,
        spatial_shapes: Optional[jt.Var] = None,
        level_start_index: Optional[jt.Var] = None,
        **kwargs
    ) -> jt.Var:
        """
        前向传播
        
        Args:
            query: Query 张量
                - batch_first=True: [bs, num_query, embed_dim]
                - batch_first=False: [num_query, bs, embed_dim]
            key: Key 张量（未使用，为了接口兼容）
            value: Value 张量，如果为 None 则使用 query
                - batch_first=True: [bs, num_value, embed_dim]
                - batch_first=False: [num_value, bs, embed_dim]
            query_pos: Query 位置编码
            key_padding_mask: Padding 掩码 [bs, num_value]
            reference_points: 参考点 [bs, num_query, num_levels, 2] 或 [bs, num_query, num_levels, 4]
            spatial_shapes: 各层级空间尺寸 [num_levels, 2]
            level_start_index: 各层级起始索引 [num_levels]
            
        Returns:
            output: 输出张量，形状与 query 相同
        """
        # Force all inputs to correct dtypes to avoid float64/float32 mixing
        if query is not None: query = query.float32()
        if value is not None: value = value.float32()
        if query_pos is not None: query_pos = query_pos.float32()
        if reference_points is not None: reference_points = reference_points.float32()
        if spatial_shapes is not None: spatial_shapes = spatial_shapes.int32()
        if level_start_index is not None: level_start_index = level_start_index.int32()
        if key_padding_mask is not None: key_padding_mask = key_padding_mask.bool()

        if value is None:
            value = query
        
        if query_pos is not None:
            query = query + query_pos
        
        # 转换为 batch first 格式
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        
        # 验证空间形状
        assert spatial_shapes is not None, "spatial_shapes is required"
        # 计算总值数量（避免使用 .item()，改用 numpy）
        spatial_product = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        total_value = int(spatial_product.sum().data)
        assert total_value == num_value, \
            f"spatial shapes sum {total_value} doesn't match num_value {num_value}"
        
        # Value 投影
        value = self.value_proj(value)
        
        # 应用 padding mask
        if key_padding_mask is not None:
            value = jt.where(
                key_padding_mask.unsqueeze(-1),
                jt.zeros_like(value),
                value
            )
        
        # 重塑 value: [bs, num_value, num_heads, head_dim]
        value = value.view(bs, num_value, self.num_heads, self.head_dim)
        
        # 预测采样偏移量
        # [bs, num_query, num_heads * num_levels * num_points * 2]
        # -> [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # 预测注意力权重
        # [bs, num_query, num_heads * num_levels * num_points]
        # -> softmax -> [bs, num_query, num_heads, num_levels, num_points]
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = nn.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        
        # 计算采样位置
        # reference_points: [bs, num_query, num_levels, 2 or 4]
        assert reference_points is not None, "reference_points is required"
        
        if reference_points.shape[-1] == 2:
            # 2D 参考点: (x, y)
            # offset_normalizer: [num_levels, 2] -> (W, H)
            offset_normalizer = jt.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1
            )
            # sampling_locations: [bs, num_query, num_heads, num_levels, num_points, 2]
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            # 4D 参考点: (x, y, w, h)
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, "
                f"but got {reference_points.shape[-1]}"
            )
        
        # 计算多尺度可变形注意力
        output = multi_scale_deformable_attn_jittor(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        
        # 输出投影
        output = self.output_proj(output)
        
        # 转换回原始格式
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return output


# 别名，保持与官方实现兼容
MultiScaleDeformableAttention = MSDeformAttn


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("Testing MSDeformAttn...")
    
    # 参数
    bs = 2
    num_query = 100
    embed_dim = 256
    num_heads = 8
    num_levels = 4
    num_points = 4
    
    # 创建模块
    ms_deform_attn = MSDeformAttn(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=True,
    )
    
    # 创建输入
    # 多尺度特征图尺寸
    spatial_shapes = jt.array([[50, 50], [25, 25], [13, 13], [7, 7]])
    num_value = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
    level_start_index = jt.concat([
        jt.zeros((1,), dtype=jt.int32),
        (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1].int32()
    ])
    
    query = jt.randn(bs, num_query, embed_dim)
    value = jt.randn(bs, num_value, embed_dim)
    reference_points = jt.rand(bs, num_query, num_levels, 2)
    
    print(f"Query shape: {query.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Spatial shapes: {spatial_shapes}")
    print(f"Reference points shape: {reference_points.shape}")
    
    # 前向传播
    output = ms_deform_attn(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    
    print(f"Output shape: {output.shape}")
    assert output.shape == query.shape, f"Output shape mismatch: {output.shape} vs {query.shape}"
    
    print("MSDeformAttn test passed!")
