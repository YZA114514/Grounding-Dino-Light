# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Transformer Decoder Module (Member A)
# ------------------------------------------------------------------------
"""
Transformer Decoder 模块说明：

Decoder的作用是根据编码后的视觉和文本特征，生成目标检测的查询结果。
主要组件：
1. DeformableTransformerDecoderLayer: 解码器层
2. TransformerDecoder: 堆叠多个解码器层

关键设计：
- 自注意力：query之间的交互
- 文本交叉注意力：query与文本特征的交互
- 可变形交叉注意力：query与多尺度视觉特征的交互
- 迭代式边界框细化：每层输出的边界框作为下一层的参考点
"""

import copy
import math
from typing import Optional

import jittor as jt
from jittor import nn

# 导入 MultiheadAttention 以注入到 nn 模块
from ..attention import MultiheadAttention
if not hasattr(nn, 'MultiheadAttention'):
    nn.MultiheadAttention = MultiheadAttention


def _get_clones(module, N, layer_share=False):
    """克隆模块N次"""
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """返回激活函数"""
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return nn.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-5):
    """
    逆sigmoid函数
    将[0,1]范围的值转换回(-inf, +inf)
    """
    x = jt.clamp(x, min_v=eps, max_v=1 - eps)
    return jt.log(x / (1 - x))


def gen_sineembed_for_position(pos_tensor):
    """
    为位置张量生成正弦位置编码
    
    Args:
        pos_tensor: [n_query, bs, 2] 或 [n_query, bs, 4]
                   2维时为(x, y)，4维时为(x, y, w, h)
    
    Returns:
        pos: [n_query, bs, 256] 或 [n_query, bs, 512]
    """
    scale = 2 * math.pi
    dim_t = jt.arange(128, dtype=jt.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = y_embed.unsqueeze(-1) / dim_t
    
    pos_x = jt.stack((jt.sin(pos_x[:, :, 0::2]), jt.cos(pos_x[:, :, 1::2])), dim=3).flatten(2)
    pos_y = jt.stack((jt.sin(pos_y[:, :, 0::2]), jt.cos(pos_y[:, :, 1::2])), dim=3).flatten(2)
    
    if pos_tensor.shape[-1] == 2:
        pos = jt.concat((pos_y, pos_x), dim=2)
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed.unsqueeze(-1) / dim_t
        pos_w = jt.stack((jt.sin(pos_w[:, :, 0::2]), jt.cos(pos_w[:, :, 1::2])), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed.unsqueeze(-1) / dim_t
        pos_h = jt.stack((jt.sin(pos_h[:, :, 0::2]), jt.cos(pos_h[:, :, 1::2])), dim=3).flatten(2)

        pos = jt.concat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.shape[-1]}")
    
    return pos


class MLP(nn.Module):
    """
    简单的多层感知机
    
    用于：
    - 边界框回归
    - Query位置编码生成
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ])

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformerDecoderLayer(nn.Module):
    """
    可变形Transformer解码器层
    
    结构（按顺序）：
    1. 自注意力：query之间的交互
    2. 文本交叉注意力（可选）：query与文本特征的交互
    3. 可变形交叉注意力：query与多尺度视觉特征的交互
    4. 前馈网络
    
    每个子层后都有残差连接和LayerNorm
    """
    
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
    ):
        """
        Args:
            d_model: 模型维度
            d_ffn: FFN隐藏层维度
            dropout: dropout率
            activation: 激活函数
            n_levels: 特征图层级数
            n_heads: 注意力头数
            n_points: 每个注意力头的采样点数
            use_text_feat_guide: 是否使用文本特征引导
            use_text_cross_attention: 是否使用文本交叉注意力
        """
        super().__init__()

        # 可变形交叉注意力（与视觉特征）
        from ..attention.ms_deform_attn import MSDeformAttn
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # 文本交叉注意力（可选）
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        
        # 保存参数
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

    @staticmethod
    def with_pos_embed(tensor, pos):
        """将位置编码添加到张量"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """前馈网络"""
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def execute(
        self,
        # query相关
        tgt: Optional[jt.Var],  # [nq, bs, d_model]
        tgt_query_pos: Optional[jt.Var] = None,  # query位置编码
        tgt_query_sine_embed: Optional[jt.Var] = None,  # query正弦位置编码
        tgt_key_padding_mask: Optional[jt.Var] = None,
        tgt_reference_points: Optional[jt.Var] = None,  # [nq, bs, n_levels, 4]
        # 文本相关
        memory_text: Optional[jt.Var] = None,  # [bs, n_token, d_model]
        text_attention_mask: Optional[jt.Var] = None,  # [bs, n_token]
        # 视觉特征相关
        memory: Optional[jt.Var] = None,  # [hw, bs, d_model]
        memory_key_padding_mask: Optional[jt.Var] = None,
        memory_level_start_index: Optional[jt.Var] = None,
        memory_spatial_shapes: Optional[jt.Var] = None,
        memory_pos: Optional[jt.Var] = None,
        # 注意力掩码
        self_attn_mask: Optional[jt.Var] = None,
        cross_attn_mask: Optional[jt.Var] = None,
    ):
        """
        前向传播
        
        Args:
            tgt: 目标query [nq, bs, d_model]
            tgt_query_pos: query位置编码 [nq, bs, d_model]
            tgt_query_sine_embed: query正弦位置编码 [nq, bs, 256*2]
            tgt_key_padding_mask: query padding掩码
            tgt_reference_points: 参考点 [nq, bs, n_levels, 4]
            memory_text: 文本特征 [bs, n_token, d_model]
            text_attention_mask: 文本padding掩码 [bs, n_token]
            memory: 视觉特征 [hw, bs, d_model]
            memory_key_padding_mask: 视觉padding掩码 [bs, hw]
            memory_level_start_index: 层级起始索引
            memory_spatial_shapes: 空间尺寸
            memory_pos: 视觉位置编码 [hw, bs, d_model]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
            
        Returns:
            tgt: 更新后的query [nq, bs, d_model]
        """
        # 1. 自注意力
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            # print(f"Self Attn: min={tgt.min()}, max={tgt.max()}, mean={tgt.mean()}")

        # 2. 文本交叉注意力（可选）
        if self.use_text_cross_attention and memory_text is not None:
            tgt2, _ = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)
            # print(f"Text Cross Attn: min={tgt.min()}, max={tgt.max()}, mean={tgt.mean()}")

        # 3. 可变形交叉注意力（与视觉特征）
        # 使用多尺度可变形注意力
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # print(f"Deformable Cross Attn: min={tgt.min()}, max={tgt.max()}, mean={tgt.mean()}")

        # 4. FFN
        tgt = self.forward_ffn(tgt)
        # print(f"FFN: min={tgt.min()}, max={tgt.max()}, mean={tgt.mean()}")

        return tgt


class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    
    功能：
    1. 接收编码后的视觉和文本特征
    2. 通过多层解码生成检测query
    3. 迭代式细化边界框
    
    关键特性：
    - 返回所有中间层的输出（用于辅助损失）
    - 迭代式边界框细化：每层预测的边界框偏移量累加到参考点
    """
    
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
    ):
        """
        Args:
            decoder_layer: 解码器层
            num_layers: 层数
            norm: 归一化层
            return_intermediate: 是否返回中间层输出
            d_model: 模型维度
            query_dim: query维度（2或4）
            num_feature_levels: 特征层级数
        """
        super().__init__()
        
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        
        self.query_dim = query_dim
        assert query_dim in [2, 4], f"query_dim should be 2/4 but {query_dim}"
        
        self.num_feature_levels = num_feature_levels
        self.d_model = d_model

        # 参考点位置编码头
        # 将正弦位置编码转换为query位置编码
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.query_pos_sine_scale = None
        self.query_scale = None
        
        # 这些将在外部设置（由GroundingDINO主模型设置）
        self.bbox_embed = None  # 边界框回归头
        self.class_embed = None  # 分类头

        self.ref_anchor_head = None

    def execute(
        self,
        tgt,
        memory,
        tgt_mask: Optional[jt.Var] = None,
        memory_mask: Optional[jt.Var] = None,
        tgt_key_padding_mask: Optional[jt.Var] = None,
        memory_key_padding_mask: Optional[jt.Var] = None,
        pos: Optional[jt.Var] = None,
        refpoints_unsigmoid: Optional[jt.Var] = None,
        # 多尺度相关
        level_start_index: Optional[jt.Var] = None,
        spatial_shapes: Optional[jt.Var] = None,
        valid_ratios: Optional[jt.Var] = None,
        # 文本相关
        memory_text: Optional[jt.Var] = None,
        text_attention_mask: Optional[jt.Var] = None,
    ):
        """
        前向传播
        
        Args:
            tgt: 目标query [nq, bs, d_model]
            memory: 编码后的视觉特征 [hw, bs, d_model]
            tgt_mask: query自注意力掩码
            memory_mask: 交叉注意力掩码
            tgt_key_padding_mask: query padding掩码
            memory_key_padding_mask: 视觉padding掩码 [bs, hw]
            pos: 视觉位置编码 [hw, bs, d_model]
            refpoints_unsigmoid: 参考点（未sigmoid）[nq, bs, 4]
            level_start_index: 层级起始索引 [num_levels]
            spatial_shapes: 空间尺寸 [num_levels, 2]
            valid_ratios: 有效比例 [bs, num_levels, 2]
            memory_text: 文本特征 [bs, n_token, d_model]
            text_attention_mask: 文本padding掩码 [bs, n_token]
            
        Returns:
            intermediate: 各层输出 [[bs, nq, d_model], ...]
            ref_points: 各层参考点 [[bs, nq, 4], ...]
        """
        output = tgt

        intermediate = []
        reference_points = jt.sigmoid(refpoints_unsigmoid)
        ref_points = [reference_points]
        print("Starting Decoder Loop")

        for layer_id, layer in enumerate(self.layers):
            # 准备参考点输入
            if reference_points.shape[-1] == 4:
                # 4维参考点：(cx, cy, w, h)
                reference_points_input = (
                    reference_points.unsqueeze(2)
                    * jt.concat([valid_ratios, valid_ratios], dim=-1).unsqueeze(0)
                )  # [nq, bs, nlevel, 4]
            else:
                # 2维参考点：(cx, cy)
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points.unsqueeze(2) * valid_ratios.unsqueeze(0)
            
            # 生成query正弦位置编码
            # if jt.isnan(reference_points_input).any(): # print(f"Layer {layer_id}: NaN in reference_points_input")
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # [nq, bs, 256*2]

            # 条件query位置编码
            raw_query_pos = self.ref_point_head(query_sine_embed)  # [nq, bs, d_model]
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # 解码器层前向传播
            print(f"--- Decoder Layer {layer_id} Start ---")
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            print(f"--- Decoder Layer {layer_id} End ---")
            # print(f"Layer {layer_id} output: min={output.min()}, max={output.max()}, mean={output.mean()}")

            # 迭代式边界框细化
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = jt.sigmoid(outputs_unsig)

                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        # 返回格式：转置为 [bs, nq, d_model]
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


def build_decoder_layer(
    d_model=256,
    d_ffn=1024,
    dropout=0.1,
    activation="relu",
    n_levels=4,
    n_heads=8,
    n_points=4,
    use_text_cross_attention=False,
):
    """
    构建解码器层
    """
    return DeformableTransformerDecoderLayer(
        d_model=d_model,
        d_ffn=d_ffn,
        dropout=dropout,
        activation=activation,
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points,
        use_text_cross_attention=use_text_cross_attention,
    )


def build_decoder(
    decoder_layer,
    num_layers,
    d_model=256,
    query_dim=4,
    num_feature_levels=4,
):
    """
    构建解码器
    """
    decoder_norm = nn.LayerNorm(d_model)
    return TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_layers,
        norm=decoder_norm,
        return_intermediate=True,
        d_model=d_model,
        query_dim=query_dim,
        num_feature_levels=num_feature_levels,
    )
