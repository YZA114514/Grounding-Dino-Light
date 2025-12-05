# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Transformer Encoder Module (Member A)
# ------------------------------------------------------------------------
"""
Transformer Encoder 模块说明：

Encoder的作用是对多尺度视觉特征进行编码，同时与文本特征进行融合。
主要组件：
1. DeformableTransformerEncoderLayer: 使用可变形注意力的编码器层
2. TransformerEncoder: 堆叠多个编码器层，并集成文本增强和特征融合

关键设计：
- 多尺度可变形注意力(MSDeformAttn): 高效处理不同分辨率的特征图
- 文本增强层(Text Enhance Layer): 对文本特征进行自注意力处理
- 特征融合层(Feature Fusion Layer): 双向注意力融合视觉和文本特征
"""

import copy
import math
from typing import Optional, List

import jittor as jt
from jittor import nn

# 需要导入的模块（假设已实现）
# from ..attention.ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn


def _get_clones(module, N, layer_share=False):
    """
    克隆模块N次
    Args:
        module: 要克隆的模块
        N: 克隆次数
        layer_share: 是否共享权重（True则所有层共享同一模块）
    """
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


def get_sine_pos_embed(
    pos_tensor: jt.Var,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """
    生成正弦位置编码
    
    Args:
        pos_tensor: 位置张量，shape: [..., n]
        num_pos_feats: 每个位置坐标的特征维度
        temperature: 温度参数
        exchange_xy: 是否交换x和y的位置编码
    
    Returns:
        pos_embed: shape: [..., n*num_pos_feats]
    """
    scale = 2 * math.pi
    dim_t = jt.arange(num_pos_feats, dtype=jt.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    def sine_func(x: jt.Var):
        sin_x = x * scale / dim_t
        # 交替使用sin和cos
        sin_x_sin = jt.sin(sin_x[..., 0::2])
        sin_x_cos = jt.cos(sin_x[..., 1::2])
        sin_x = jt.stack((sin_x_sin, sin_x_cos), dims=-1).flatten(-2)
        return sin_x

    # 对每个坐标分量生成位置编码
    pos_res = []
    for i in range(pos_tensor.shape[-1]):
        pos_res.append(sine_func(pos_tensor[..., i:i+1]))
    
    if exchange_xy and len(pos_res) >= 2:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    
    pos_res = jt.concat(pos_res, dim=-1)
    return pos_res


class DeformableTransformerEncoderLayer(nn.Module):
    """
    可变形Transformer编码器层
    
    结构：
    1. 多尺度可变形自注意力 (MSDeformAttn)
    2. 前馈网络 (FFN)
    
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
    ):
        """
        Args:
            d_model: 模型维度
            d_ffn: 前馈网络隐藏层维度
            dropout: dropout率
            activation: 激活函数类型
            n_levels: 特征图层级数
            n_heads: 注意力头数
            n_points: 每个注意力头的采样点数
        """
        super().__init__()

        # 多尺度可变形自注意力
        # 注意：MSDeformAttn需要单独实现，这里先用占位符
        # self.self_attn = MSDeformAttn(
        #     embed_dim=d_model,
        #     num_levels=n_levels,
        #     num_heads=n_heads,
        #     num_points=n_points,
        #     batch_first=True,
        # )
        
        # 临时使用标准多头注意力作为占位
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.use_deformable = False  # 标记是否使用可变形注意力
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络 (FFN)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 保存参数用于后续替换为真正的MSDeformAttn
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

    @staticmethod
    def with_pos_embed(tensor, pos):
        """将位置编码添加到张量"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        """前馈网络"""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def execute(
        self, 
        src, 
        pos, 
        reference_points=None, 
        spatial_shapes=None, 
        level_start_index=None, 
        key_padding_mask=None
    ):
        """
        前向传播
        
        Args:
            src: 输入特征 [bs, sum(hi*wi), d_model]
            pos: 位置编码 [bs, sum(hi*wi), d_model]
            reference_points: 参考点 [bs, sum(hi*wi), n_levels, 2]
            spatial_shapes: 各层级的空间尺寸 [n_levels, 2]
            level_start_index: 各层级的起始索引 [n_levels]
            key_padding_mask: padding掩码 [bs, sum(hi*wi)]
        
        Returns:
            src: 编码后的特征 [bs, sum(hi*wi), d_model]
        """
        if self.use_deformable:
            # 使用可变形注意力
            src2 = self.self_attn(
                query=self.with_pos_embed(src, pos),
                reference_points=reference_points,
                value=src,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
        else:
            # 使用标准多头注意力（临时方案）
            # 需要转置为 [seq_len, batch, d_model] 格式
            src_t = src.transpose(0, 1)
            pos_t = pos.transpose(0, 1) if pos is not None else None
            q = k = self.with_pos_embed(src_t, pos_t)
            src2, _ = self.self_attn(q, k, src_t)
            src2 = src2.transpose(0, 1)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src = self.forward_ffn(src)

        return src


class TransformerEncoderLayer(nn.Module):
    """
    标准Transformer编码器层（用于文本增强）
    
    结构：
    1. 多头自注意力
    2. 前馈网络
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[jt.Var]):
        return tensor if pos is None else tensor + pos

    def execute(
        self,
        src,
        src_mask: Optional[jt.Var] = None,
        src_key_padding_mask: Optional[jt.Var] = None,
        pos: Optional[jt.Var] = None,
    ):
        """
        Args:
            src: [seq_len, batch, d_model]
            src_mask: 注意力掩码 [batch, seq_len, seq_len] 或 [seq_len, seq_len]
            src_key_padding_mask: padding掩码 [batch, seq_len]
            pos: 位置编码 [seq_len, batch, d_model]
        """
        # 处理注意力掩码
        if src_mask is not None and src_mask.ndim == 3 and src_mask.shape[0] == src.shape[1]:
            # [bs, num_q, num_k] -> [bs*nhead, num_q, num_k]
            src_mask = src_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            src_mask = src_mask.reshape(-1, src_mask.shape[2], src_mask.shape[3])

        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, src, attn_mask=src_mask)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class BiMultiHeadAttention(nn.Module):
    """
    双向多头注意力
    
    实现视觉和语言特征之间的双向注意力交互：
    - 视觉 -> 语言：视觉特征作为query，语言特征作为key/value
    - 语言 -> 视觉：语言特征作为query，视觉特征作为key/value
    """
    
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert self.head_dim * self.num_heads == self.embed_dim, \
            f"embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        # 投影层
        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        # 数值稳定性参数
        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: jt.Var, seq_len: int, bsz: int):
        """重塑张量为多头格式"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _reset_parameters(self):
        """初始化参数"""
        for module in [self.v_proj, self.l_proj, self.values_v_proj, 
                       self.values_l_proj, self.out_v_proj, self.out_l_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def execute(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """
        Args:
            v: 视觉特征 [bs, n_img, v_dim]
            l: 语言特征 [bs, n_text, l_dim]
            attention_mask_v: 视觉掩码 [bs, n_img]，True表示padding
            attention_mask_l: 语言掩码 [bs, n_text]，True表示padding

        Returns:
            attn_output_v: 更新后的视觉特征 [bs, n_img, v_dim]
            attn_output_l: 更新后的语言特征 [bs, n_text, l_dim]
        """
        bsz, tgt_len, _ = v.size()

        # 投影
        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        
        # 计算注意力权重: [bs*nhead, n_img, n_text]
        attn_weights = jt.bmm(query_states, key_states.transpose(1, 2))

        # 数值稳定性处理
        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = jt.clamp(attn_weights, min_v=-50000)
        if self.clamp_max_for_overflow:
            attn_weights = jt.clamp(attn_weights, max_v=50000)

        # 计算语言到视觉的注意力
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - attn_weights_T.max(dim=-1, keepdims=True)[0]
        
        if self.clamp_min_for_underflow:
            attn_weights_l = jt.clamp(attn_weights_l, min_v=-50000)
        if self.clamp_max_for_overflow:
            attn_weights_l = jt.clamp(attn_weights_l, max_v=50000)

        # 应用掩码
        if attention_mask_v is not None:
            attention_mask_v = attention_mask_v.unsqueeze(1).unsqueeze(2)
            attention_mask_v = attention_mask_v.repeat(1, self.num_heads, 1, 1)
            attention_mask_v = attention_mask_v.reshape(-1, 1, attention_mask_v.shape[-1])
            attn_weights_l = jt.where(
                attention_mask_v, 
                jt.full_like(attn_weights_l, float("-inf")), 
                attn_weights_l
            )

        attn_weights_l = nn.softmax(attn_weights_l, dim=-1)

        if attention_mask_l is not None:
            attention_mask_l = attention_mask_l.unsqueeze(1).unsqueeze(2)
            attention_mask_l = attention_mask_l.repeat(1, self.num_heads, 1, 1)
            attention_mask_l = attention_mask_l.reshape(-1, 1, attention_mask_l.shape[-1])
            attn_weights = jt.where(
                attention_mask_l,
                jt.full_like(attn_weights, float("-inf")),
                attn_weights
            )
        
        attn_weights_v = nn.softmax(attn_weights, dim=-1)

        # Dropout
        if self.training and self.dropout > 0:
            attn_probs_v = nn.dropout(attn_weights_v, p=self.dropout)
            attn_probs_l = nn.dropout(attn_weights_l, p=self.dropout)
        else:
            attn_probs_v = attn_weights_v
            attn_probs_l = attn_weights_l

        # 计算输出
        attn_output_v = jt.bmm(attn_probs_v, value_l_states)
        attn_output_l = jt.bmm(attn_probs_l, value_v_states)

        # 重塑回原始形状
        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(bsz, src_len, self.embed_dim)

        # 输出投影
        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class BiAttentionBlock(nn.Module):
    """
    双向注意力块
    
    包含：
    1. LayerNorm
    2. 双向多头注意力
    3. 残差连接 + LayerScale
    """
    
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
    ):
        super().__init__()

        # Pre-LayerNorm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, 
            l_dim=l_dim, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout
        )

        # LayerScale参数
        self.gamma_v = jt.Var(init_values * jt.ones((v_dim,)))
        self.gamma_l = jt.Var(init_values * jt.ones((l_dim,)))
        
        # DropPath（这里简化为Identity）
        self.drop_path = drop_path

    def execute(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """
        Args:
            v: 视觉特征 [bs, n_img, v_dim]
            l: 语言特征 [bs, n_text, l_dim]
        """
        v_normed = self.layer_norm_v(v)
        l_normed = self.layer_norm_l(l)
        
        delta_v, delta_l = self.attn(
            v_normed, l_normed, 
            attention_mask_v=attention_mask_v, 
            attention_mask_l=attention_mask_l
        )
        
        # 残差连接 + LayerScale
        v = v + self.gamma_v * delta_v
        l = l + self.gamma_l * delta_l
        
        return v, l


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    
    功能：
    1. 对多尺度视觉特征进行编码
    2. 对文本特征进行增强
    3. 融合视觉和文本特征
    
    处理流程（每层）：
    1. 特征融合层：双向注意力融合视觉和文本
    2. 文本增强层：文本自注意力
    3. 视觉编码层：可变形注意力编码视觉特征
    """
    
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        super().__init__()
        
        # 编码器层
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点
        
        对于每个特征图位置，生成归一化的参考点坐标
        
        Args:
            spatial_shapes: 各层级空间尺寸 [(H1,W1), (H2,W2), ...]
            valid_ratios: 有效区域比例 [bs, num_levels, 2]
            device: 设备
            
        Returns:
            reference_points: [bs, sum(Hi*Wi), num_levels, 2]
        """
        reference_points_list = []
        
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成网格坐标
            ref_y, ref_x = jt.meshgrid(
                jt.linspace(0.5, float(H_) - 0.5, int(H_)),
                jt.linspace(0.5, float(W_) - 0.5, int(W_)),
            )
            # 归一化
            ref_y = ref_y.reshape(-1).unsqueeze(0) / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1).unsqueeze(0) / (valid_ratios[:, None, lvl, 0] * W_)
            ref = jt.stack((ref_x, ref_y), dim=-1)
            reference_points_list.append(ref)
        
        reference_points = jt.concat(reference_points_list, dim=1)
        reference_points = reference_points.unsqueeze(2) * valid_ratios.unsqueeze(1)
        
        return reference_points

    def execute(
        self,
        # 视觉输入
        src: jt.Var,
        pos: jt.Var,
        spatial_shapes: jt.Var,
        level_start_index: jt.Var,
        valid_ratios: jt.Var,
        key_padding_mask: jt.Var,
        # 文本输入
        memory_text: jt.Var = None,
        text_attention_mask: jt.Var = None,
        pos_text: jt.Var = None,
        text_self_attention_masks: jt.Var = None,
        position_ids: jt.Var = None,
    ):
        """
        前向传播
        
        Args:
            src: 视觉特征 [bs, sum(hi*wi), d_model]
            pos: 位置编码 [bs, sum(hi*wi), d_model]
            spatial_shapes: 空间尺寸 [num_level, 2]
            level_start_index: 层级起始索引 [num_level]
            valid_ratios: 有效比例 [bs, num_level, 2]
            key_padding_mask: padding掩码 [bs, sum(hi*wi)]
            memory_text: 文本特征 [bs, n_text, d_model]
            text_attention_mask: 文本padding掩码 [bs, n_text]
            pos_text: 文本位置编码 [bs, n_text, d_model]
            text_self_attention_masks: 文本自注意力掩码 [bs, n_text, n_text]
            position_ids: 位置ID [bs, n_text]
            
        Returns:
            output: 编码后的视觉特征 [bs, sum(hi*wi), d_model]
            memory_text: 增强后的文本特征 [bs, n_text, d_model]
        """
        output = src

        # 生成参考点
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        # 生成文本位置编码
        if self.text_layers:
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = jt.arange(n_text).float().unsqueeze(0).unsqueeze(-1)
                pos_text = pos_text.repeat(bs, 1, 1)
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids.unsqueeze(-1), num_pos_feats=256, exchange_xy=False
                )

        # 主处理循环
        for layer_id, layer in enumerate(self.layers):
            # 1. 特征融合
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    v=output,
                    l=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )

            # 2. 文本增强
            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks if text_self_attention_masks is not None else None,
                    src_key_padding_mask=text_attention_mask,
                    pos=pos_text.transpose(0, 1) if pos_text is not None else None,
                ).transpose(0, 1)

            # 3. 视觉编码
            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )

        return output, memory_text
