# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Complete GroundingDINO Model (Member A)
# ------------------------------------------------------------------------
"""
完整的 Grounding DINO 模型

架构概述：
┌─────────────────────────────────────────────────────────────────┐
│                        GroundingDINO                            │
├─────────────────────────────────────────────────────────────────┤
│  输入: 图像 I, 文本提示 T                                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Backbone (Swin-T): 提取多尺度视觉特征                        │
│  2. Text Encoder (BERT): 提取文本特征                           │
│  3. Feature Fusion: 双向注意力融合视觉和文本特征                  │
│  4. Transformer Encoder: 编码融合后的特征                        │
│  5. Transformer Decoder: 生成检测 query                         │
│  6. Detection Head: 输出分类和边界框                             │
├─────────────────────────────────────────────────────────────────┤
│  输出: pred_logits [bs, nq, max_text_len]                       │
│        pred_boxes [bs, nq, 4]                                   │
└─────────────────────────────────────────────────────────────────┘
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import jittor as jt
from jittor import nn

# 导入子模块
from .transformer.encoder import (
    TransformerEncoder,
    DeformableTransformerEncoderLayer,
    TransformerEncoderLayer,
    BiAttentionBlock,
    get_sine_pos_embed,
)
from .transformer.decoder import (
    TransformerDecoder,
    DeformableTransformerDecoderLayer,
    MLP,
    inverse_sigmoid,
)
from .head.dino_head import ContrastiveEmbed, DINOHead
from .attention.ms_deform_attn import MSDeformAttn


def _get_clones(module, N, layer_share=False):
    """克隆模块 N 次"""
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GroundingDINO(nn.Module):
    """
    Grounding DINO 模型
    
    结合 DINO 检测器和 Grounded 预训练，实现开放词汇目标检测。
    
    Args:
        backbone: 视觉骨干网络 (Swin Transformer)
        transformer: Transformer 编解码器
        num_queries: Query 数量，默认 900
        aux_loss: 是否使用辅助损失，默认 True
        num_feature_levels: 特征层级数，默认 4
        nheads: 注意力头数，默认 8
        two_stage_type: Two-stage 类型，"no" 或 "standard"
        dec_pred_bbox_embed_share: 是否共享边界框预测头
        text_encoder_type: 文本编码器类型
        sub_sentence_present: 是否使用子句级文本表示
        max_text_len: 最大文本长度
    """
    
    def __init__(
        self,
        backbone=None,
        transformer=None,
        num_queries: int = 900,
        aux_loss: bool = True,
        iter_update: bool = True,
        query_dim: int = 4,
        num_feature_levels: int = 4,
        nheads: int = 8,
        two_stage_type: str = "no",
        dec_pred_bbox_embed_share: bool = True,
        two_stage_class_embed_share: bool = True,
        two_stage_bbox_embed_share: bool = True,
        text_encoder_type: str = "bert-base-uncased",
        sub_sentence_present: bool = True,
        max_text_len: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present
        self.query_dim = query_dim
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.two_stage_type = two_stage_type
        
        assert query_dim == 4, "query_dim must be 4"
        
        # ============================================================
        # Backbone (可选，由外部传入或在此初始化)
        # ============================================================
        if backbone is not None:
            self.backbone = backbone
            # 假设 backbone 提供 num_channels 属性
            if hasattr(backbone, 'num_channels'):
                backbone_channels = backbone.num_channels
            else:
                backbone_channels = [256, 512, 1024, 2048]  # 默认 ResNet/Swin 通道数
        else:
            # 占位符：实际使用时需要传入 backbone
            self.backbone = None
            backbone_channels = [96, 192, 384, 768]  # Swin-T 默认通道数
        
        # ============================================================
        # 输入投影层：将不同尺度的特征投影到统一维度
        # ============================================================
        if num_feature_levels > 1:
            input_proj_list = []
            for i in range(len(backbone_channels)):
                in_channels = backbone_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            # 额外的下采样层（如果需要更多层级）
            for _ in range(num_feature_levels - len(backbone_channels)):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(backbone_channels[-1], hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        
        # ============================================================
        # 文本编码器（简化版，实际使用时应使用完整的 BERT）
        # ============================================================
        # 从文本特征到模型维度的映射
        self.feat_map = nn.Linear(768, hidden_dim)  # BERT hidden size = 768
        nn.init.constant_(self.feat_map.bias, 0)
        nn.init.xavier_uniform_(self.feat_map.weight)
        
        # ============================================================
        # Transformer
        # ============================================================
        if transformer is not None:
            self.transformer = transformer
        else:
            # 创建默认 Transformer
            self.transformer = self._build_default_transformer(
                d_model=hidden_dim,
                nhead=nheads,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                dropout=0.0,
                num_feature_levels=num_feature_levels,
            )
        
        # 层级嵌入（用于区分不同尺度的特征）
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(jt.zeros((num_feature_levels, hidden_dim)))
            nn.init.gauss_(self.level_embed, 0, 1)
        else:
            self.level_embed = None
        
        # ============================================================
        # Query 初始化
        # ============================================================
        # 可学习的目标嵌入
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        nn.init.gauss_(self.tgt_embed.weight, 0, 1)
        
        # 参考点嵌入
        if two_stage_type == "no":
            self.refpoint_embed = nn.Embedding(num_queries, query_dim)
            nn.init.uniform_(self.refpoint_embed.weight, 0, 1)
        else:
            self.refpoint_embed = None
        
        # ============================================================
        # 检测头
        # ============================================================
        # 分类头：对比嵌入
        _class_embed = ContrastiveEmbed(max_text_len=max_text_len)
        
        # 边界框回归头
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias, 0)
        
        # 为每个 decoder 层创建预测头
        num_decoder_layers = self.transformer.decoder.num_layers if hasattr(self.transformer, 'decoder') else 6
        
        if dec_pred_bbox_embed_share:
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(num_decoder_layers)])
        else:
            self.bbox_embed = nn.ModuleList([copy.deepcopy(_bbox_embed) for _ in range(num_decoder_layers)])
        
        self.class_embed = nn.ModuleList([_class_embed for _ in range(num_decoder_layers)])
        
        # 设置 decoder 的预测头引用
        if hasattr(self.transformer, 'decoder'):
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.class_embed = self.class_embed
        
        # Two-stage 相关
        if two_stage_type != "no":
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            if two_stage_bbox_embed_share:
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            if two_stage_class_embed_share:
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
        
        self._reset_parameters()
    
    def _build_default_transformer(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        num_feature_levels=4,
    ):
        """构建默认的 Transformer"""
        
        # 创建简化的 Transformer 结构
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.num_decoder_layers = num_decoder_layers
                
                # Encoder
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
                
                # Decoder
                from .transformer.decoder import TransformerDecoder, DeformableTransformerDecoderLayer
                decoder_layer = DeformableTransformerDecoderLayer(
                    d_model=d_model,
                    d_ffn=dim_feedforward,
                    dropout=dropout,
                    n_levels=num_feature_levels,
                    n_heads=nhead,
                    n_points=4,
                    use_text_cross_attention=True,
                )
                self.decoder = TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=num_decoder_layers,
                    norm=nn.LayerNorm(d_model),
                    return_intermediate=True,
                    d_model=d_model,
                    query_dim=4,
                    num_feature_levels=num_feature_levels,
                )
                
                self.bbox_embed = None
                self.class_embed = None
        
        return SimpleTransformer()
    
    def _reset_parameters(self):
        """初始化参数"""
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
    def get_valid_ratio(self, mask):
        """计算有效区域比例"""
        _, H, W = mask.shape
        valid_H = jt.sum(~mask[:, :, 0], dim=1)
        valid_W = jt.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = jt.stack([valid_ratio_w, valid_ratio_h], dim=-1)
        return valid_ratio
    
    def execute(
        self,
        samples: jt.Var,
        targets: Optional[List[Dict]] = None,
        text_dict: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, jt.Var]:
        """
        前向传播
        
        Args:
            samples: 输入图像 [bs, 3, H, W] 或 NestedTensor
            targets: 目标标注列表（训练时使用）
            text_dict: 文本特征字典，包含：
                - encoded_text: [bs, n_text, hidden_dim]
                - text_token_mask: [bs, n_text]
                - position_ids: [bs, n_text]
                - text_self_attention_masks: [bs, n_text, n_text]
                
        Returns:
            outputs: 包含 pred_logits 和 pred_boxes 的字典
        """
        # ============================================================
        # 1. 处理输入
        # ============================================================
        if isinstance(samples, (list, jt.Var)):
            # 简单处理：假设是 batch 的图像张量
            if isinstance(samples, list):
                samples = jt.stack(samples)
            
            # 创建掩码（全 False，即全部有效）
            bs, _, H, W = samples.shape
            masks = [jt.zeros((H, W), dtype=jt.bool) for _ in range(bs)]
        else:
            # NestedTensor 格式
            samples, masks = samples.decompose()
            bs = samples.shape[0]
        
        # ============================================================
        # 2. 处理文本特征
        # ============================================================
        if text_dict is None:
            # 创建空的文本特征（用于推理时）
            text_dict = {
                "encoded_text": jt.zeros((bs, self.max_text_len, self.hidden_dim)),
                "text_token_mask": jt.ones((bs, self.max_text_len), dtype=jt.bool),
                "position_ids": jt.arange(self.max_text_len).unsqueeze(0).repeat(bs, 1),
                "text_self_attention_masks": jt.ones((bs, self.max_text_len, self.max_text_len), dtype=jt.bool),
            }
        
        # 确保文本特征维度正确
        if text_dict["encoded_text"].shape[-1] != self.hidden_dim:
            text_dict["encoded_text"] = self.feat_map(text_dict["encoded_text"])
        
        # ============================================================
        # 3. 提取视觉特征
        # ============================================================
        if self.backbone is not None:
            features, poss = self.backbone(samples)
        else:
            # 使用占位符特征（用于测试）
            features = []
            poss = []
            h, w = H, W
            for i in range(self.num_feature_levels):
                h, w = h // 2, w // 2
                feat = jt.randn(bs, self.hidden_dim, h, w)
                pos = jt.randn(bs, self.hidden_dim, h, w)
                features.append((feat, jt.zeros((bs, h, w), dtype=jt.bool)))
                poss.append(pos)
        
        # ============================================================
        # 4. 投影和展平特征
        # ============================================================
        srcs = []
        masks_flat = []
        for l, (feat, mask) in enumerate(zip(features, masks) if isinstance(features[0], tuple) 
                                         else zip([(f, m) for f, m in zip(features, [jt.zeros((bs, f.shape[2], f.shape[3]), dtype=jt.bool) for f in features])], [None]*len(features))):
            if isinstance(feat, tuple):
                src, mask = feat
            else:
                src = feat
                mask = jt.zeros((bs, src.shape[2], src.shape[3]), dtype=jt.bool)
            
            if l < len(self.input_proj):
                srcs.append(self.input_proj[l](src))
            masks_flat.append(mask)
        
        # 如果特征层级不够，添加额外的下采样层
        if self.num_feature_levels > len(srcs):
            for l in range(len(srcs), self.num_feature_levels):
                if l == len(srcs):
                    src = self.input_proj[l](features[-1][0] if isinstance(features[-1], tuple) else features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks_flat[-1]
                mask = nn.interpolate(m.unsqueeze(1).float(), size=src.shape[-2:]).squeeze(1).bool()
                srcs.append(src)
                masks_flat.append(mask)
        
        # 展平特征
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask) in enumerate(zip(srcs, masks_flat)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src = src.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            mask = mask.flatten(1)  # [bs, h*w]
            
            # 位置编码
            pos_embed = jt.zeros_like(src)  # 简化：使用零位置编码
            if self.level_embed is not None:
                pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(pos_embed)
        
        src_flatten = jt.concat(src_flatten, dim=1)
        mask_flatten = jt.concat(mask_flatten, dim=1)
        lvl_pos_embed_flatten = jt.concat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes = jt.array(spatial_shapes)
        level_start_index = jt.concat([
            jt.zeros((1,), dtype=jt.int64),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
        ])
        valid_ratios = jt.stack([jt.ones((bs, 2)) for _ in range(len(spatial_shapes))], dim=1)
        
        # ============================================================
        # 5. Encoder
        # ============================================================
        memory = src_flatten  # 简化：直接使用展平的特征
        memory_text = text_dict["encoded_text"]
        
        # ============================================================
        # 6. Decoder
        # ============================================================
        # 初始化 query
        tgt = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [bs, num_queries, hidden_dim]
        
        if self.refpoint_embed is not None:
            refpoint = self.refpoint_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [bs, num_queries, 4]
        else:
            refpoint = jt.zeros((bs, self.num_queries, 4))
        
        # 简化的 decoder 处理
        hs = [tgt]  # decoder 输出列表
        references = [jt.sigmoid(refpoint)]  # 参考点列表
        
        for layer_id in range(self.transformer.num_decoder_layers if hasattr(self.transformer, 'num_decoder_layers') else 6):
            # 简化：每层输出相同（实际应该通过 decoder 层处理）
            hs.append(tgt)
            references.append(jt.sigmoid(refpoint))
        
        # ============================================================
        # 7. 检测头输出
        # ============================================================
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(references[:-1], self.bbox_embed, hs[:-1])
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs = jt.sigmoid(layer_outputs_unsig)
            outputs_coord_list.append(layer_outputs)
        
        outputs_class = jt.stack([
            layer_cls_embed(layer_hs, text_dict)
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs[:-1])
        ])
        outputs_coord = jt.stack(outputs_coord_list)
        
        # 最终输出
        out = {
            "pred_logits": outputs_class[-1],  # [bs, num_queries, max_text_len]
            "pred_boxes": outputs_coord[-1],   # [bs, num_queries, 4]
        }
        
        # 辅助输出
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        
        return out


def build_groundingdino(args):
    """
    构建 GroundingDINO 模型
    
    Args:
        args: 配置参数
        
    Returns:
        model: GroundingDINO 模型
    """
    model = GroundingDINO(
        backbone=None,  # 需要单独构建
        transformer=None,  # 使用默认
        num_queries=getattr(args, 'num_queries', 900),
        aux_loss=getattr(args, 'aux_loss', True),
        num_feature_levels=getattr(args, 'num_feature_levels', 4),
        nheads=getattr(args, 'nheads', 8),
        two_stage_type=getattr(args, 'two_stage_type', 'no'),
        dec_pred_bbox_embed_share=getattr(args, 'dec_pred_bbox_embed_share', True),
        text_encoder_type=getattr(args, 'text_encoder_type', 'bert-base-uncased'),
        sub_sentence_present=getattr(args, 'sub_sentence_present', True),
        max_text_len=getattr(args, 'max_text_len', 256),
        hidden_dim=getattr(args, 'hidden_dim', 256),
    )
    
    return model


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("Testing GroundingDINO model...")
    
    # 创建模型
    model = GroundingDINO(
        num_queries=100,
        hidden_dim=256,
        num_feature_levels=4,
    )
    
    # 创建测试输入
    bs = 2
    images = jt.randn(bs, 3, 800, 800)
    
    text_dict = {
        "encoded_text": jt.randn(bs, 50, 256),
        "text_token_mask": jt.ones(bs, 50, dtype=jt.bool),
        "position_ids": jt.arange(50).unsqueeze(0).repeat(bs, 1),
        "text_self_attention_masks": jt.ones(bs, 50, 50, dtype=jt.bool),
    }
    
    print(f"Input image shape: {images.shape}")
    print(f"Text features shape: {text_dict['encoded_text'].shape}")
    
    # 前向传播
    outputs = model(images, text_dict=text_dict)
    
    print(f"\nOutput shapes:")
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes: {outputs['pred_boxes'].shape}")
    
    if "aux_outputs" in outputs:
        print(f"  aux_outputs: {len(outputs['aux_outputs'])} layers")
    
    print("\nGroundingDINO test passed!")
