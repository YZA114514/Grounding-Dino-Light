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
from .text_encoder import BERTWrapper


class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Uses separate temperatures for H and W dimensions (matching PyTorch implementation).
    """
    def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def execute(self, tensor_list: jt.Var, mask: Optional[jt.Var] = None):
        x = tensor_list
        if mask is None:
            mask = jt.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=jt.bool)
        
        not_mask = jt.logical_not(mask).float32()
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Use floor division like PyTorch: torch.div(dim_tx, 2, rounding_mode='floor')
        dim_tx = jt.arange(self.num_pos_feats, dtype=jt.float32)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed.unsqueeze(-1) / dim_tx

        dim_ty = jt.arange(self.num_pos_feats, dtype=jt.float32)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed.unsqueeze(-1) / dim_ty
        
        pos_x = jt.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = jt.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = jt.concat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



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
        embed_init_tgt: bool = True,  # Whether to use learnable tgt_embed in two-stage
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
        self.embed_init_tgt = embed_init_tgt  # Store for use in forward
        
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
                # Swin-T 输出通道数: [192, 384, 768] + 额外下采样 768
                # 匹配官方预训练权重
                backbone_channels = [192, 384, 768]
        else:
            # 占位符：实际使用时需要传入 backbone
            self.backbone = None
            # Swin-T 输出通道数: [192, 384, 768] + 额外下采样 768
            # 匹配官方预训练权重
            backbone_channels = [192, 384, 768]        # ============================================================
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
            in_channels = backbone_channels[-1]
            for _ in range(num_feature_levels - len(backbone_channels)):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim  # 后续层输入是 hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])
        
        # ============================================================
        # 文本编码器 (BERT)
        # ============================================================
        # 从文本特征到模型维度的映射
        self.feat_map = nn.Linear(768, hidden_dim)  # BERT hidden size = 768
        nn.init.constant_(self.feat_map.bias, 0)
        nn.init.xavier_uniform_(self.feat_map.weight)
        
        # 初始化 BERT 编码器
        self.text_encoder = BERTWrapper(
            model_name=text_encoder_type,
            max_text_len=max_text_len
        )
        # 设置特征映射层
        self.text_encoder.set_feat_map(self.feat_map)
        
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
            self.level_embed = jt.zeros((num_feature_levels, hidden_dim), dtype=jt.float32)
            nn.init.gauss_(self.level_embed, 0, 1)
        else:
            self.level_embed = None
        
        # Position Embedding (use PositionEmbeddingSineHW with temperature=20 like PyTorch)
        self.position_embedding = PositionEmbeddingSineHW(
            num_pos_feats=hidden_dim // 2,
            temperatureH=20,
            temperatureW=20,
            normalize=True
        )
        
        # ============================================================
        # Query 初始化
        # ============================================================
        # 可学习的目标嵌入
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        nn.init.gauss_(self.tgt_embed.weight, 0, 1)
        
        # Label Encoder (for DN training/inference)
        # Matching weight shape (2001, 256)
        self.label_enc = nn.Embedding(2001, hidden_dim)
        
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
            def __init__(self_inner):
                super().__init__()
                self_inner.d_model = d_model
                self_inner.nhead = nhead
                self_inner.num_decoder_layers = num_decoder_layers
                
                # Encoder
                encoder_layer = DeformableTransformerEncoderLayer(
                    d_model=d_model,
                    d_ffn=dim_feedforward,
                    dropout=dropout,
                    n_levels=num_feature_levels,
                    n_heads=nhead,
                    n_points=4,
                )
                text_enhance_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead // 2,  # Fixed: nhead -> nhead // 2 to match PyTorch
                    dim_feedforward=1024, # Fixed: 2048 -> 1024
                    dropout=dropout,
                )
                feature_fusion_layer = BiAttentionBlock(
                    v_dim=d_model,
                    l_dim=d_model,
                    embed_dim=1024, # Fixed: d_model (256) -> 1024
                    num_heads=nhead,
                    dropout=dropout,
                    drop_path=0.1,  # Match PyTorch fusion_droppath
                )
                
                self_inner.encoder = TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=num_encoder_layers,
                    d_model=d_model,
                    text_enhance_layer=text_enhance_layer,
                    feature_fusion_layer=feature_fusion_layer,
                )
                
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
                self_inner.decoder = TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=num_decoder_layers,
                    norm=nn.LayerNorm(d_model),
                    return_intermediate=True,
                    d_model=d_model,
                    query_dim=4,
                    num_feature_levels=num_feature_levels,
                )
                
                self_inner.bbox_embed = None
                self_inner.class_embed = None
        
        return SimpleTransformer()
    
    def _reset_parameters(self):
        """初始化参数"""
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
    def get_valid_ratio(self, mask):
        """计算有效区域比例"""
        _, H, W = mask.shape
        valid_H = jt.sum(jt.logical_not(mask[:, :, 0]), dim=1)
        valid_W = jt.sum(jt.logical_not(mask[:, 0, :]), dim=1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = jt.stack([valid_ratio_w, valid_ratio_h], dim=-1)
        return valid_ratio
    
    def set_tokenizer(self, tokenizer):
        """设置 tokenizer"""
        self.tokenizer = tokenizer
    
    def encode_text(self, captions: List[str]) -> Dict:
        """
        使用 BERT 编码文本

        Args:
            captions: 文本列表

        Returns:
            text_dict: 文本特征字典
        """
        # 使用 BERT 编码器
        text_dict = self.text_encoder(captions, sub_sentence_present=self.sub_sentence_present)

        # DEBUG: Check shape after text_encoder
        # print(f"After text_encoder: encoded_text shape={text_dict['encoded_text'].shape}, hidden_dim={self.hidden_dim}")

        return text_dict

    def encode_image_projection(self, samples):
        """
        Extract and project vision features from image for caching (backbone + projection only).
        This allows skipping the expensive backbone computation while preserving text-vision fusion.

        Args:
            samples: Input image tensor [bs, 3, H, W], [3, H, W], or NestedTensor

        Returns:
            projected_features: Dict containing projected features for encoder
        """
        # ============================================================
        # 1. 处理输入
        # ============================================================
        if isinstance(samples, (list, jt.Var)):
            # 简单处理：假设是 batch 的图像张量
            if isinstance(samples, list):
                samples = jt.stack(samples)

            # Handle both single image [C, H, W] and batch [bs, C, H, W]
            if samples.ndim == 3:
                samples = samples.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

            # 创建掩码（全 False，即全部有效）
            bs, _, H, W = samples.shape
            masks = [jt.zeros((H, W), dtype=jt.bool) for _ in range(bs)]
        else:
            # NestedTensor 格式
            samples, masks = samples.decompose()
            bs = samples.shape[0]

        # ============================================================
        # 2. 提取视觉特征
        # ============================================================
        # 获取 backbone 通道数（用于生成正确维度的特征）
        if hasattr(self, 'input_proj') and len(self.input_proj) > 0:
            # 从 input_proj 推断 backbone 通道数
            backbone_channels = []
            for proj in self.input_proj:
                if hasattr(proj[0], 'in_channels'):
                    backbone_channels.append(proj[0].in_channels)
                else:
                    backbone_channels.append(self.hidden_dim)
        else:
            backbone_channels = [192, 384, 768, 768]

        if self.backbone is not None:
            # features, poss = self.backbone(samples)
            features = self.backbone(samples)
            if isinstance(features, tuple) and len(features) == 2:
                features, poss = features
            else:
                poss = None
        else:
            # 使用占位符特征（用于测试，无 backbone）
            # backbone_channels = [192, 384, 768, 768]
            # 前 3 层来自 backbone 的多尺度输出，第 4 层是额外下采样
            # 但是由于没有 backbone，我们需要生成所有 4 层的特征
            features = []
            poss = []
            h, w = H // 4, W // 4  # 初始下采样 4x（Swin Stage 1 输出）

            # 生成所有层的特征
            for i in range(self.num_feature_levels):
                if i < len(backbone_channels):
                    in_ch = backbone_channels[i]
                else:
                    in_ch = backbone_channels[-1]

                feat = jt.randn(bs, in_ch, h, w) * 0.1
                pos = jt.zeros(bs, self.hidden_dim, h, w, dtype=jt.float32)
                features.append((feat, jt.zeros((bs, h, w), dtype=jt.bool)))
                poss.append(pos)

                # 下采样 2x（除了最后一层）
                if i < self.num_feature_levels - 1:
                    h, w = max(1, h // 2), max(1, w // 2)

        # ============================================================
        # 3. 投影和展平特征
        # ============================================================
        srcs = []
        masks_flat = []

        # 确定有多少层直接来自 backbone（或占位符）
        num_backbone_levels = min(len(features), len(backbone_channels), self.num_feature_levels)

        # 处理来自 backbone 的特征层（使用 1x1 卷积投影）
        for l in range(num_backbone_levels):
            feat = features[l]
            if hasattr(feat, 'decompose'):
                src, mask = feat.decompose()
            elif isinstance(feat, tuple):
                src, mask = feat
            else:
                src = feat
                mask = jt.zeros((bs, src.shape[2], src.shape[3]), dtype=jt.bool)

            if l < len(self.input_proj):
                srcs.append(self.input_proj[l](src))
                masks_flat.append(mask)

        # 处理额外的下采样层（使用 3x3 stride=2 卷积）
        if self.num_feature_levels > len(srcs) and len(self.input_proj) > len(srcs):
            # 获取最后一层的原始特征（用于下采样）
            if len(features) > num_backbone_levels:
                # 如果有更多特征，使用它们
                feat = features[num_backbone_levels]
                if hasattr(feat, 'decompose'):
                    last_feat, _ = feat.decompose()
                elif isinstance(feat, tuple):
                    last_feat = feat[0]
                else:
                    last_feat = feat

            else:
                # features is a dict or list
                if isinstance(features, dict):
                    feat = features[len(features)-1]
                else:
                    feat = features[-1]

                if hasattr(feat, 'decompose'):
                    last_feat, _ = feat.decompose()
                elif isinstance(feat, tuple):
                    last_feat = feat[0]
                else:
                    last_feat = feat

            for l in range(len(srcs), self.num_feature_levels):
                if l < len(self.input_proj):
                    if l == num_backbone_levels:
                        # 从原始特征下采样
                        src = self.input_proj[l](last_feat)
                    else:
                        # 从上一个投影结果下采样（注意这里的 input_proj 可能是下采样层）
                        src = self.input_proj[l](srcs[-1])

                    # 创建对应的 mask
                    if len(masks_flat) > 0:
                        m = masks_flat[-1]
                        mask = nn.interpolate(m.unsqueeze(1).float(), size=src.shape[-2:]).squeeze(1).bool()
                    else:
                        mask = jt.zeros((bs, src.shape[2], src.shape[3]), dtype=jt.bool)

                    srcs.append(src)
                    masks_flat.append(mask)

        # 展平特征
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_ratios_list = []

        for lvl, (src, mask) in enumerate(zip(srcs, masks_flat)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))

            # Calculate valid_ratios before flattening
            valid_ratios_list.append(self.get_valid_ratio(mask))

            # Generate position embedding
            pos = self.position_embedding(src, mask)

            src = src.flatten(2).transpose(1, 2)  # [bs, h*w, c]
            mask = mask.flatten(1)  # [bs, h*w]
            pos = pos.flatten(2).transpose(1, 2) # [bs, h*w, c]

            # 位置编码
            pos_embed = pos
            if self.level_embed is not None:
                pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(pos_embed)

        src_flatten = jt.concat(src_flatten, dim=1).float32()
        mask_flatten = jt.concat(mask_flatten, dim=1)
        lvl_pos_embed_flatten = jt.concat(lvl_pos_embed_flatten, dim=1).float32()
        spatial_shapes = jt.array(spatial_shapes).int64()
        level_start_index = jt.concat([
            jt.zeros((1,), dtype=jt.int64),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
        ])
        # valid_ratios = jt.stack([jt.ones((bs, 2), dtype=jt.float32) for _ in range(len(spatial_shapes))], dim=1)
        valid_ratios = jt.stack(valid_ratios_list, dim=1).float32()

        # Return projected features (encoder will be run later with real text)
        projected_features = {
            'src_flatten': src_flatten,
            'mask_flatten': mask_flatten,
            'lvl_pos_embed_flatten': lvl_pos_embed_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
        }

        return projected_features
    
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        Generate proposals from encoder output for two-stage selection.
        
        Args:
            memory: [bs, n_tokens, d_model]
            memory_padding_mask: [bs, n_tokens]
            spatial_shapes: [n_levels, 2]
            
        Returns:
            output_proposals: [bs, n_tokens, 4] (cx, cy, w, h) in normalized coordinates
            output_proposals_valid: [bs, n_tokens] mask
        """
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes.numpy()):
            H, W = int(H), int(W)
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(N_, H, W, 1)
            valid_H = jt.sum(jt.logical_not(mask_flatten_[:, :, 0, 0]), dim=1)
            valid_W = jt.sum(jt.logical_not(mask_flatten_[:, 0, :, 0]), dim=1)

            grid_y, grid_x = jt.meshgrid(
                jt.linspace(0, H - 1, H).float32(),
                jt.linspace(0, W - 1, W).float32()
            )
            grid = jt.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1) # [H, W, 2]

            scale = jt.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], dim=1).view(N_, 1, 1, 2)
            
            # Debug: Check scale
            # print(f"Lvl {lvl} scale min: {scale.min().item()}, max: {scale.max().item()}")
            scale = jt.clamp(scale, 1e-6, 1e6) # Prevent division by zero

            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = jt.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = jt.concat((grid, wh), dim=-1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
            
        output_proposals = jt.concat(proposals, dim=1)
        
        # DEBUG: proposals before inverse sigmoid
        # print(f"proposals before inv_sigmoid: shape={output_proposals.shape}")

        # 使用 keepdims=True 以匹配 PyTorch 实现 (keepdim=True)
        # Jittor 的 all() 不支持 keepdims，手动添加维度
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1).unsqueeze(-1)
        
        # Clamp to avoid log errors
        output_proposals = jt.clamp(output_proposals, 1e-4, 1-1e-4)
        
        output_proposals = jt.log(output_proposals / (1 - output_proposals)) # inverse sigmoid
        
        # Debug after inverse sigmoid
        # print(f"proposals after inv_sigmoid min: {output_proposals.min().item()}, max: {output_proposals.max().item()}")

        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # output_proposals_valid 已经是 [bs, n_tokens, 1]，需要 squeeze 以匹配 PyTorch 的 keepdim=True 行为
        output_proposals_valid_squeezed = output_proposals_valid.squeeze(-1) if output_proposals_valid.ndim == 3 else output_proposals_valid
        output_proposals = output_proposals.masked_fill(jt.logical_not(output_proposals_valid_squeezed).unsqueeze(-1), float('inf'))
        
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
        output_memory = output_memory.masked_fill(jt.logical_not(output_proposals_valid_squeezed).unsqueeze(-1), 0.0)

        return output_memory, output_proposals, output_proposals_valid

    def execute(
        self,
        samples: jt.Var,
        targets: Optional[List[Dict]] = None,
        text_dict: Optional[Dict] = None,
        captions: Optional[List[str]] = None,
        vision_features: Optional[Dict] = None,
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
            captions: 文本提示列表（可选，用于自动编码）
            vision_features: 预计算的视觉特征字典（可选，用于加速推理）

        Returns:
            outputs: 包含 pred_logits 和 pred_boxes 的字典
        """
        # Safety assertions to prevent placeholder logic during inference
        assert self.backbone is not None or vision_features is not None, "Backbone must be provided or vision_features must be cached for inference"
        assert text_dict is not None or captions is not None, "Text input (text_dict or captions) must be provided for inference"

        # ============================================================
        # 1. 处理输入
        # ============================================================
        if vision_features is not None:
            # 使用缓存的投影特征，直接提取 batch size
            bs = vision_features['src_flatten'].shape[0]
        else:
            if isinstance(samples, (list, jt.Var)):
                # 简单处理：假设是 batch 的图像张量
                if isinstance(samples, list):
                    samples = jt.stack(samples)

                # Handle both 3D [C, H, W] and 4D [bs, C, H, W] tensors
                if samples.ndim == 4:
                    bs, _, H, W = samples.shape
                elif samples.ndim == 3:
                    # Single image case [C, H, W]
                    bs = 1
                    _, H, W = samples.shape
                else:
                    raise ValueError(f"Unexpected tensor shape: {samples.shape}")

                # 创建掩码（全 False，即全部有效）
                masks = [jt.zeros((H, W), dtype=jt.bool) for _ in range(bs)]
            else:
                # NestedTensor 格式
                samples, masks = samples.decompose()
                bs = samples.shape[0]

        # ============================================================
        # 2. 处理文本特征
        # ============================================================
        # 如果提供了 captions，自动编码
        if captions is not None and text_dict is None:
            text_dict = self.encode_text(captions)

        if text_dict is None:
            # 创建空的文本特征（用于推理时）
            text_dict = {
                "encoded_text": jt.zeros((bs, self.max_text_len, self.hidden_dim), dtype=jt.float32),
                "text_token_mask": jt.ones((bs, self.max_text_len), dtype=jt.bool),
                "position_ids": jt.arange(self.max_text_len, dtype=jt.int64).unsqueeze(0).repeat(bs, 1),
                "text_self_attention_masks": jt.ones((bs, self.max_text_len, self.max_text_len), dtype=jt.bool),
            }

        # 确保文本特征维度正确
        # print(f"Before dimension check: encoded_text shape={text_dict['encoded_text'].shape}, hidden_dim={self.hidden_dim}")
        if text_dict["encoded_text"].shape[-1] != self.hidden_dim:
            # print(f"Applying feat_map because {text_dict['encoded_text'].shape[-1]} != {self.hidden_dim}")
            text_dict["encoded_text"] = self.feat_map(text_dict["encoded_text"])
        else:
            pass  # feat_map not needed, shape is correct

        # ============================================================
        # 3. 提取视觉特征 (或使用缓存)
        # ============================================================
        if vision_features is not None:
            # 使用缓存的投影特征 - 运行 encoder 获取完整特征
            src_flatten = vision_features['src_flatten']
            mask_flatten = vision_features['mask_flatten']
            lvl_pos_embed_flatten = vision_features['lvl_pos_embed_flatten']
            spatial_shapes = vision_features['spatial_shapes']
            level_start_index = vision_features['level_start_index']
            valid_ratios = vision_features['valid_ratios']

            # 运行 encoder 以获取增强的文本特征和视觉特征
            memory, memory_text = self.transformer.encoder(
                src=src_flatten,  # [bs, sum(hw), c] - cached projected features
                pos=lvl_pos_embed_flatten,  # [bs, sum(hw), c]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                memory_text=text_dict["encoded_text"],
                text_attention_mask=jt.logical_not(text_dict["text_token_mask"]),
                position_ids=text_dict["position_ids"],
                text_self_attention_masks=text_dict["text_self_attention_masks"],
            )
            text_dict["encoded_text"] = memory_text
        else:
            # 原始视觉特征提取逻辑
            # 获取 backbone 通道数（用于生成正确维度的特征）
            if hasattr(self, 'input_proj') and len(self.input_proj) > 0:
                # 从 input_proj 推断 backbone 通道数
                backbone_channels = []
                for proj in self.input_proj:
                    if hasattr(proj[0], 'in_channels'):
                        backbone_channels.append(proj[0].in_channels)
                    else:
                        backbone_channels.append(self.hidden_dim)
            else:
                backbone_channels = [192, 384, 768, 768]

            if self.backbone is not None:
                # features, poss = self.backbone(samples)
                features = self.backbone(samples)
                if isinstance(features, tuple) and len(features) == 2:
                    features, poss = features
                else:
                    poss = None
            else:
                # 使用占位符特征（用于测试，无 backbone）
                # backbone_channels = [192, 384, 768, 768]
                # 前 3 层来自 backbone 的多尺度输出，第 4 层是额外下采样
                # 但是由于没有 backbone，我们需要生成所有 4 层的特征
                features = []
                poss = []
                h, w = H // 4, W // 4  # 初始下采样 4x（Swin Stage 1 输出）

                # 生成所有层的特征
                for i in range(self.num_feature_levels):
                    if i < len(backbone_channels):
                        in_ch = backbone_channels[i]
                    else:
                        in_ch = backbone_channels[-1]

                    feat = jt.randn(bs, in_ch, h, w) * 0.1
                    pos = jt.zeros(bs, self.hidden_dim, h, w, dtype=jt.float32)
                    features.append((feat, jt.zeros((bs, h, w), dtype=jt.bool)))
                    poss.append(pos)

                    # 下采样 2x（除了最后一层）
                    if i < self.num_feature_levels - 1:
                        h, w = max(1, h // 2), max(1, w // 2)

            # ============================================================
            # 4. 投影和展平特征
            # ============================================================
            srcs = []
            masks_flat = []

            # 确定有多少层直接来自 backbone（或占位符）
            num_backbone_levels = min(len(features), len(backbone_channels), self.num_feature_levels)

            # 处理来自 backbone 的特征层（使用 1x1 卷积投影）
            for l in range(num_backbone_levels):
                feat = features[l]
                if hasattr(feat, 'decompose'):
                    src, mask = feat.decompose()
                elif isinstance(feat, tuple):
                    src, mask = feat
                else:
                    src = feat
                    mask = jt.zeros((bs, src.shape[2], src.shape[3]), dtype=jt.bool)

                if l < len(self.input_proj):
                    srcs.append(self.input_proj[l](src))
                    masks_flat.append(mask)

            # 处理额外的下采样层（使用 3x3 stride=2 卷积）
            if self.num_feature_levels > len(srcs) and len(self.input_proj) > len(srcs):
                # 获取最后一层的原始特征（用于下采样）
                if len(features) > num_backbone_levels:
                    # 如果有更多特征，使用它们
                    feat = features[num_backbone_levels]
                    if hasattr(feat, 'decompose'):
                        last_feat, _ = feat.decompose()
                    elif isinstance(feat, tuple):
                        last_feat = feat[0]
                    else:
                        last_feat = feat
                else:
                    # features is a dict or list
                    if isinstance(features, dict):
                        feat = features[len(features)-1]
                    else:
                        feat = features[-1]

                    if hasattr(feat, 'decompose'):
                        last_feat, _ = feat.decompose()
                    elif isinstance(feat, tuple):
                        last_feat = feat[0]
                    else:
                        last_feat = feat

                for l in range(len(srcs), self.num_feature_levels):
                    if l < len(self.input_proj):
                        if l == num_backbone_levels:
                            # 从原始特征下采样
                            src = self.input_proj[l](last_feat)
                        else:
                            # 从上一个投影结果下采样（注意这里的 input_proj 可能是下采样层）
                            src = self.input_proj[l](srcs[-1])

                        # 创建对应的 mask
                        if len(masks_flat) > 0:
                            m = masks_flat[-1]
                            mask = nn.interpolate(m.unsqueeze(1).float(), size=src.shape[-2:]).squeeze(1).bool()
                        else:
                            mask = jt.zeros((bs, src.shape[2], src.shape[3]), dtype=jt.bool)

                        srcs.append(src)
                        masks_flat.append(mask)

            # 展平特征
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            valid_ratios_list = []

            for lvl, (src, mask) in enumerate(zip(srcs, masks_flat)):
                bs, c, h, w = src.shape
                spatial_shapes.append((h, w))

                # Calculate valid_ratios before flattening
                valid_ratios_list.append(self.get_valid_ratio(mask))

                # Generate position embedding
                pos = self.position_embedding(src, mask)

                src = src.flatten(2).transpose(1, 2)  # [bs, h*w, c]
                mask = mask.flatten(1)  # [bs, h*w]
                pos = pos.flatten(2).transpose(1, 2) # [bs, h*w, c]

                # 位置编码
                pos_embed = pos
                if self.level_embed is not None:
                    pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

                src_flatten.append(src)
                mask_flatten.append(mask)
                lvl_pos_embed_flatten.append(pos_embed)

            src_flatten = jt.concat(src_flatten, dim=1).float32()
            mask_flatten = jt.concat(mask_flatten, dim=1)
            lvl_pos_embed_flatten = jt.concat(lvl_pos_embed_flatten, dim=1).float32()
            spatial_shapes = jt.array(spatial_shapes).int64()
            level_start_index = jt.concat([
                jt.zeros((1,), dtype=jt.int64),
                (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]
            ])
            # valid_ratios = jt.stack([jt.ones((bs, 2), dtype=jt.float32) for _ in range(len(spatial_shapes))], dim=1)
            valid_ratios = jt.stack(valid_ratios_list, dim=1).float32()

            # ============================================================
            # 5. Encoder
            # ============================================================
            # print("Entering Encoder...")
            # print(f"src_flatten dtype: {src_flatten.dtype}")
            # print(f"src_flatten stats: min={src_flatten.min().item()}, max={src_flatten.max().item()}, mean={src_flatten.mean().item()}")
            # if jt.any(jt.isnan(src_flatten)) or jt.any(jt.isinf(src_flatten)):
            #     print("WARNING: src_flatten contains NaN or Inf!")

            # print(f"lvl_pos_embed_flatten dtype: {lvl_pos_embed_flatten.dtype}")
            # print(f"spatial_shapes dtype: {spatial_shapes.dtype}")
            # print(f"valid_ratios dtype: {valid_ratios.dtype}")
            # print(f"valid_ratios values:\n{valid_ratios.numpy()}")
            # print(f"Input memory_text stats: min={float(text_dict['encoded_text'].min())}, max={float(text_dict['encoded_text'].max())}")

            # Ensure all inputs are float32
            src_flatten = src_flatten.float32()
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.float32()
            valid_ratios = valid_ratios.float32()
            text_dict["encoded_text"] = text_dict["encoded_text"].float32()

            # Note: encoder expects [bs, sum(hw), c] format (NOT transposed)
            # PyTorch: text_attention_mask=~text_dict["text_token_mask"]
            # We need to invert the mask: True means pad (ignore), False means valid
            memory, memory_text = self.transformer.encoder(
                src=src_flatten,  # [bs, sum(hw), c]
                pos=lvl_pos_embed_flatten,  # [bs, sum(hw), c]
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                memory_text=text_dict["encoded_text"],
                text_attention_mask=jt.logical_not(text_dict["text_token_mask"]),  # Invert mask like PyTorch
                position_ids=text_dict["position_ids"],
                text_self_attention_masks=text_dict["text_self_attention_masks"],
            )
            # print("Encoder finished.")

            # CRITICAL: Update text_dict with encoder-enhanced text features
            # This is done in PyTorch's Transformer.forward (line 279)
            # print(f"DEBUG before update: encoded_text min={text_dict['encoded_text'].min().item():.3f}, max={text_dict['encoded_text'].max().item():.3f}")
            # print(f"DEBUG memory_text: min={memory_text.min().item():.3f}, max={memory_text.max().item():.3f}")
            text_dict["encoded_text"] = memory_text
        
        # ============================================================
        # 6. Decoder
        # ============================================================
        # 初始化 query
        tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [nq, bs, hidden_dim]
        
        if self.refpoint_embed is not None:
            refpoint = self.refpoint_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [nq, bs, 4]
        else:
            refpoint = jt.zeros((self.num_queries, bs, 4), dtype=jt.float32)
        
        # Two-stage selection
        if self.two_stage_type != 'no':
            # memory is already [bs, n_tokens, d_model] from encoder
            output_memory = memory
            output_memory, output_proposals, proposals_valid = self.gen_encoder_output_proposals(output_memory, mask_flatten, spatial_shapes)
            
            if self.two_stage_type == 'standard':
                output_memory = self.enc_output(output_memory)
                output_memory = self.enc_output_norm(output_memory)
            
            enc_outputs_class = self.transformer.enc_out_class_embed(output_memory, text_dict)
            enc_outputs_coord_unact = self.transformer.enc_out_bbox_embed(output_memory) + output_proposals

            # Select top-k
            topk = self.num_queries
            class_prob = enc_outputs_class.max(-1)[0]  # [bs, n_tokens]

            # Mask invalid proposals
            # proposals_valid is [bs, n_tokens, 1] from keepdims, squeeze to [bs, n_tokens]
            proposals_valid_squeezed = proposals_valid.squeeze(-1) if len(proposals_valid.shape) == 3 else proposals_valid
            final_mask = mask_flatten | jt.logical_not(proposals_valid_squeezed)
            class_prob = class_prob.masked_fill(final_mask, -1e9)

            # Ensure class_prob is [bs, n_tokens]
            # If batch dimension was squeezed, restore it
            if class_prob.ndim == 1:
                class_prob = class_prob.unsqueeze(0)
            
            # Use jt.sort to get values and indices
            # jt.sort returns (values, indices)
            _, sorted_indices = jt.sort(class_prob, dim=1, descending=True)
            
            topk_proposals = sorted_indices[:, :topk] # [bs, topk]
            
            # Gather
            # output_memory: [bs, n_tokens, d_model]
            # topk_proposals: [bs, topk]
            # We need to gather along dim 1
            
            # Jittor gather implementation
            # index: [bs, topk, d_model]
            idx = topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
            tgt_gathered = jt.gather(output_memory, 1, idx)  # [bs, topk, d_model]
            
            # CRITICAL: Use tgt_embed.weight if embed_init_tgt is True (like PyTorch)
            if self.embed_init_tgt:
                # Use learnable tgt_embed instead of gathered output_memory
                tgt = self.tgt_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [nq, bs, hidden_dim]
            else:
                # Use gathered output_memory
                tgt = tgt_gathered.transpose(0, 1).detach()  # [topk, bs, d_model]
            
            # Gather refpoint
            # enc_outputs_coord_unact: [bs, n_tokens, 4]
            
            # Clamp to avoid infs poisoning the reference points
            # enc_outputs_coord_unact = jt.clamp(enc_outputs_coord_unact, -10, 10)
            idx_coord = topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            refpoint_unact = jt.gather(enc_outputs_coord_unact, 1, idx_coord) # [bs, topk, 4]
            # print(f"refpoint_unact gathered stats: min={refpoint_unact.min().item()}, max={refpoint_unact.max().item()}")
            # print(f"refpoint_unact gathered first 5: {refpoint_unact[0, :5].numpy()}")
            
            # refpoint = jt.sigmoid(refpoint_unact) # [bs, topk, 4]
            # print(f"refpoint sigmoid stats: min={refpoint.min().item()}, max={refpoint.max().item()}")
            
            # refpoint = refpoint.transpose(0, 1) # [topk, bs, 4]
            
            # Pass raw logits to decoder (it expects unsigmoid)
            refpoint = refpoint_unact.transpose(0, 1)  # [topk, bs, 4]

            # Detach
            tgt = tgt.detach()
            refpoint = refpoint.detach()

        # print("Entering Decoder...")
        # print(f"memory_text shape: {memory_text.shape}")
        # print(f"text_attention_mask shape: {text_dict['text_token_mask'].shape}")

        # Decoder forward
        # Note: decoder expects memory in [hw, bs, c] format, but encoder returns [bs, hw, c]
        # PyTorch: text_attention_mask=~text_dict["text_token_mask"]
        hs, references = self.transformer.decoder(
            tgt=tgt,
            memory=memory.transpose(0, 1),  # [bs, hw, c] -> [hw, bs, c]
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),  # [bs, hw, c] -> [hw, bs, c]
            refpoints_unsigmoid=refpoint,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=jt.logical_not(text_dict["text_token_mask"]),  # Invert mask like PyTorch
        )
        # print("Decoder finished.")
        
        # DEBUG
        # print(f"hs type: {type(hs)}")
        # print(f"hs len: {len(hs)}")
        # print(f"hs[-1] shape: {hs[-1].shape}")
        # print(f"references type: {type(references)}")
        # print(f"references len: {len(references)}")
        # print(f"references[-1] shape: {references[-1].shape}")

        # ============================================================
        # 7. 检测头输出
        # ============================================================
        # 与 PyTorch 一致：在主模型中重新计算边界框输出
        # 使用 norm 后的 hs 和 reference[:-1] (不包括最后一个参考点)
        
        # deformable-detr-like anchor update
        # 重新计算边界框输出，使用 norm 后的 hs
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(references[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = jt.sigmoid(layer_outputs_unsig)
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = jt.stack(outputs_coord_list)
        
        # 分类输出 - 使用所有层的 hs (与 PyTorch 一致)
        outputs_class = jt.stack([
            layer_cls_embed(layer_hs, text_dict)
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        ])
        
        # DEBUG: Check logits distribution
        # print(f"DEBUG outputs_class: min={outputs_class.min().item():.3f}, max={outputs_class.max().item():.3f}")
        # print(f"DEBUG text_dict encoded_text: min={text_dict['encoded_text'].min().item():.3f}, max={text_dict['encoded_text'].max().item():.3f}")
        # print(f"DEBUG hs[-1]: min={hs[-1].min().item():.3f}, max={hs[-1].max().item():.3f}")
        # print(f"DEBUG text_token_mask: shape={text_dict['text_token_mask'].shape}, sum={text_dict['text_token_mask'].sum().item()}")
        
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }
        
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
        "position_ids": jt.arange(50, dtype=jt.int64).unsqueeze(0).repeat(bs, 1),
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
