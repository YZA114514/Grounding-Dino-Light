# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# DINO Head Module (Member A)
# ------------------------------------------------------------------------
"""
DINO Head 模块说明：

DINO Head 是 Grounding DINO 的检测头，负责将 Decoder 输出转换为最终的检测结果。

主要组件：
1. ContrastiveEmbed: 对比嵌入分类头（用于开放词汇检测）
2. MLP: 边界框回归头
3. 辅助损失支持：返回所有中间层的预测结果

关键设计：
- 对比嵌入：通过计算视觉特征与文本特征的相似度来进行分类
  这使得模型可以检测训练时未见过的类别（开放词汇检测）
- 边界框回归：使用MLP预测边界框的(cx, cy, w, h)
- 迭代式细化：每层的边界框预测是相对于参考点的偏移量
"""

import math
from typing import Dict, List, Optional

import jittor as jt
from jittor import nn


class MLP(nn.Module):
    """
    简单的多层感知机（也称为FFN）
    
    用于边界框回归：将d_model维特征映射到4维边界框坐标
    
    结构：Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # Jittor 的 ModuleList 需要传入列表而非生成器
        layers = [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        self.layers = nn.ModuleList(layers)

    def execute(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [..., input_dim]
            
        Returns:
            output: [..., output_dim]
        """
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ContrastiveEmbed(nn.Module):
    """
    对比嵌入分类头
    
    核心思想：
    不使用传统的线性分类器，而是计算视觉特征与文本特征的相似度。
    这使得模型可以检测任意文本描述的对象（开放词汇检测）。
    
    工作流程：
    1. 接收视觉特征 x: [bs, nq, d_model]
    2. 接收文本特征 y: [bs, n_text, d_model]
    3. 计算相似度矩阵: x @ y^T -> [bs, nq, n_text]
    4. 对padding位置填充-inf
    5. 填充到固定长度 max_text_len
    
    输出解释：
    - 每个query对每个文本token的匹配分数
    - 高分数表示该query检测到了对应文本描述的对象
    """
    
    def __init__(self, max_text_len=256):
        """
        Args:
            max_text_len: 最大文本长度，用于padding输出
        """
        super().__init__()
        self.max_text_len = max_text_len

    def execute(self, x, text_dict):
        """
        前向传播
        
        Args:
            x: 视觉特征 [bs, nq, d_model]
            text_dict: 文本字典，包含：
                - 'encoded_text': 编码后的文本特征 [bs, n_text, d_model]
                - 'text_token_mask': 文本token掩码 [bs, n_text]
                  True表示有效token，False表示padding
                  
        Returns:
            res: 相似度分数 [bs, nq, max_text_len]
        """
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]  # [bs, n_text, d_model]
        text_token_mask = text_dict["text_token_mask"]  # [bs, n_text]

        # 计算相似度矩阵
        # x: [bs, nq, d_model], y: [bs, n_text, d_model]
        # res: [bs, nq, n_text]
        
        res = jt.matmul(x, y.transpose(-1, -2))
        
        # 对padding位置填充-inf
        # text_token_mask: [bs, n_text] -> [bs, 1, n_text]
        # True 表示有效 token，False 表示 padding
        # 我们需要掩码 padding 位置，所以取反
        mask = jt.logical_not(text_token_mask).unsqueeze(1)  # True表示padding
        
        # DEBUG
        # print(f"ContrastiveEmbed: x shape={x.shape}, y shape={y.shape}")
        # print(f"ContrastiveEmbed: x min={x.min().item():.3f}, max={x.max().item():.3f}")
        # print(f"ContrastiveEmbed: y min={y.min().item():.3f}, max={y.max().item():.3f}")
        # print(f"ContrastiveEmbed: res pre-mask min={res.min().item():.3f}, max={res.max().item():.3f}")
        # print(f"ContrastiveEmbed: mask sum (padding count)={mask.sum().item()}")
        
        # Use a large negative number instead of -inf to avoid NaN issues in sigmoid
        res = jt.where(mask, jt.full_like(res, -1e9), res)
        
        # print(f"ContrastiveEmbed: res post-mask min={res.min().item():.3f}, max={res.max().item():.3f}")

        # 填充到固定长度
        bs, nq, n_text = res.shape
        new_res = jt.full((bs, nq, self.max_text_len), float("-inf"))
        new_res[:, :, :n_text] = res

        return new_res


class DINOHead(nn.Module):
    """
    DINO 检测头
    
    功能：
    1. 分类：使用对比嵌入计算query与文本的匹配分数
    2. 边界框回归：预测边界框坐标
    3. 支持辅助损失：为每个decoder层提供预测头
    
    输出格式：
    - pred_logits: [bs, nq, max_text_len] 分类分数
    - pred_boxes: [bs, nq, 4] 边界框坐标 (cx, cy, w, h)，归一化到[0,1]
    """
    
    def __init__(
        self,
        d_model=256,
        num_decoder_layers=6,
        max_text_len=256,
        dec_pred_bbox_embed_share=True,
    ):
        """
        Args:
            d_model: 模型维度
            num_decoder_layers: decoder层数
            max_text_len: 最大文本长度
            dec_pred_bbox_embed_share: 是否在各decoder层之间共享边界框回归头
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.max_text_len = max_text_len
        
        # 分类头：对比嵌入
        _class_embed = ContrastiveEmbed(max_text_len=max_text_len)
        
        # 边界框回归头：3层MLP
        _bbox_embed = MLP(d_model, d_model, 4, 3)
        # 初始化最后一层为零（预测小的偏移量）
        nn.init.constant_(_bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias, 0)
        
        # 为每个decoder层创建预测头
        if dec_pred_bbox_embed_share:
            # 共享边界框回归头
            bbox_embeds = [_bbox_embed for _ in range(num_decoder_layers)]
            self.bbox_embed = nn.ModuleList(bbox_embeds)
        else:
            # 每层独立的边界框回归头
            import copy
            bbox_embeds = [copy.deepcopy(_bbox_embed) for _ in range(num_decoder_layers)]
            self.bbox_embed = nn.ModuleList(bbox_embeds)
        
        # 分类头（对比嵌入）对所有层共享
        class_embeds = [_class_embed for _ in range(num_decoder_layers)]
        self.class_embed = nn.ModuleList(class_embeds)

    def execute(
        self,
        hs: List[jt.Var],
        references: List[jt.Var],
        text_dict: Dict,
    ):
        """
        前向传播
        
        Args:
            hs: decoder各层输出 [n_dec, bs, nq, d_model] 或 列表形式
            references: 各层参考点 [n_dec+1, bs, nq, 4] 或 列表形式
            text_dict: 文本字典
            
        Returns:
            outputs: 字典，包含：
                - 'pred_logits': [bs, nq, max_text_len]
                - 'pred_boxes': [bs, nq, 4]
                - 'aux_outputs': 辅助输出列表（可选）
        """
        # 处理输入格式
        if isinstance(hs, list):
            # 列表格式：每个元素是 [bs, nq, d_model]
            pass
        else:
            # 张量格式：[n_dec, bs, nq, d_model]
            hs = [hs[i] for i in range(hs.shape[0])]
        
        if isinstance(references, list):
            pass
        else:
            references = [references[i] for i in range(references.shape[0])]
        
        # 计算各层输出
        outputs_class_list = []
        outputs_coord_list = []
        
        for dec_lid, (layer_hs, layer_ref_sig) in enumerate(
            zip(hs, references[:-1])  # references比hs多一个（初始参考点）
        ):
            # 分类：对比嵌入
            layer_cls = self.class_embed[dec_lid](layer_hs, text_dict)
            outputs_class_list.append(layer_cls)
            
            # 边界框回归
            layer_delta_unsig = self.bbox_embed[dec_lid](layer_hs)
            # 迭代式细化：偏移量 + 参考点
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs = jt.sigmoid(layer_outputs_unsig)
            outputs_coord_list.append(layer_outputs)
        
        # 堆叠所有层的输出
        outputs_class = jt.stack(outputs_class_list)  # [n_dec, bs, nq, max_text_len]
        outputs_coord = jt.stack(outputs_coord_list)  # [n_dec, bs, nq, 4]
        
        # 最终输出（最后一层）
        out = {
            "pred_logits": outputs_class[-1],  # [bs, nq, max_text_len]
            "pred_boxes": outputs_coord[-1],   # [bs, nq, 4]
        }
        
        # 辅助输出（用于辅助损失）
        aux_outputs = []
        for i in range(len(outputs_class_list) - 1):
            aux_outputs.append({
                "pred_logits": outputs_class_list[i],
                "pred_boxes": outputs_coord_list[i],
            })
        
        if aux_outputs:
            out["aux_outputs"] = aux_outputs
        
        return out


def inverse_sigmoid(x, eps=1e-5):
    """
    逆sigmoid函数
    
    将[0,1]范围的值转换回(-inf, +inf)
    用于迭代式边界框细化
    """
    x = jt.clamp(x, min_v=eps, max_v=1 - eps)
    return jt.log(x / (1 - x))


class SimpleHead(nn.Module):
    """
    简化版检测头（用于调试）
    
    直接使用线性层进行分类，不使用对比嵌入
    """
    
    def __init__(
        self,
        d_model=256,
        num_classes=256,
    ):
        super().__init__()
        
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        
        # 初始化
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)

    def execute(self, hs, references=None):
        """
        Args:
            hs: decoder输出 [bs, nq, d_model]
            references: 参考点 [bs, nq, 4]（可选）
            
        Returns:
            pred_logits: [bs, nq, num_classes]
            pred_boxes: [bs, nq, 4]
        """
        pred_logits = self.class_embed(hs)
        pred_boxes_delta = self.bbox_embed(hs)
        
        if references is not None:
            pred_boxes = jt.sigmoid(pred_boxes_delta + inverse_sigmoid(references))
        else:
            pred_boxes = jt.sigmoid(pred_boxes_delta)
        
        return pred_logits, pred_boxes


# ============================================================
# 辅助函数
# ============================================================

def build_dino_head(
    d_model=256,
    num_decoder_layers=6,
    max_text_len=256,
    dec_pred_bbox_embed_share=True,
):
    """
    构建DINO检测头
    """
    return DINOHead(
        d_model=d_model,
        num_decoder_layers=num_decoder_layers,
        max_text_len=max_text_len,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 测试MLP
    print("Testing MLP...")
    mlp = MLP(256, 256, 4, 3)
    x = jt.randn(2, 100, 256)
    out = mlp(x)
    print(f"MLP input: {x.shape}, output: {out.shape}")
    assert out.shape == (2, 100, 4)
    print("MLP test passed!")
    
    # 测试ContrastiveEmbed
    print("\nTesting ContrastiveEmbed...")
    ce = ContrastiveEmbed(max_text_len=256)
    x = jt.randn(2, 100, 256)  # [bs, nq, d_model]
    text_dict = {
        "encoded_text": jt.randn(2, 50, 256),  # [bs, n_text, d_model]
        "text_token_mask": jt.ones(2, 50).bool(),  # [bs, n_text]
    }
    out = ce(x, text_dict)
    print(f"ContrastiveEmbed input: {x.shape}, output: {out.shape}")
    assert out.shape == (2, 100, 256)
    print("ContrastiveEmbed test passed!")
    
    # 测试DINOHead
    print("\nTesting DINOHead...")
    head = DINOHead(
        d_model=256,
        num_decoder_layers=6,
        max_text_len=256,
    )
    hs = [jt.randn(2, 100, 256) for _ in range(6)]
    references = [jt.sigmoid(jt.randn(2, 100, 4)) for _ in range(7)]
    text_dict = {
        "encoded_text": jt.randn(2, 50, 256),
        "text_token_mask": jt.ones(2, 50).bool(),
    }
    out = head(hs, references, text_dict)
    print(f"DINOHead outputs:")
    print(f"  pred_logits: {out['pred_logits'].shape}")
    print(f"  pred_boxes: {out['pred_boxes'].shape}")
    if "aux_outputs" in out:
        print(f"  aux_outputs: {len(out['aux_outputs'])} layers")
    print("DINOHead test passed!")
    
    print("\nAll tests passed!")
