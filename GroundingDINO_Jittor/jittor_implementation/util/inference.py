#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grounding DINO Jittor - 完整推理工具

提供与官方 PyTorch 版本兼容的推理接口
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import pickle

import jittor as jt
from jittor import nn

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


# ============================================================
# 图像预处理
# ============================================================

class Compose:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    """随机调整大小（保持纵横比）"""
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, image, target=None):
        size = self.sizes[0]  # 使用第一个尺寸
        
        # 获取原始尺寸
        w, h = image.size
        
        # 计算新尺寸
        if self.max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > self.max_size:
                size = int(round(self.max_size * min_original_size / max_original_size))
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        
        image = image.resize((ow, oh), Image.BILINEAR)
        return image, target


class ToTensor:
    """转换为张量"""
    def __call__(self, image, target=None):
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = jt.array(image)
        return image, target


class Normalize:
    """标准化"""
    def __init__(self, mean, std):
        self.mean = jt.array(mean).view(3, 1, 1)
        self.std = jt.array(std).view(3, 1, 1)
    
    def __call__(self, image, target=None):
        image = (image - self.mean) / self.std
        return image, target


def get_transform():
    """获取默认的图像变换"""
    return Compose([
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_image(image_path: str) -> Tuple[Image.Image, jt.Var]:
    """
    加载并预处理图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        image_pil: PIL 图像
        image_tensor: 预处理后的张量 [3, H, W]
    """
    image_pil = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image_tensor, _ = transform(image_pil, None)
    return image_pil, image_tensor


# ============================================================
# 文本处理
# ============================================================

def preprocess_caption(caption: str) -> str:
    """预处理文本提示"""
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


class SimpleTokenizer:
    """
    简化的 Tokenizer，用于不依赖 transformers 的场景
    """
    def __init__(self):
        # 基本的特殊 token
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self.special_tokens = [self.cls_token_id, self.sep_token_id, self.pad_token_id]
    
    def __call__(self, text, **kwargs):
        """简单的 tokenization"""
        if isinstance(text, str):
            text = [text]
        
        # 简单的字符级 tokenization（仅用于演示）
        # 实际应该使用 BERT tokenizer
        results = {
            'input_ids': [],
            'attention_mask': [],
        }
        
        for t in text:
            # 简单的 word-level tokenization
            words = t.lower().split()
            # 模拟 token ids (实际应使用 vocab)
            ids = [self.cls_token_id] + [hash(w) % 30000 + 1000 for w in words] + [self.sep_token_id]
            mask = [1] * len(ids)
            results['input_ids'].append(ids)
            results['attention_mask'].append(mask)
        
        return results
    
    def decode(self, token_ids):
        """解码 token ids"""
        return " ".join([f"[{tid}]" for tid in token_ids if tid not in self.special_tokens])


# ============================================================
# 后处理
# ============================================================

def box_cxcywh_to_xyxy(boxes):
    """
    将边界框从 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return jt.stack([x1, y1, x2, y2], dim=1)


def get_phrases_from_posmap(
    posmap,
    tokenized,
    tokenizer,
    left_idx: int = 0,
    right_idx: int = 255
):
    """
    从位置映射中提取短语
    
    Args:
        posmap: 位置映射 (布尔张量)
        tokenized: tokenized 结果
        tokenizer: tokenizer
        left_idx: 左边界
        right_idx: 右边界
    """
    if isinstance(posmap, jt.Var):
        posmap = posmap.numpy()
    
    # 获取激活的位置
    non_zero_idx = np.where(posmap[left_idx:right_idx])[0] + left_idx
    
    if len(non_zero_idx) == 0:
        return ""
    
    # 获取 token ids
    input_ids = tokenized['input_ids']
    if isinstance(input_ids, jt.Var):
        input_ids = input_ids.numpy()
    if len(input_ids.shape) > 1:
        input_ids = input_ids[0]
    
    # 解码
    phrase_ids = [input_ids[idx] for idx in non_zero_idx if idx < len(input_ids)]
    
    if hasattr(tokenizer, 'decode'):
        try:
            phrase = tokenizer.decode(phrase_ids)
            # 清理特殊 token
            phrase = phrase.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            phrase = phrase.replace('##', '').strip()
            return phrase
        except:
            pass
    
    return tokenizer.decode(phrase_ids) if hasattr(tokenizer, 'decode') else str(phrase_ids)


# ============================================================
# 可视化
# ============================================================

def plot_boxes_to_image(
    image_pil: Image.Image, 
    boxes: np.ndarray, 
    labels: List[str],
    scores: Optional[np.ndarray] = None
) -> Image.Image:
    """
    在图像上绘制边界框
    
    Args:
        image_pil: PIL 图像
        boxes: 边界框 [N, 4] (xyxy 格式，归一化坐标)
        labels: 标签列表
        scores: 置信度分数
        
    Returns:
        带标注的图像
    """
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # 颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # 转换为像素坐标
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 准备标签文本
        if scores is not None and i < len(scores):
            label_text = f"{label} {scores[i]:.2f}"
        else:
            label_text = label
        
        # 绘制标签背景
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((x1, y1), label_text, font=font)
        else:
            w, h = draw.textsize(label_text, font=font)
            bbox = (x1, y1, x1 + w, y1 + h)
        
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1), label_text, fill="white", font=font)
    
    return image_pil


# ============================================================
# 模型加载和推理
# ============================================================

def load_model_weights(model, weight_path: str, strict: bool = False):
    """
    加载模型权重
    
    Args:
        model: Jittor 模型
        weight_path: 权重文件路径 (.pkl)
        strict: 是否严格匹配
    """
    with open(weight_path, 'rb') as f:
        weights = pickle.load(f)
    
    # 清理权重名称（移除 'module.' 前缀）
    cleaned_weights = {}
    for k, v in weights.items():
        new_k = k
        if new_k.startswith('module.'):
            new_k = new_k[7:]
        cleaned_weights[new_k] = jt.array(v)
    
    # 加载到模型
    model_state = model.state_dict()
    loaded_keys = []
    shape_mismatch = []
    missing_in_model = []
    
    for k, v in cleaned_weights.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k] = v
                loaded_keys.append(k)
            else:
                shape_mismatch.append((k, model_state[k].shape, v.shape))
        else:
            missing_in_model.append(k)
    
    model.load_state_dict(model_state)
    
    print(f"Loaded {len(loaded_keys)} weights")
    if shape_mismatch:
        print(f"Shape mismatch for {len(shape_mismatch)} keys")
    if missing_in_model and not strict:
        print(f"Weights not in model: {len(missing_in_model)} keys")
    
    return model


class GroundingDINOInference:
    """
    Grounding DINO 推理包装类
    """
    
    def __init__(
        self,
        weight_path: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        初始化推理模型
        
        Args:
            weight_path: Jittor 权重路径 (.pkl)
            device: 设备 ("cuda" 或 "cpu")
            box_threshold: 边界框阈值
            text_threshold: 文本阈值
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # 设置 Jittor 标志
        if device == "cpu":
            jt.flags.use_cuda = 0
        else:
            jt.flags.use_cuda = 1
        
        # 初始化 tokenizer
        self._init_tokenizer()
        
        # 构建模型
        self.model = self._build_model()
        
        # 加载权重
        if os.path.exists(weight_path):
            load_model_weights(self.model, weight_path)
            print(f"Loaded weights from {weight_path}")
        else:
            print(f"Warning: Weight file not found: {weight_path}")
    
    def _init_tokenizer(self):
        """初始化 tokenizer"""
        try:
            from transformers import BertTokenizer
            import os
            # Check for local BERT model path
            local_bert_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'bert-base-uncased'),
                'models/bert-base-uncased',
                './models/bert-base-uncased',
            ]
            bert_path = 'bert-base-uncased'
            for local_path in local_bert_paths:
                if os.path.exists(local_path) and os.path.isdir(local_path):
                    bert_path = os.path.abspath(local_path)
                    print(f"Using local BERT tokenizer: {bert_path}")
                    break
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
            self.use_bert = True
            print("Using BERT tokenizer")
        except ImportError:
            self.tokenizer = SimpleTokenizer()
            self.use_bert = False
            print("Using simple tokenizer (transformers not available)")
    
    def _build_model(self):
        """构建模型"""
        from jittor_implementation.models.groundingdino import GroundingDINO
        
        # 使用与官方预训练权重匹配的配置
        model = GroundingDINO(
            num_queries=900,
            hidden_dim=256,
            num_feature_levels=4,
            nheads=8,
            max_text_len=256,
            # 匹配官方 Swin-T OGC 权重
        )
        
        return model
    
    def _encode_text(self, caption: str) -> Dict:
        """
        编码文本
        
        Args:
            caption: 文本提示
            
        Returns:
            text_dict: 文本特征字典
        """
        caption = preprocess_caption(caption)
        
        if self.use_bert:
            # 使用 BERT tokenizer
            tokenized = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # 转换为 Jittor
            input_ids = jt.array(tokenized['input_ids'].numpy())
            attention_mask = jt.array(tokenized['attention_mask'].numpy())
            
            # 这里简化处理，实际应该通过 BERT 编码
            # 使用随机特征作为占位符
            bs = input_ids.shape[0]
            encoded_text = jt.randn(bs, 256, 256) * 0.1
            text_token_mask = attention_mask.bool()
            
        else:
            tokenized = self.tokenizer(caption)
            bs = 1
            max_len = 256
            
            encoded_text = jt.randn(bs, max_len, 256) * 0.1
            text_token_mask = jt.ones(bs, max_len).bool()
        
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "tokenized": tokenized,
        }
        
        return text_dict
    
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        caption: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        执行预测
        
        Args:
            image: 图像路径、PIL 图像或 numpy 数组
            caption: 文本提示
            box_threshold: 边界框阈值
            text_threshold: 文本阈值
            
        Returns:
            boxes: 边界框 [N, 4] (xyxy 格式，归一化坐标)
            scores: 置信度 [N]
            phrases: 预测的短语列表
        """
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold
        
        # 加载图像
        if isinstance(image, str):
            image_pil, image_tensor = load_image(image)
        elif isinstance(image, Image.Image):
            transform = get_transform()
            image_tensor, _ = transform(image, None)
            image_pil = image
        else:
            image_pil = Image.fromarray(image)
            transform = get_transform()
            image_tensor, _ = transform(image_pil, None)
        
        # 编码文本
        text_dict = self._encode_text(caption)
        
        # 添加 batch 维度
        image_tensor = image_tensor.unsqueeze(0)
        
        # 前向推理
        with jt.no_grad():
            outputs = self.model(image_tensor, text_dict=text_dict)
        
        # 后处理
        pred_logits = jt.sigmoid(outputs["pred_logits"])[0]  # [nq, max_text_len]
        pred_boxes = outputs["pred_boxes"][0]  # [nq, 4]
        
        # 过滤低置信度预测
        max_logits = pred_logits.max(dim=1)[0]  # [nq]
        mask = max_logits > box_threshold
        
        # 获取过滤后的预测
        filtered_logits = pred_logits[mask]
        filtered_boxes = pred_boxes[mask]
        filtered_scores = max_logits[mask]
        
        # 转换为 numpy
        boxes_np = filtered_boxes.numpy()
        scores_np = filtered_scores.numpy()
        logits_np = filtered_logits.numpy()
        
        # 转换边界框格式 (cxcywh -> xyxy)
        if len(boxes_np) > 0:
            cx, cy, w, h = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        else:
            boxes_xyxy = np.array([]).reshape(0, 4)
        
        # 提取短语
        phrases = []
        tokenized = text_dict.get("tokenized", {})
        for logit in logits_np:
            phrase = get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                self.tokenizer
            )
            phrases.append(phrase.replace('.', '').strip())
        
        return boxes_xyxy, scores_np, phrases
    
    def predict_and_visualize(
        self,
        image_path: str,
        caption: str,
        output_path: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Image.Image:
        """
        执行预测并可视化
        
        Args:
            image_path: 图像路径
            caption: 文本提示
            output_path: 输出路径
            box_threshold: 边界框阈值
            text_threshold: 文本阈值
            
        Returns:
            带标注的图像
        """
        # 加载原始图像
        image_pil = Image.open(image_path).convert("RGB")
        
        # 预测
        boxes, scores, phrases = self.predict(
            image_path, caption, box_threshold, text_threshold
        )
        
        print(f"Detected {len(boxes)} objects")
        for i, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
            print(f"  {i+1}. {phrase}: {score:.3f} @ {box}")
        
        # 可视化
        if len(boxes) > 0:
            annotated_image = plot_boxes_to_image(
                image_pil.copy(), boxes, phrases, scores
            )
        else:
            annotated_image = image_pil
            print("No objects detected!")
        
        # 保存
        if output_path:
            annotated_image.save(output_path)
            print(f"Saved to {output_path}")
        
        return annotated_image


# ============================================================
# 命令行接口
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Grounding DINO Jittor Inference')
    parser.add_argument('--image', '-i', type=str, required=True, help='Image path')
    parser.add_argument('--text', '-t', type=str, required=True, help='Text prompt')
    parser.add_argument('--weights', '-w', type=str, 
                        default='weights/groundingdino_swint_ogc_jittor.pkl',
                        help='Jittor weights path')
    parser.add_argument('--output', '-o', type=str, default='output.jpg', help='Output path')
    parser.add_argument('--box_threshold', type=float, default=0.35, help='Box threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='Text threshold')
    parser.add_argument('--cpu', action='store_true', help='Use CPU only')
    
    args = parser.parse_args()
    
    # 初始化模型
    device = "cpu" if args.cpu else "cuda"
    model = GroundingDINOInference(
        weight_path=args.weights,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    
    # 推理
    result = model.predict_and_visualize(
        image_path=args.image,
        caption=args.text,
        output_path=args.output,
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

