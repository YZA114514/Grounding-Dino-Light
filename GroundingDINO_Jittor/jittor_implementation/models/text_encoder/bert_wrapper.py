# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# BERT Wrapper for Text Encoding (Pure Jittor - No PyTorch dependency)
# ------------------------------------------------------------------------
"""
BERT Wrapper 模块

使用纯 Jittor 实现的 BERT 模型进行文本编码。
完全不依赖 PyTorch。
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import jittor as jt
from jittor import nn
import numpy as np

# 导入纯 Jittor BERT
from .bert_jittor import BertModel, load_bert_weights_from_dict


class BERTWrapper(nn.Module):
    """
    完整的 BERT 文本编码器包装器
    
    使用纯 Jittor 实现，可以直接加载从 PyTorch 转换的权重。
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_text_len: int = 256):
        super().__init__()
        self.max_text_len = max_text_len
        self.model_name = model_name
        
        # 初始化纯 Jittor BERT
        self.bert = BertModel(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
        )
        
        # 初始化 tokenizer
        self._init_tokenizer(model_name)
        
        # 获取特殊 tokens
        self.special_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ]
        
        # Feature mapping layer (will be set externally)
        self.feat_map = None
    
    def _init_tokenizer(self, model_name: str):
        """初始化 tokenizer"""
        # 检查本地 BERT 模型路径
        local_bert_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'bert-base-uncased'),
            'models/bert-base-uncased',
            './models/bert-base-uncased',
        ]
        
        vocab_file = None
        for local_path in local_bert_paths:
            vocab_path = os.path.join(local_path, 'vocab.txt')
            if os.path.exists(vocab_path):
                vocab_file = vocab_path
                print(f"Using local BERT vocab: {vocab_path}")
                break
        
        # 尝试使用 transformers tokenizer（如果可用）
        try:
            from transformers import BertTokenizer
            if vocab_file:
                self.tokenizer = BertTokenizer(vocab_file=vocab_file)
            else:
                # 尝试从本地加载
                for local_path in local_bert_paths:
                    if os.path.exists(local_path) and os.path.isdir(local_path):
                        self.tokenizer = BertTokenizer.from_pretrained(local_path, local_files_only=True)
                        break
                else:
                    # 最后尝试在线加载
                    self.tokenizer = BertTokenizer.from_pretrained(model_name)
            print("Using HuggingFace BertTokenizer")
        except Exception as e:
            print(f"Warning: Could not load HuggingFace tokenizer: {e}")
            # 使用简单的备用 tokenizer
            from .bert_jittor import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(vocab_file=vocab_file)
            print("Using SimpleTokenizer (fallback)")
    
    def set_feat_map(self, feat_map: nn.Linear):
        """设置特征映射层"""
        self.feat_map = feat_map
    
    def load_bert_weights(self, weights: Dict[str, np.ndarray], prefix: str = "module.bert."):
        """
        从权重字典加载 BERT 权重
        
        Args:
            weights: 完整的模型权重字典
            prefix: BERT 权重的前缀
        """
        loaded, total = load_bert_weights_from_dict(self.bert, weights, prefix)
        print(f"Loaded {loaded}/{total} BERT weights")
    
    def execute(self, text: Union[str, List[str]], sub_sentence_present: bool = True) -> Dict:
        """
        处理文本并返回编码特征
        
        Args:
            text: 输入文本字符串或字符串列表
            sub_sentence_present: 是否使用子句级表示
            
        Returns:
            包含编码文本和相关掩码的字典
        """
        # Tokenize text
        if isinstance(text, str):
            text = [text]
        
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="np"  # 返回 numpy 数组
        )
        
        # 转换为 Jittor 张量
        input_ids = jt.array(tokenized["input_ids"]).int64()
        attention_mask = jt.array(tokenized["attention_mask"]).int64()
        token_type_ids = jt.array(tokenized.get("token_type_ids", np.zeros_like(tokenized["input_ids"]))).int64()
        
        # 生成特殊 tokens 掩码和位置 IDs
        if sub_sentence_present:
            text_self_attention_masks, position_ids, cate_to_token_mask_list = \
                generate_masks_with_special_tokens_and_transfer_map(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    self.special_tokens
                )
        else:
            text_self_attention_masks, position_ids = \
                generate_masks_with_special_tokens(
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    self.special_tokens
                )
            cate_to_token_mask_list = None
        
        # 截断（如果需要）
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]
            position_ids = position_ids[:, :self.max_text_len]
            input_ids = input_ids[:, :self.max_text_len]
            attention_mask = attention_mask[:, :self.max_text_len]
            token_type_ids = token_type_ids[:, :self.max_text_len]
        
        # 运行 BERT
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=text_self_attention_masks if sub_sentence_present else attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
        # 应用特征映射
        if self.feat_map is not None:
            encoded_text = self.feat_map(last_hidden_state)
        else:
            encoded_text = last_hidden_state
        
        # 确保 float32
        encoded_text = encoded_text.float32()
        
        # 获取 token mask
        text_token_mask = attention_mask.bool()
        
        # 截断输出（如果需要）
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, :self.max_text_len, :]
            text_token_mask = text_token_mask[:, :self.max_text_len]
            position_ids = position_ids[:, :self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]
        
        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
            "cate_to_token_mask_list": cate_to_token_mask_list,
        }


def generate_masks_with_special_tokens(
    tokenized: Dict,
    special_tokens_list: List[int]
) -> Tuple[jt.Var, jt.Var]:
    """
    生成带有特殊 tokens 的注意力掩码
    
    Args:
        tokenized: 分词后的输入字典
        special_tokens_list: 特殊 token IDs 列表
        
    Returns:
        (attention_mask, position_ids)
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # 特殊 tokens 掩码
    special_tokens_mask = jt.zeros((bs, num_token), dtype=jt.bool)
    for special_token in special_tokens_list:
        special_tokens_mask = special_tokens_mask | (input_ids == special_token)
    
    # 获取特殊 tokens 的索引
    idxs = jt.nonzero(special_tokens_mask)
    
    # 生成注意力掩码和位置 IDs
    attention_mask = (jt.init.eye(num_token, dtype=jt.float32) > 0.5).unsqueeze(0).repeat(bs, 1, 1)
    position_ids = jt.zeros((bs, num_token), dtype=jt.int64)
    
    previous_col = 0
    for i in range(idxs.shape[0]):
        row = int(idxs[i, 0].item())
        col = int(idxs[i, 1].item())
        
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1, previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = jt.arange(0, col - previous_col, dtype=jt.int64)
        previous_col = col
    
    return attention_mask, position_ids


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized: Dict,
    special_tokens_list: List[int]
) -> Tuple[jt.Var, jt.Var, List]:
    """
    生成带有特殊 tokens 和传输映射的注意力掩码
    
    Args:
        tokenized: 分词后的输入字典
        special_tokens_list: 特殊 token IDs 列表
        
    Returns:
        (attention_mask, position_ids, cate_to_token_mask_list)
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # 特殊 tokens 掩码
    special_tokens_mask = jt.zeros((bs, num_token), dtype=jt.bool)
    for special_token in special_tokens_list:
        special_tokens_mask = special_tokens_mask | (input_ids == special_token)
    
    # 获取特殊 tokens 的索引
    idxs = jt.nonzero(special_tokens_mask)
    
    # 生成注意力掩码和位置 IDs
    attention_mask = jt.init.eye(num_token, dtype=jt.bool).unsqueeze(0).repeat(bs, 1, 1)
    position_ids = jt.zeros((bs, num_token), dtype=jt.int64)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    
    previous_col = 0
    for i in range(idxs.shape[0]):
        row = int(idxs[i, 0].item())
        col = int(idxs[i, 1].item())
        
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1, previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = jt.arange(0, col - previous_col, dtype=jt.int64)
            c2t_maski = jt.zeros((num_token,), dtype=jt.bool)
            c2t_maski[previous_col + 1:col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    
    # 堆叠 category to token masks
    for i in range(bs):
        if cate_to_token_mask_list[i]:
            cate_to_token_mask_list[i] = jt.stack(cate_to_token_mask_list[i], dim=0)
        else:
            cate_to_token_mask_list[i] = jt.zeros((0, num_token), dtype=jt.bool)
    
    return attention_mask, position_ids, cate_to_token_mask_list


# 为了向后兼容，保留旧的类名
class BertModelWarper(nn.Module):
    """向后兼容的别名"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: BertModelWarper is deprecated, use BERTWrapper instead")


class TextEncoderShell(nn.Module):
    """向后兼容的别名"""
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
    
    def execute(self, **kwargs):
        return self.text_encoder(**kwargs)


class BaseModelOutputWithPoolingAndCrossAttentions:
    """模型输出的基类"""
    
    def __init__(
        self,
        last_hidden_state,
        pooler_output=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
    ):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions
