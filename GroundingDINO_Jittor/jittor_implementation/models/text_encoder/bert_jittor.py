# ------------------------------------------------------------------------
# Grounding DINO - Jittor Implementation
# Pure Jittor BERT Implementation (No PyTorch dependency)
# ------------------------------------------------------------------------
"""
纯 Jittor 实现的 BERT 模型

这个实现完全不依赖 PyTorch，可以直接加载从 PyTorch 转换的权重。

结构：
- BertEmbeddings: 词嵌入 + 位置嵌入 + Token类型嵌入
- BertSelfAttention: 多头自注意力
- BertAttention: 注意力 + 残差连接
- BertIntermediate: FFN 中间层
- BertOutput: FFN 输出层
- BertLayer: 一个完整的 Transformer 层
- BertEncoder: 堆叠多个 BertLayer
- BertPooler: 池化层
- BertModel: 完整模型
"""

import math
from typing import Optional, Tuple, Dict, List, Union

import jittor as jt
from jittor import nn
import numpy as np


class BertEmbeddings(nn.Module):
    """BERT 嵌入层"""
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # 注册 position_ids 作为 buffer
        self.position_ids = jt.arange(max_position_embeddings).expand((1, -1))
    
    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        token_type_ids: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
    ) -> jt.Var:
        if input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        else:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = jt.zeros(input_shape, dtype=jt.int64)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """BERT 自注意力"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: jt.Var) -> jt.Var:
        """将张量重塑为多头注意力格式"""
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, jt.Var]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = jt.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = nn.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = jt.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer, attention_probs


class BertSelfOutput(nn.Module):
    """BERT 自注意力输出层"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def execute(self, hidden_states: jt.Var, input_tensor: jt.Var) -> jt.Var:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT 注意力模块 (自注意力 + 输出)"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.self = BertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.output = BertSelfOutput(
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
    
    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, jt.Var]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output, self_outputs[1]


class BertIntermediate(nn.Module):
    """BERT FFN 中间层"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
    
    def execute(self, hidden_states: jt.Var) -> jt.Var:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT FFN 输出层"""
    
    def __init__(
        self,
        intermediate_size: int = 3072,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def execute(self, hidden_states: jt.Var, input_tensor: jt.Var) -> jt.Var:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT Transformer 层"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.output = BertOutput(
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
    
    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, jt.Var]:
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    """BERT Encoder (堆叠多层 BertLayer)"""
    
    def __init__(
        self,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])
    
    def execute(
        self,
        hidden_states: jt.Var,
        attention_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        for layer_module in self.layer:
            hidden_states, _ = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """BERT Pooler"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def execute(self, hidden_states: jt.Var) -> jt.Var:
        # 取 [CLS] token 的输出
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """
    纯 Jittor 实现的 BERT 模型
    
    可以直接加载从 PyTorch 转换的权重。
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            pad_token_id=pad_token_id,
        )
        
        self.encoder = BertEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        
        self.pooler = BertPooler(hidden_size=hidden_size)
        
        # 配置信息
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
    
    def get_extended_attention_mask(
        self,
        attention_mask: jt.Var,
        input_shape: Tuple[int, int],
    ) -> jt.Var:
        """
        将 attention_mask 转换为扩展格式
        
        输入: [batch_size, seq_length] 或 [batch_size, seq_length, seq_length]
        输出: [batch_size, 1, 1, seq_length] 或 [batch_size, 1, seq_length, seq_length]
        """
        if attention_mask.dim() == 3:
            # [batch_size, seq_length, seq_length] -> [batch_size, 1, seq_length, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Wrong attention mask dim: {attention_mask.dim()}")
        
        # 转换: 1 -> 0, 0 -> -10000
        extended_attention_mask = extended_attention_mask.float32()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def execute(
        self,
        input_ids: Optional[jt.Var] = None,
        attention_mask: Optional[jt.Var] = None,
        token_type_ids: Optional[jt.Var] = None,
        position_ids: Optional[jt.Var] = None,
        inputs_embeds: Optional[jt.Var] = None,
    ) -> Tuple[jt.Var, jt.Var]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length] 或 [batch_size, seq_length, seq_length]
            token_type_ids: [batch_size, seq_length]
            position_ids: [batch_size, seq_length]
            inputs_embeds: [batch_size, seq_length, hidden_size]
            
        Returns:
            last_hidden_state: [batch_size, seq_length, hidden_size]
            pooler_output: [batch_size, hidden_size]
        """
        if input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        else:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        
        if attention_mask is None:
            attention_mask = jt.ones((batch_size, seq_length))
        
        # 扩展 attention_mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        
        # Encoder
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
        )
        
        # Pooler
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output


def load_bert_weights_from_dict(
    model: BertModel,
    weights: Dict[str, np.ndarray],
    prefix: str = "module.bert.",
) -> Tuple[int, int]:
    """
    从权重字典加载 BERT 权重
    
    Args:
        model: BertModel 实例
        weights: 权重字典 {name: numpy_array}
        prefix: 权重名称前缀
        
    Returns:
        (loaded_count, total_count)
    """
    model_state = model.state_dict()
    loaded = 0
    total = len([k for k in weights.keys() if k.startswith(prefix)])
    
    for name, param in model_state.items():
        weight_name = prefix + name
        if weight_name in weights:
            weight = weights[weight_name]
            if param.shape == tuple(weight.shape):
                model_state[name] = jt.array(weight)
                loaded += 1
            else:
                print(f"Shape mismatch: {name}: model {param.shape} vs weight {weight.shape}")
    
    model.load_state_dict(model_state)
    
    return loaded, total


# ============================================================
# 简单的 Tokenizer（如果没有 transformers 库）
# ============================================================

class SimpleTokenizer:
    """
    简单的分词器（基本功能）
    
    用于没有 transformers 库时的备用方案。
    实际使用时应该加载 vocab.txt 文件。
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        self.pad_token_id = 0
        self.unk_token_id = 100
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.mask_token_id = 103
        
        self.vocab = {}
        self.ids_to_tokens = {}
        
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
    
    def _load_vocab(self, vocab_file: str):
        """加载词表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx
                self.ids_to_tokens[idx] = token
        
        # 更新特殊 token ID
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 100)
        self.cls_token_id = self.vocab.get(self.cls_token, 101)
        self.sep_token_id = self.vocab.get(self.sep_token, 102)
        self.mask_token_id = self.vocab.get(self.mask_token, 103)
    
    def tokenize(self, text: str) -> List[str]:
        """简单分词（按空格）"""
        return text.lower().split()
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将 tokens 转换为 IDs"""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "np",
    ) -> Dict:
        """
        分词并编码
        """
        if isinstance(text, str):
            text = [text]
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        
        for t in text:
            tokens = [self.cls_token] + self.tokenize(t) + [self.sep_token]
            input_ids = self.convert_tokens_to_ids(tokens)
            
            # 截断
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            
            # 填充
            if padding == "max_length":
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                token_type_ids = token_type_ids + [0] * padding_length
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
        
        result = {
            "input_ids": np.array(batch_input_ids, dtype=np.int64),
            "attention_mask": np.array(batch_attention_mask, dtype=np.int64),
            "token_type_ids": np.array(batch_token_type_ids, dtype=np.int64),
        }
        
        if return_tensors == "jt":
            result = {k: jt.array(v) for k, v in result.items()}
        
        return result


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("Testing Pure Jittor BERT...")
    
    # 创建模型
    model = BertModel(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    
    # 测试输入
    batch_size = 2
    seq_length = 128
    
    input_ids = jt.randint(0, 30522, (batch_size, seq_length))
    attention_mask = jt.ones((batch_size, seq_length))
    
    print(f"Input shape: {input_ids.shape}")
    
    # 前向传播
    last_hidden_state, pooled_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    
    print(f"Last hidden state shape: {last_hidden_state.shape}")
    print(f"Pooled output shape: {pooled_output.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    print("\nPure Jittor BERT test passed!")










