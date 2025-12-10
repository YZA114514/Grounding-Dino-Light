# BERT Wrapper (Member C)
import jittor as jt
from jittor import nn
from transformers import BertConfig, BertModel, BertTokenizer
from typing import Dict, List, Optional, Tuple, Union
import torch


class BertModelWarper(nn.Module):
    """Wrapper for BERT model to work with Jittor tensors"""
    
    def __init__(self, bert_model):
        super().__init__()
        self.config = bert_model.config
        
        # Copy components from the original BERT model
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler
        
        # Copy methods from the original BERT model
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.invert_attention_mask = bert_model.invert_attention_mask
        self.get_head_mask = bert_model.get_head_mask
    
    def execute(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Forward pass of BERT model"""
        # Handle input validation
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = "cuda" # Jittor auto-manages device
        
        # Set defaults
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Past key values length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = jt.ones((batch_size, seq_length + past_key_values_length), dtype=jt.bool)
        
        # Create token type ids if not provided
        if token_type_ids is None:
            token_type_ids = jt.zeros(input_shape, dtype=jt.int64)
        
        # Extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        
        # If decoder and encoder hidden states provided
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = jt.ones(encoder_hidden_shape, dtype=jt.bool)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Get embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # Convert embedding_output to PyTorch for encoder
        embedding_output_torch = torch.from_numpy(embedding_output.numpy())
        extended_attention_mask_torch = torch.from_numpy(extended_attention_mask.numpy())
        if encoder_hidden_states is not None:
            encoder_hidden_states_torch = torch.from_numpy(encoder_hidden_states.numpy())
        else:
            encoder_hidden_states_torch = None
        
        # Run encoder using PyTorch BERT
        encoder_outputs = self.encoder(
            embedding_output_torch,
            attention_mask=extended_attention_mask_torch,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states_torch,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Convert back to Jittor
        sequence_output = jt.array(encoder_outputs[0].numpy())
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
            # Convert remaining outputs to Jittor
            jittor_outputs = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    jittor_outputs.append(jt.array(output.numpy()))
                else:
                    jittor_outputs.append(output)
            return tuple(jittor_outputs)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class TextEncoderShell(nn.Module):
    """Shell wrapper for text encoder to handle input/output conversion"""
    
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.config = text_encoder.config
    
    def execute(self, **kwargs):
        """Forward text through encoder"""
        return self.text_encoder(**kwargs)


class BERTWrapper(nn.Module):
    """Complete BERT wrapper for text encoding in GroundingDINO"""
    
    def __init__(self, model_name='bert-base-uncased', max_text_len=256):
        super().__init__()
        self.model_name = model_name
        self.max_text_len = max_text_len
        
        # Initialize tokenizer and BERT model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Wrap the BERT model
        self.text_encoder = BertModelWarper(self.bert)
        
        # Get special tokens
        self.special_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ]
        
        # Feature mapping from BERT to hidden_dim
        self.feat_map = None  # Will be set to match the model's hidden_dim
    
    def set_feat_map(self, feat_map):
        """Set feature mapping layer"""
        print(f"Setting feat_map: {feat_map}")
        self.feat_map = feat_map
        print(f"feat_map set to: {self.feat_map}")
    
    def execute(self, text: Union[str, List[str]], sub_sentence_present=True):
        """
        Process text and return encoded features
        
        Args:
            text: Input text string or list of strings
            sub_sentence_present: Whether to use sub-sentence presentation
            
        Returns:
            Dict containing encoded text and related masks
        """
        print(f"In execute: self.feat_map is {self.feat_map}")  # DEBUG
        # Tokenize text
        if isinstance(text, str):
            text = [text]
        
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
        # Convert to Jittor tensors
        for key in tokenized:
            tokenized[key] = jt.array(tokenized[key].numpy())
        
        # Generate special tokens masks and position IDs
        if sub_sentence_present:
            text_self_attention_masks, position_ids, cate_to_token_mask_list = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens, self.tokenizer
                )
        else:
            text_self_attention_masks, position_ids = \
                generate_masks_with_special_tokens(
                    tokenized, self.special_tokens, self.tokenizer
                )
            cate_to_token_mask_list = None
        
        # Truncate if necessary
        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]
            position_ids = position_ids[:, :self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :self.max_text_len]
        
        # Prepare for encoder
        if sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized
        
        # Run BERT (using PyTorch backend for this part)
        import torch
        
        # Convert inputs to PyTorch
        pt_inputs = {}
        for k, v in tokenized_for_encoder.items():
            if isinstance(v, jt.Var):
                pt_inputs[k] = torch.from_numpy(v.numpy())
            else:
                pt_inputs[k] = v
        
        # Ensure attention_mask is float for PyTorch BERT if it's a boolean mask
        # PyTorch BERT expects float mask (0.0 for keep, -10000.0 for mask) if it's 3D/4D
        # Or boolean mask (True for keep, False for mask) depending on version.
        # But standard BERT usually takes 2D mask. 
        # If we pass 3D mask, we might need to adjust.
        # However, let's try passing it as is first, but convert to int/float if needed.
        
        # Run PyTorch model
        with torch.no_grad():
            # self.bert is the PyTorch model
            bert_output = self.bert(**pt_inputs)
        
        # Get last hidden state and convert to Jittor
        last_hidden_state_pt = bert_output.last_hidden_state if hasattr(bert_output, 'last_hidden_state') else bert_output[0]
        
        # DEBUG
        print(f"BERT last_hidden_state_pt stats: min={last_hidden_state_pt.min().item()}, max={last_hidden_state_pt.max().item()}")
        
        last_hidden_state = jt.array(last_hidden_state_pt.numpy())
        
        # DEBUG: Check feat_map weights
        if self.feat_map is not None:
            print(f"feat_map.weight stats: min={self.feat_map.weight.min()}, max={self.feat_map.weight.max()}")
            print(f"feat_map.bias stats: min={self.feat_map.bias.min()}, max={self.feat_map.bias.max()}")
            print(f"feat_map.weight shape: {self.feat_map.weight.shape}")
            print(f"last_hidden_state shape: {last_hidden_state.shape}")
        
        # Apply feature mapping if available
        if self.feat_map is not None:
            print(f"Before feat_map: min={last_hidden_state.min()}, max={last_hidden_state.max()}")
            # Let's manually compute to check
            manual_output = last_hidden_state @ self.feat_map.weight.t() + self.feat_map.bias
            print(f"Manual computation: min={manual_output.min()}, max={manual_output.max()}")
            encoded_text = self.feat_map(last_hidden_state)
            print(f"After feat_map: min={encoded_text.min()}, max={encoded_text.max()}")
            diff = (manual_output - encoded_text).abs().max()
            print(f"Max difference: {diff}")
        else:
            encoded_text = last_hidden_state
        
        # Ensure float32 dtype for consistency
        encoded_text = encoded_text.float32()
        
        # Get token mask
        text_token_mask = tokenized.attention_mask.bool()
        
        # Truncate if necessary
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, :self.max_text_len, :]
            text_token_mask = text_token_mask[:, :self.max_text_len]
            position_ids = position_ids[:, :self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, :self.max_text_len, :self.max_text_len]
        
        # Return text features dictionary
        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
            "cate_to_token_mask_list": cate_to_token_mask_list,
        }


def generate_masks_with_special_tokens(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens
    
    Args:
        tokenized: Tokenized input dictionary
        special_tokens_list: List of special token IDs
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of attention_mask and position_ids
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # Special tokens mask
    special_tokens_mask = jt.zeros((bs, num_token), dtype=jt.bool)
    for special_token in special_tokens_list:
        special_tokens_mask = special_tokens_mask | (input_ids == special_token)
    
    # Get indices of special tokens
    idxs = jt.nonzero(special_tokens_mask)
    
    # Generate attention mask and positional IDs
    # attention_mask = jt.eye(num_token, dtype=jt.bool).unsqueeze(0).repeat(bs, 1, 1)
    attention_mask = (jt.init.eye(num_token, dtype=jt.float32) > 0.5).unsqueeze(0).repeat(bs, 1, 1)
    position_ids = jt.zeros((bs, num_token), dtype=jt.int64)
    
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = jt.arange(
                0, col - previous_col, dtype=jt.int64
            )
        previous_col = col
    
    return attention_mask, position_ids


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list, tokenizer):
    """Generate attention mask between each pair of special tokens with transfer map
    
    Args:
        tokenized: Tokenized input dictionary
        special_tokens_list: List of special token IDs
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple of attention_mask, position_ids, and cate_to_token_mask_list
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    
    # Special tokens mask
    special_tokens_mask = jt.zeros((bs, num_token), dtype=jt.bool)
    for special_token in special_tokens_list:
        special_tokens_mask = special_tokens_mask | (input_ids == special_token)
    
    # Get indices of special tokens
    idxs = jt.nonzero(special_tokens_mask)
    
    # Generate attention mask and positional IDs
    attention_mask = jt.init.eye(num_token, dtype=jt.bool).unsqueeze(0).repeat(bs, 1, 1)
    position_ids = jt.zeros((bs, num_token), dtype=jt.int64)
    cate_to_token_mask_list = [[] for _ in range(bs)]
    
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = jt.arange(
                0, col - previous_col, dtype=jt.int64
            )
            c2t_maski = jt.zeros((num_token), dtype=jt.bool)
            c2t_maski[previous_col + 1 : col] = True
            cate_to_token_mask_list[row].append(c2t_maski)
        previous_col = col
    
    # Stack category to token masks
    cate_to_token_mask_list = [
        jt.stack(cate_to_token_mask_listi, dim=0)
        for cate_to_token_mask_listi in cate_to_token_mask_list
    ]
    
    return attention_mask, position_ids, cate_to_token_mask_list


class BaseModelOutputWithPoolingAndCrossAttentions:
    """Base class for model's outputs with pooling and cross-attentions"""
    
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