# Text Processor (Member C)
import jittor as jt
from jittor import nn
from transformers import BertTokenizer
from typing import Dict, List, Optional, Tuple, Union
import re
import numpy as np


class TextProcessor:
    """Text processor for clause-level processing in GroundingDINO"""
    
    def __init__(self, model_name='bert-base-uncased', max_text_len=256):
        super().__init__()
        self.max_text_len = max_text_len
        
        # Check for local BERT model path
        import os
        local_bert_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'bert-base-uncased'),
            'models/bert-base-uncased',
            './models/bert-base-uncased',
        ]
        
        actual_model_path = model_name
        for local_path in local_bert_paths:
            if os.path.exists(local_path) and os.path.isdir(local_path):
                actual_model_path = os.path.abspath(local_path)
                break
        
        self.model_name = actual_model_path
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(actual_model_path)
        
        # Get special tokens
        self.special_tokens = {
            'cls': self.tokenizer.cls_token_id,
            'sep': self.tokenizer.sep_token_id,
            'pad': self.tokenizer.pad_token_id,
        }
    
    def execute(
        self, 
        text: Union[str, List[str]], 
        categories: Optional[List[List[str]]] = None,
        phrases: Optional[List[List[str]]] = None
    ):
        """
        Process text, build clause-level representations
        
        Args:
            text: Input text string or list of strings
            categories: List of categories for each text
            phrases: List of phrases for each text
            
        Returns:
            Dict containing processed text and masks
        """
        if isinstance(text, str):
            text = [text]
            if categories is not None and len(categories) > 0:
                categories = [categories]
            if phrases is not None and len(phrases) > 0:
                phrases = [phrases]
        
        processed_texts = []
        attention_masks = []
        position_ids_list = []
        cate_to_token_masks = []
        
        for i, txt in enumerate(text):
            if phrases is not None and len(phrases) > i:
                # Use provided phrases
                phrase_list = phrases[i]
            elif categories is not None and len(categories) > i:
                # Extract phrases from categories
                phrase_list = self._extract_phrases_from_categories(txt, categories[i])
            else:
                # Split text by '.' (sub-sentence delimiter)
                phrase_list = self._split_text_to_phrases(txt)
            
            # Process phrases
            processed_txt, attn_mask, pos_ids, cate_token_mask = self._process_phrases(
                phrase_list, txt, categories[i] if categories is not None and len(categories) > i else None
            )
            
            processed_texts.append(processed_txt)
            attention_masks.append(attn_mask)
            position_ids_list.append(pos_ids)
            if cate_token_mask is not None:
                cate_to_token_masks.append(cate_token_mask)
        
        # Tokenize all texts
        tokenized = self.tokenizer(
            processed_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
        # Convert to Jittor tensors
        for key in tokenized:
            tokenized[key] = jt.array(tokenized[key].numpy())
        
        # Convert lists to tensors
        attention_masks = jt.stack(attention_masks, dim=0)
        position_ids_list = jt.stack(position_ids_list, dim=0)
        
        # Handle category to token masks
        cate_to_token_mask_list = None
        if len(cate_to_token_masks) > 0:
            # Pad to same number of categories
            max_cats = max(len(masks) for masks in cate_to_token_masks)
            padded_cate_to_token_masks = []
            
            for masks in cate_to_token_masks:
                if len(masks) < max_cats:
                    # Pad with empty masks
                    padding = [jt.zeros(self.max_text_len, dtype=jt.bool) for _ in range(max_cats - len(masks))]
                    masks = masks + padding
                
                # Stack masks for this sample
                sample_masks = jt.stack(masks, dim=0)
                padded_cate_to_token_masks.append(sample_masks)
            
            cate_to_token_mask_list = jt.stack(padded_cate_to_token_masks, dim=0)
        
        # Return processed text dictionary
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            "text_self_attention_masks": attention_masks,
            "position_ids": position_ids_list,
            "cate_to_token_mask_list": cate_to_token_mask_list,
        }
    
    def _extract_phrases_from_categories(self, text: str, categories: List[str]):
        """Extract phrases from categories in text"""
        # This is a simplified implementation
        # In practice, more sophisticated NLP techniques might be used
        
        # Create phrases from categories
        phrases = []
        for cat in categories:
            # Look for exact category name in text
            if cat.lower() in text.lower():
                phrases.append(cat)
            else:
                # Split category into words and look for matches
                words = re.findall(r'\w+', cat.lower())
                phrase_parts = []
                for word in words:
                    if word in text.lower():
                        phrase_parts.append(word)
                
                if phrase_parts:
                    phrases.append(" ".join(phrase_parts))
        
        # If no phrases found, split by '.'
        if not phrases:
            phrases = self._split_text_to_phrases(text)
        
        return phrases
    
    def _split_text_to_phrases(self, text: str):
        """Split text into phrases by '.' and clean"""
        # Split by '.' (sentence delimiter)
        raw_phrases = text.split('.')
        
        # Clean and filter empty phrases
        phrases = []
        for phrase in raw_phrases:
            phrase = phrase.strip()
            if phrase:  # Only add non-empty phrases
                phrases.append(phrase)
        
        # If still empty after filtering, use the whole text
        if not phrases:
            phrases = [text.strip()]
        
        return phrases
    
    def _process_phrases(
        self, 
        phrase_list: List[str], 
        original_text: str,
        categories: Optional[List[str]] = None
    ):
        """Process phrases and generate attention masks"""
        # Reconstruct text with phrases separated by '.'
        processed_text = ' . '.join(phrase_list)
        
        # Tokenize processed text
        tokenized = self.tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # Find special token positions
        special_token_positions = []
        for i, token_id in enumerate(input_ids):
            if token_id in [self.special_tokens['cls'], self.special_tokens['sep']]:
                special_token_positions.append(i)
        
        # Generate attention mask between special tokens
        num_tokens = len(input_ids)
        # attention_mask_matrix = jt.eye(num_tokens, dtype=jt.bool)
        attention_mask_matrix = (jt.init.eye(num_tokens, dtype=jt.float32) > 0.5)
        position_ids = jt.zeros(num_tokens, dtype=jt.int64)
        cate_to_token_masks = []
        
        previous_special = 0
        for special_pos in special_token_positions:
            if special_pos == 0 or special_pos == num_tokens - 1:
                # For CLS and final SEP
                attention_mask_matrix[special_pos, special_pos] = True
                position_ids[special_pos] = 0
            else:
                # For phrase delimiters
                attention_mask_matrix[previous_special + 1 : special_pos + 1, 
                                    previous_special + 1 : special_pos + 1] = True
                position_ids[previous_special + 1 : special_pos + 1] = jt.arange(
                    0, special_pos - previous_special, dtype=jt.int64
                )
                
                # Create category to token mask
                cate_token_mask = jt.zeros(num_tokens, dtype=jt.bool)
                cate_token_mask[previous_special + 1 : special_pos] = True
                cate_to_token_masks.append(cate_token_mask)
            
            previous_special = special_pos
        
        return processed_text, attention_mask_matrix, position_ids, cate_to_token_masks


class PhraseProcessor(nn.Module):
    """Module to process phrases for GroundingDINO"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear layer to map phrase embeddings
        self.phrase_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def execute(self, text_features, cate_to_token_mask_list):
        """
        Process text features at phrase level
        
        Args:
            text_features: (B, L, D) text features from BERT
            cate_to_token_mask_list: List of category-to-token masks
            
        Returns:
            Processed phrase features (B, N, D) where N is number of phrases
        """
        batch_size = text_features.shape[0]
        
        # Process each sample in the batch
        phrase_features = []
        for i in range(batch_size):
            sample_features = []
            
            # Get masks for this sample
            if cate_to_token_mask_list is not None and i < len(cate_to_token_mask_list):
                sample_masks = cate_to_token_mask_list[i]
                
                for mask in sample_masks:
                    # Apply mask to get phrase tokens
                    phrase_tokens = text_features[i][mask].unsqueeze(0)  # (1, num_tokens_in_phrase, D)
                    
                    # Average pooling over phrase tokens
                    phrase_feat = jt.mean(phrase_tokens, dim=1)  # (1, D)
                    
                    # Project and normalize
                    phrase_feat = self.phrase_proj(phrase_feat)
                    phrase_feat = self.layer_norm(phrase_feat)
                    
                    sample_features.append(phrase_feat)
            
            if sample_features:
                # Stack phrase features for this sample
                sample_phrase_features = jt.concat(sample_features, dim=0)  # (N, D)
            else:
                # If no phrases, use [CLS] token
                sample_phrase_features = text_features[i][0:1]  # (1, D)
            
            phrase_features.append(sample_phrase_features)
        
        # Pad to same number of phrases across samples
        max_phrases = max(feat.shape[0] for feat in phrase_features)
        padded_phrase_features = []
        
        for feat in phrase_features:
            if feat.shape[0] < max_phrases:
                # Pad with zeros
                padding = jt.zeros(max_phrases - feat.shape[0], feat.shape[1])
                feat = jt.concat([feat, padding], dim=0)
            padded_phrase_features.append(feat)
        
        # Stack across batch
        phrase_features = jt.stack(padded_phrase_features, dim=0)  # (B, N, D)
        
        return phrase_features