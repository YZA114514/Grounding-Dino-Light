# Text encoder module
from .bert_wrapper import (
    BertModelWarper,
    TextEncoderShell,
    BERTWrapper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
    BaseModelOutputWithPoolingAndCrossAttentions
)

from .text_processor import (
    TextProcessor,
    PhraseProcessor
)