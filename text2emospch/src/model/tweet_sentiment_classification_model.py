
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig
from built.registry import Registry


@Registry.register(category="model")
class TweetSentimentClassificationModel(nn.Module):
    """Tweet Sentiment Classification Model.

    Parameters
    ----------
    transformer_type: 
    transformer_path:
    drop_out_rate:
    num_classes:
    
    Returns
    -------
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Probabilities of each label.
    """

    def __init__(self, transformer_type, transformer_path, drop_out_rate, num_classes):
        super().__init__()
        self.transformer_type = transformer_type
        self.transformer_path = transformer_path
        
        self.config = AutoConfig.from_pretrained(transformer_type)
        self.transformer = AutoModel.from_pretrained(
            transformer_type, output_hidden_states=True)
        
        self.drop_out = nn.Dropout(drop_out_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # classification
        pooled_output = outputs['pooler_output']
        sequence_output = outputs[1]
        pooled_output = self.drop_out(sequence_output)
        classifier_logits = self.classifier(pooled_output)

        return classifier_logits
