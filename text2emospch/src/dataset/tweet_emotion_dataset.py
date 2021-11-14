
import os
import numpy as np
import pandas as pd
import torch

from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig
from built.registry import Registry

@Registry.register(category='dataset')
class TweetEmotionDataset(torch.utils.data.Dataset):
    def __init__(self, transformer_path, csv_path, transformer_type='roberta', train=False, split='train', max_len=96, inference=False):
        df = pd.read_csv(csv_path)
        
        self.emotion2target = {'anger': 0, 'joy': 1,
                               'disgust': 2, 'fear': 3, 'sadness': 4, 'surprise': 5}
        
        self.df = df.dropna().reset_index(drop=True)
        self.max_len = max_len
        self.transformer_type = transformer_type
        self.tokenizer = self.get_tokenizer(self.transformer_type)
        self.inference = inference

    def get_tokenizer(self, transformer):
        config = AutoConfig.from_pretrained(transformer)
        tokenizer = AutoTokenizer.from_pretrained(
            transformer, config=config)
        
        return tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]
        try:
            tweet = " " + " ".join(row.Comments.lower().split())
        except:
            raise RuntimeError(f'{row}')

        encoded = self.tokenizer(
            tweet,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt')

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']

        inputs = {}
        inputs['input_ids'] = torch.squeeze(input_ids)
        inputs['attention_mask'] = torch.squeeze(attention_mask)
        inputs['token_type_ids'] = torch.squeeze(token_type_ids)
        inputs['tweet'] = tweet

        if self.inference:
            target = torch.tensor(np.nan)
        else:
            target = torch.tensor(self.emotion2target[row.Emotion])

        return inputs, target

    def __len__(self):
        return len(self.df)

    
