import os
import numpy as np
import pandas as pd
import torch

from torch.nn import functional as F
from transformers import AutoTokenizer, AutoConfig
from built.registry import Registry


@Registry.register(category='dataset')
class TweetIndexDatasetBase(torch.utils.data.Dataset):
    def __init__(self, transformer_path, csv_path, transformer_type='roberta', train=False, split='train', max_len=96, inference=False):
        df = pd.read_csv(csv_path)

        self.sentiment2target = {'neutral': 0, 'positive': 1, 'negative': 2}
        self.df = df.dropna().reset_index(drop=True)
        self.max_len = max_len
        self.labeled = 'selected_text' in self.df
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
            tweet = " " + " ".join(row.text.lower().split())
        except:
            raise RuntimeError(f'{row}')

        sentiment = row.sentiment
        
        encoded = self.tokenizer(
            tweet,
            sentiment,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_tensors='pt')
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        offsets = encoded['offset_mapping']
        char_centers = [(x[0] + x[1]) / 2 for x in offsets]
        
        inputs = {}
        inputs['input_ids'] = torch.squeeze(input_ids)
        inputs['attention_mask'] = torch.squeeze(attention_mask)
        inputs['token_type_ids'] = torch.squeeze(token_type_ids)
        inputs['offsets'] = torch.squeeze(offsets)
        inputs['tweet'] = tweet
        
        targets = {}
        if self.inference:
            targets = torch.tensor(np.nan)
        else:
            # target = torch.tensor(self.sentiment2target[row.sentiment])
            start_idx, end_idx, selected_text = self.get_target_idx(row, tweet, offsets)
            targets['start_idx'] = start_idx
            targets['end_idx'] = end_idx
            targets['selected_text'] = selected_text
            
        return inputs, targets
        # ids, masks, tweet, offsets, sentiment_id, sentiment_target, char_centers = self.get_input_data(
        #     row)
        # data['ids'] = ids
        # data['masks'] = masks
        # data['tweet'] = tweet
        # data['offsets'] = offsets
        # data['sentiment_id'] = sentiment_id
        # data['sentiment_target'] = sentiment_target
        # data['char_centers'] = char_centers
        # target['sentiment_id'] = sentiment_id
        # target['sentiment_target'] = sentiment_target

        # if self.labeled:
        #     start_idx, end_idx, selected_text = self.get_target_idx(
        #         row, tweet, offsets)
        #     data['start_idx'] = start_idx
        #     data['end_idx'] = end_idx
        #     data['selected_text'] = selected_text
        #     target['start_idx'] = start_idx
        #     target['end_idx'] = end_idx
        #     target['selected_text'] = selected_text
        #     target['offsets'] = offsets
        #     target['tweet'] = tweet

        # return data, target

    def __len__(self):
        return len(self.df)

    def get_target_idx(self, row, tweet, offsets):
        start_idx = 0
        end_idx = 0
        try:
            tweet = " " + " ".join(str(tweet).split())
            selected_text = " ".join(row.selected_text.lower().split())

            if len(selected_text) != selected_text.count(' '):
                start_idx = tweet.find(selected_text)
                end_idx = start_idx + len(selected_text)

                char_targets = [0] * len(tweet)
                if start_idx != None and end_idx != None:
                    for ct in range(start_idx, end_idx):
                        char_targets[ct] = 1

                target_idx = []
                for j, (offset1, offset2) in enumerate(offsets):
                    if sum(char_targets[offset1: offset2]) > 0:
                        target_idx.append(j)

                start_idx = target_idx[0]
                end_idx = target_idx[-1]
        except:
            print("selected_text is empty with spaces")

        return start_idx, end_idx, selected_text
