from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import yaml
import torch
import numpy as np
import os.path


from google_drive_downloader import GoogleDriveDownloader as gdd
from transformers import AutoTokenizer, AutoConfig
from easydict import EasyDict as edict
from built.builder import Builder
from built.utils.util_functions import *
# from text2emospch.src.splitter.traintest_splitter import SentimentDataSplitter, EmotionDataSplitter
from text2emospch.src.model.tweet_sentiment_classification_model import TweetSentimentClassificationModel


class SentimentClassifier:
    def __init__(self, config):
        with open(config) as f:
            self.config = edict(yaml.load(f, Loader=yaml.FullLoader))
            self.config = replace_placeholder(self.config, self.config)

            self.sentiment2target = {'neutral': 0, 'positive': 1, 'negative': 2}
            self.target2sentiment = dict([(value, key) for key, value in self.sentiment2target.items()])
            self.max_len = self.config.dataset.params.max_len

            transformer_type = self.config.dataset.params.transformer_type
            config = AutoConfig.from_pretrained(transformer_type)
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_type, config=config)

            builder = Builder()
            self.model = builder.build_model(self.config)

            # download trained model from the following link
            # https://drive.google.com/file/d/1jKxjAM30GmOUPm0ZKFmSlXSmj7c9Lmlb/view?usp=sharing

            model_path = './sentiment_classification_trained.pth'
            if not os.path.exists(model_path):
                gdd.download_file_from_google_drive(file_id='1jKxjAM30GmOUPm0ZKFmSlXSmj7c9Lmlb',
                                        dest_path=model_path,
                                        unzip=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
    

    def generate_input(self, text):
        tweet = " " + " ".join(text.lower().split())

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

        return inputs

    def predict(self, text):
        inputs = self.generate_input(text)
        output = self.model(
            input_ids=inputs['input_ids'].unsqueeze(0),
            attention_mask=inputs['attention_mask'].unsqueeze(0),
            token_type_ids=inputs['token_type_ids'].unsqueeze(0)
        )
        output = output.squeeze(0)
        output = output.cpu().detach().numpy()
        sentiment_index = np.argmax(output, axis=0)
        sentiment_prob = np.exp(output)/np.sum(np.exp(output))
        return self.target2sentiment[sentiment_index], sentiment_prob[sentiment_index]
    


if __name__ == '__main__':
    sc = SentimentClassifier('config/sentiment_classification.yaml')

    texts = [
        "hello, my name is James",
        "i'm so happy for you",
        "it's too bad",
        "The seven-member Expedition 66 crew spent Thursday servicing physics research gear and exercise hardware aboard the International Space Station",
        "I don’t like their customer support",
        "I was on hold for 40 minutes, their customer support service is a nightmare",
        "I love how Zapier takes different apps and ties them together",
        "Zapier is sooooo confusing to me",
        "I still need to further test Zapier to say if its useful for me or not",
        "The older interface was much simpler",
        "I don't think there is anything I really dislike about the product"
    ]

    for text in texts:
        s = time.time()        
        sentiment, prob = sc.predict(text)
        print(f'[{text}]: [{sentiment}] [prob: {prob}] [took {time.time() - s} sec]')



    