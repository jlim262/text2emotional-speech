#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=text2emospch/input
export EmotionDataRepo=6H-AP_emotion_labelled_tweets

kaggle competitions download -c $KaggleCompName -p $DownloadPath
unzip $DownloadPath/$KaggleCompName.zip -d $DownloadPath/$KaggleCompName
git clone https://github.com/EmotionDetection/$EmotionDataRepo.git $DownloadPath/tweet-emotion-data
cd $EmotionDataRepo && git lfs pull