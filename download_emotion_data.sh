#!/bin/bash

export DownloadPath=text2emospch/input
export EmotionDataRepo=6H-AP_emotion_labelled_tweets

git clone https://github.com/EmotionDetection/$EmotionDataRepo.git $DownloadPath/tweet-emotion-data
cd $EmotionDataRepo && git lfs pull