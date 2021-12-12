# text2emotional-speech
accuracy for sentiment classification(neutral, positive, negative): ~80%
accuracy for sentiment classification(anger, joy, disgust, fear, sadness, surprise): ~78%

## Quick Demo
### download pre-trained model
### emotion classification
### sentiment classification

## Training Sentiment Classification Model
### download dataset
Download sentiment dataset using download_sentiment_data.sh file. To download it, kaggle(https://kaggle.com) API credendial is required because of license issue. Please refer to https://github.com/Kaggle/kaggle-api for instruction. 
### train a model
```
python main.py train  with config/sentiment_classification.yaml -f
```


## Training Emotion Classification Model
### download dataset
Download emotion dataset using download_sentiment_data.sh
### train a model
python main.py train  with config/emotion_classification.yaml -f


