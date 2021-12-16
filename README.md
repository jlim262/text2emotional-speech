# text2emotional-speech
accuracy for sentiment classification(neutral, positive, negative): ~80%
accuracy for sentiment classification(anger, joy, disgust, fear, sadness, surprise): ~78%

## Prerequisite
Tested on Python 3.8

## Quick Demo
### Sentiment classification
```
git clone git@github.com:jlim262/text2emotional-speech.git
cd text2emotional-speech
pip install -r requirements.txt
python sentiment_classifier.py
```

This demo predicts sentiment of some example sentences as below
```
[hello, my name is James]: [neutral] (prob: 0.9779650568962097) (took 0.30719780921936035 sec)
[i'm so happy for you]: [positive] (prob: 0.996790885925293) (took 0.302915096282959 sec)
[it's too bad]: [negative] (prob: 0.9916732311248779) (took 0.29944705963134766 sec)
[The seven-member Expedition 66 crew spent Thursday servicing physics research gear and exercise hardware aboard the International Space Station]: [neutral] (prob: 0.982895016670227) (took 0.2983851432800293 sec)
[I don’t like their customer support]: [negative] (prob: 0.9921891093254089) (took 0.297562837600708 sec)
[I was on hold for 40 minutes, their customer support service is a nightmare]: [negative] (prob: 0.9848304986953735) (took 0.29828524589538574 sec)
[I love how Zapier takes different apps and ties them together]: [positive] (prob: 0.9852762818336487) (took 0.2990422248840332 sec)
[Zapier is sooooo confusing to me]: [negative] (prob: 0.9877282977104187) (took 0.3012700080871582 sec)
[I still need to further test Zapier to say if its useful for me or not]: [neutral] (prob: 0.8137592077255249) (took 0.29779767990112305 sec)
[The older interface was much simpler]: [positive] (prob: 0.47357383370399475) (took 0.2997128963470459 sec)
[I don't think there is anything I really dislike about the product]: [negative] (prob: 0.6636339426040649) (took 0.2977290153503418 sec)
```

## Training Sentiment Classification Model
TBD

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

