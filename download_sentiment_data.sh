#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction

kaggle competitions download -c $KaggleCompName -p $DownloadPath
unzip $DownloadPath/$KaggleCompName.zip -d $DownloadPath/$KaggleCompName