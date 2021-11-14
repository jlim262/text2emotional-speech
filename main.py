from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import torch
import pandas as pd
import datetime
import numpy as np
import wandb

from sklearn import metrics
from pathlib import Path
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from built.trainer_base import TrainerBase
from built.inference import Inference
from built.builder import Builder
from built.checkpoint_manager import CheckpointManager
from built.ensembler import Ensembler
from built.utils.util_functions import *
from built.utils.kaggle_helper import dataset_initialize
from built.registry import Registry

from text2emospch.src.splitter.traintest_splitter import SentimentDataSplitter, EmotionDataSplitter

ex = Experiment('text2emotion-speech')
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    description = 'Tweet Sentiment Classification'


@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def split(_run, _config):
    config = edict(_config)
    config = replace_placeholder(config, config)
    print(
        f'Split dataset into train and test by {config.splitter.params.ratio} ratio')

    builder = Builder()
    splitter = builder.build_splitter(config)
    splitter.split()


@ex.command
def train(_run, _config):
    config = edict(_config)
    config = replace_placeholder(config, config)
    pprint.PrettyPrinter(indent=2).pprint(config)

    run = wandb.init(project=config.wandb.project.name,
                     group=config.wandb.group.name, reinit=True)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': True, 'split': 'train', 'csv_path': config.splitter.params.train_csv_path})
    config.dataset.splits.append(
        {'train': False, 'split': 'test', 'csv_path': config.splitter.params.test_csv_path})

    if not os.path.exists(config.train.dir):
        os.makedirs(config.train.dir)

    builder = Builder()
    trainer = TrainerBase(config, builder, run)
    score = trainer.run()

    print(f'score: {score}')
    run.finish()


@ex.command
def inference_sentiment(_run, _config):
    config = edict(_config)
    config = replace_placeholder(config, config)
    # test_path = 'text2emospch/input/tweet-sentiment-extraction/test.csv'
    test_path = 'text2emospch/input/tweet-sentiment-extraction/scenario.csv'

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': False, 'split': 'test', 'csv_path': test_path})
    config.dataset.params.inference = True

    builder = Builder()
    model_path = os.path.join(config.train.dir, config.train.name)
    inferencer = Inference(config, builder, model_path)
    outputs = inferencer.predict()

    # outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    outputs = np.argmax(outputs, axis=1)

    test_df = pd.read_csv(test_path)

    # This should be same as TweetSentimentDataset > self.sentiment2target
    sentiment2target = {'neutral': 0, 'positive': 1, 'negative': 2}
    target2sentiment = dict(
        [(value, key) for key, value in sentiment2target.items()])
    

    # targets = []
    # for t in test_df.sentiment.to_numpy():
    #     targets.append(sentiment2target[t])
    # accuracy = metrics.accuracy_score(targets, outputs)
    # print(f'Accuracy: {accuracy}')
    predictions = []
    for o in outputs:
        predictions.append(target2sentiment[o])

    # accuracy = metrics.accuracy_score(targets, outputs)
    # print(f'Accuracy: {accuracy}')

    id = test_df.textID.to_numpy()
    text = test_df.text
    data = np.stack((id, text, predictions), axis=-1)
    df = pd.DataFrame(data, columns=["id", "text", "predict"])
    df.to_csv('sentiment_prediction.csv', index=False)
    # id = test_df.id.to_numpy()
    # data = np.stack((id, target), axis=-1)
    # df = pd.DataFrame(data, columns=["id", "target"])
    # df.to_csv(os.path.join(config.train.dir,
    #                        'inference_result_avg.csv'), index=False)


@ex.command
def inference_emotion(_run, _config):
    config = edict(_config)
    config = replace_placeholder(config, config)
    test_path = 'text2emospch/input/tweet-emotion-data/scenario2.csv'

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': False, 'split': 'test', 'csv_path': test_path})
    config.dataset.params.inference = True

    builder = Builder()
    model_path = os.path.join(config.train.dir, config.train.name)
    inferencer = Inference(config, builder, model_path)
    outputs = inferencer.predict()

    # outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    outputs = np.argmax(outputs, axis=1)
    test_df = pd.read_csv(test_path)

    # This should be same as TweetSentimentDataset > self.sentiment2target
    emotion2target = {'anger': 0, 'joy': 1,
                      'disgust': 2, 'fear': 3, 'sadness': 4, 'surprise': 5}
    target2emotion = dict(
        [(value, key) for key, value in emotion2target.items()])
    # targets = []
    # for t in test_df.Emotion.to_numpy():
    #     targets.append(emotion2target[t])
    predictions = []
    for o in outputs:
        predictions.append(target2emotion[o])

    # accuracy = metrics.accuracy_score(targets, outputs)
    # print(f'Accuracy: {accuracy}')

    id = test_df.id.to_numpy()
    data = np.stack((id, predictions), axis=-1)
    df = pd.DataFrame(data, columns=["id", "predict"])
    df.to_csv('emotion_prediction2.csv', index=False)

@ex.command
def train_kfold(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)

    if 'use_date' in config and config['use_date'] is True:
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d-%H%M%S")
        config.train.dir = os.path.join(config.train.dir, now)

    builder = Builder()
    splitter = builder.build_splitter(config)
    df = pd.read_csv(splitter.csv_path)

    if not os.path.exists(config.train.dir):
        os.makedirs(config.train.dir)

    for i_fold in range(splitter.n_splits):
        run = wandb.init(
            project=config.wandb.project.name, group=config.wandb.group.name, reinit=True)

        print(f'Training start: {i_fold} fold')
        train_idx, val_idx = splitter.get_fold(i_fold)

        train_df = df.iloc[train_idx]
        train_csv_path = os.path.join(
            config.train.dir, str(i_fold) + '_train.csv')
        train_df.to_csv(train_csv_path)

        val_df = df.iloc[val_idx]
        val_csv_path = os.path.join(config.train.dir, str(i_fold) + '_val.csv')
        val_df.to_csv(val_csv_path)
        config.dataset.splits = []
        config.dataset.splits.append(
            {'train': True, 'split': 'train', 'csv_path': train_csv_path})
        config.dataset.splits.append(
            {'train': False, 'split': 'val', 'csv_path': val_csv_path})
        config.dataset.splits.append(
            {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test.csv'})
        config.train.name = str(i_fold) + '_fold'

        tr = TrainerBase(config, builder, run)
        tr.run()
        print(f'Training end\n')
        run.finish()


@ex.command
def cam_sentiment(_run, _config):
    config = edict(_config)

    checkpoint_path = os.path.join(config.train.dir, 'checkpoint')

    cm = CheckpointManager(checkpoint_path)
    ckpt = cm.latest()
    builder = Builder()
    run = wandb.init(
        project=f'cam_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)

    tr = TrainerBase(config, builder, run)

    last_epoch, step, last_accuracy = cm.load(tr.model, tr.optimizer, ckpt)
    print(f'{checkpoint_path}:{last_epoch} , {last_accuracy}')

    output, targets = tr.forward()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    ex.run_commandline()
