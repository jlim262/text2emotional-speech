
import pandas as pd

from sklearn.model_selection import train_test_split
from built.registry import Registry


# @Registry.register(category="splitter")
class TraintestSplitter(object):
    def __init__(self, csv_path, train_csv_path, test_csv_path, ratio=0.8, shuffle=True, random_state=42):
        self.csv_path = csv_path
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.ratio = ratio
        self.shuffle = shuffle
        self.random_state = random_state
        self.target_key = None

    def split(self):
        df = self.read_file()
            
        train, test = train_test_split(
            df, train_size=self.ratio, shuffle=self.shuffle, random_state=self.random_state, stratify=df[self.target_key])

        train.to_csv(self.train_csv_path, index=False)
        test.to_csv(self.test_csv_path, index=False)

    def read_file(self):
        pass


@Registry.register(category="splitter")
class SentimentDataSplitter(TraintestSplitter):
    def __init__(self, csv_path, train_csv_path, test_csv_path, ratio=0.8, shuffle=True, random_state=42):
        super().__init__(csv_path, train_csv_path,
                         test_csv_path, ratio, shuffle, random_state)
        self.target_key = 'sentiment'
        
    def read_file(self):
        return pd.read_csv(self.csv_path)


@Registry.register(category="splitter")
class EmotionDataSplitter(TraintestSplitter):
    def __init__(self, csv_path, train_csv_path, test_csv_path, ratio=0.8, shuffle=True, random_state=42):
        super().__init__(csv_path, train_csv_path,
                         test_csv_path, ratio, shuffle, random_state)
        self.target_key = 'Emotion'
        
    def read_file(self):
        return pd.read_csv(self.csv_path, sep='\t', encoding='utf-16')
    

@Registry.register(category="splitter")
class Emotion13DataSplitter(TraintestSplitter):
    def __init__(self, csv_path, train_csv_path, test_csv_path, ratio=0.8, shuffle=True, random_state=42):
        super().__init__(csv_path, train_csv_path,
                         test_csv_path, ratio, shuffle, random_state)
        self.target_key = 'Emotion'

    def read_file(self):
        return pd.read_csv(self.csv_path)
