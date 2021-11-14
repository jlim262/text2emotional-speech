
import numpy as np
import pandas as pd

from built.splitter import CsvSplitter
from built.registry import Registry


@Registry.register(category="splitter")
class KfoldSplitter(CsvSplitter):
    def __init__(self, csv_path, n_splits=5, shuffle=True, random_state=42):
        self.csv_path = csv_path
        self.train_data = pd.read_csv(csv_path)

        # num_bins = int(np.floor(1 + np.log2(len(self.train_data))))
        # self.train_data.loc[:, 'bins'] = pd.cut(
        #     self.train_data['target'], bins=num_bins, labels=False)

        targets = self.train_data['MGMT_value'].to_numpy()
        # bins = self.train_data.bins.to_numpy()

        super().__init__(self.train_data, targets, n_splits=n_splits,
                         shuffle=shuffle, random_state=random_state)

    def len(self):
        return len(self.train_data)
