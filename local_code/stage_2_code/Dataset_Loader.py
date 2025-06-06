'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pandas as pd
import torch


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name, header=None)
        data = df.values.astype('float32')
        X = data[:, 1:] / 255.0     # Normalize the data to [0, 1]
        y = data[:, 0].astype('long')
        # converted X and y to tensors
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return {'X': X, 'y': y}