import h5py
import numpy as np
import torch
from . import tools


class Feeder(torch.utils.data.Dataset):
    """
    Arguments:
        path: the path to '.h5' data, the shape of data should be C(channel), V(vertex), T(time).
        C: output channel (for sanity check only)
        V: output vertex (for sanity check only)
        T: output time
        random_crop: if False, then the sequence will start at time 0.

    Iter Returns:
        data: with shape (C, V, T)
    """

    def __init__(self, path, C, V, T, random_crop=False, dataset='ntu'):
        self.path = path
        self.C = C
        self.V = V
        self.T = T
        self.random_crop = random_crop
        self.dataset = dataset
        self.load_data()


    def load_data(self):
        self.data = {}
        self.label = {}

        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
            for k in self.keys:
                self.data[k] = f[k][:].astype('float32')

                # get label
                if self.dataset == 'ntu':
                    if 'A' in k:
                        i = k.rfind('A')
                        self.label[k] = int(k[i + 1:i + 4]) - 1
                    else:
                        self.label[k] = -1
                elif self.dataset == 'gta':
                    self.label[k] = 0
                else:
                    self.label[k] = -1
                assert self.data[k].shape[:2] == (self.C, self.V)
        self.N = len(self.keys)


    def __len__(self):
        return self.N


    def __getitem__(self, index):
        index = index % self.N
        # get data (C, V, T)
        data_numpy = self.data[self.keys[index]]

        data_numpy = tools.frame_snake(data_numpy, self.T, self.random_crop)
        if not (data_numpy.shape == (self.C, self.V, self.T)):
            print(data_numpy.shape, self.keys[index])
        assert data_numpy.shape == (self.C, self.V, self.T)

        return data_numpy, self.label[self.keys[index]]