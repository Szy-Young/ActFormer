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

    Iter Returns:
        data: with shape (C, V, T, 1)
    """

    def __init__(self, path, C, V, T, dataset='ntu', frame_offset=-1):
        self.path = path
        self.C = C
        self.V = V
        self.T = T
        self.dataset = dataset
        self.frame_offset = frame_offset
        self.load_data()


    def load_data(self):
        self.data = []

        with h5py.File(self.path, 'r') as f:
            self.keys = list(f.keys())
            for k in self.keys:
                # get label
                if self.dataset == 'ntu':
                    if 'A' in k:
                        i = k.rfind('A')
                        label = int(k[i + 1:i + 4]) - 1
                    else:
                        label = -1
                elif self.dataset == 'gta':
                    label = 0
                else:
                    label = -1
                data = f[k][:]
                assert data.shape[:2] == (self.C, self.V)

                if self.frame_offset < 0:
                    # Cut long sequences into several fixed-length samples
                    for f_offset in range(0, data.shape[2] - self.T, self.T):
                        pos = (k, f_offset, f_offset + self.T, label)
                        self.data.append(pos)
                else:
                    pos = (k, self.frame_offset, self.frame_offset + self.T, label)
                    self.data.append(pos)

        self.h5 = h5py.File(self.path, 'r')


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        name, f_beg, f_end, label = self.data[index]
        data_numpy = self.h5[name][:, :, f_beg:f_end]

        data_numpy = data_numpy.reshape(self.C, self.V, self.T, 1)
        data_numpy = data_numpy.transpose(0, 2, 1, 3)
        return data_numpy, label