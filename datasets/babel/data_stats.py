import h5py
import numpy as np
from matplotlib import pyplot as plt

DATA_FILES = ['datasets/babel/train120.h5', 'datasets/babel/val120.h5']


def load_data(data_files):
    samples = []
    for data_file in data_files:
        f = h5py.File(data_file, 'r')
        sample_names = list(f.keys())
        f_samples = [f[sample_name][:] for sample_name in sample_names]
        samples.extend(f_samples)
    return samples


def category_stats(data_file):
    count = 120 * [0]
    f = h5py.File(data_file, 'r')
    sample_names = list(f.keys())

    for sample_name in sample_names:
        i = sample_name.rfind('A')
        label = int(sample_name[i + 1:i + 4]) - 1
        count[label] += 1

    print (data_file)
    print (count)
    for c in range(120):
        if count[c] < 1:
            print (c)

    count = np.log2(np.array(count)+1)

    plt.clf()
    plt.bar(list(range(120)), count)
    plt.savefig(f'{data_file[:-3]}_category.png')


if __name__ == '__main__':

    # Statistics for samples in different categories
    for data_file in DATA_FILES:
        category_stats(data_file)

    # samples = load_data(DATA_FILES)
    #
    # # Statistics for sequence length
    # print (len(samples))
    # samples = [sample for sample in samples if sample.shape[2]<300]
    # print (len(samples))
    # seq_lens = [sample.shape[2] for sample in samples]
    # fig = plt.hist(seq_lens, bins=50)
    # plt.savefig('datasets/babel/seq_lens_120.png')
    #
    # # Statistics for root joint positions
    # roots = np.concatenate(samples, axis=2)[:, 0]
    # roots = np.concatenate(np.split(roots, 2, axis=0), axis=-1)
    # for ax in range(3):
    #     fig = plt.hist(roots[ax])
    #     plt.savefig(f'datasets/babel/stats_rootpos_axis{ax}.png')