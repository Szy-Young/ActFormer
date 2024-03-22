import argparse
import itertools
import h5py
import numpy as np


training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                     58, 59, 70, 74, 78,80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                     93, 94, 95, 97, 98, 100, 103]
training_cameras = [2, 3]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', default='datasets/ntu120/data2p/ntu120.h5')
parser.add_argument('-o', '--out_path', default='datasets/ntu120/data2p')
parser.add_argument('--two_person', dest='two_person', default=False, action='store_true')
args = parser.parse_args()

IS_2P = args.two_person
class_bias = 0


def main(data_path, out_path, benchmark, split):
    fout = h5py.File(out_path, "w")
    with h5py.File(data_path, 'r') as f:
        sample_name = list(f.keys())
        for i, filename in enumerate(sample_name):
            action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])

            subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

            if benchmark == 'xview':
                istraining = (camera_id in training_cameras)
            elif benchmark == 'xsub':
                istraining = (subject_id in training_subjects)
            else:
                raise ValueError()

            if split == 'train':
                issample = istraining
            elif split == 'test':
                issample = not istraining
            else:
                raise ValueError()

            if issample:
                h5 = f[filename]
                filename = filename[:17] + '%03d'%(action_class-class_bias) + filename[20:]
                fout.create_dataset(filename, data=h5, dtype='f')

                if IS_2P:
                    # Add a flipped sample
                    h5_flip = np.zeros(h5.shape)
                    h5_flip[:3] = h5[3:]
                    h5_flip[3:] = h5[:3]
                    fout.create_dataset(filename+'-flip', data=h5_flip, dtype='f')
        print(f'Processed {benchmark}.{split}.')
    fout.close()


if __name__ == '__main__':
    for b, s in itertools.product(['xsub', 'xview'], ['train', 'test']):
        main(args.data_path, f'{args.out_path}/{b}.{s}.h5', b, s)
