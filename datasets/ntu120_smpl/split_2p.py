import os
import argparse
import itertools
import h5py
import numpy as np

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35,38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                     58, 59, 70, 74, 78,80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                     93, 94, 95, 97, 98, 100, 103]
training_cameras = [2, 3]

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', default='datasets/ntu120_smpl/ntu_2p.h5')
parser.add_argument('-o', '--out_path', default='datasets/ntu120_smpl/data_2p')
arg = parser.parse_args()

selected_class = list(range(50, 61)) + list(range(106, 121)) # 2p
def main(data_path, out_path, benchmark, split):
    fout = h5py.File(out_path, "w")
    with h5py.File(data_path, 'r') as f:
        sample_name = list(f.keys())
        for i, filename in enumerate(sample_name):
            action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
            if action_class not in selected_class:
                continue

            subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])

            if benchmark == 'xsub':
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
                # Make class IDs to be consistent (for single-person classes)
                if action_class < 61:
                    action_class -= 49
                else:
                    action_class -= 94
                poses = f[filename]
                filename = filename[:17] + '%03d'%(action_class) + filename[20:]
                fout.create_dataset(filename, data=poses, dtype='f')
                # # Add a flipped sample
                # h5_flip = np.zeros(h5.shape)
                # h5_flip[:3] = h5[3:]
                # h5_flip[3:] = h5[:3]
                # fout.create_dataset(filename+'-flip', data=h5_flip, dtype='f')
            # print(f'Processing {benchmark}.{split}.{i}.')
    fout.close()


if __name__ == '__main__':
    os.makedirs(arg.out_path, exist_ok=True)
    # Only cross-subject
    for b, s in itertools.product(['xsub'], ['train', 'test']):
        main(arg.data_path, f'{arg.out_path}/{b}.{s}.h5', b, s)