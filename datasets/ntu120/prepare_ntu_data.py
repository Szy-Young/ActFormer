"""
Prepare the NTU-RGBD skeleton data.
Missing detections are filled with relatively reasonable values.
"""
import argparse
import os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src_path', default='./datasets/NTU120_RGBD/')
args = parser.parse_args()

SRC_PATHS = []
for path in ['skeletons']:
    SRC_PATHS.append(os.path.join(args.src_path, path))
MISSING_RECORD_FILE = os.path.join(args.src_path, 'ntu_rgb120_missings.txt')

os.makedirs('datasets/ntu120/data1p', exist_ok=True)
os.makedirs('datasets/ntu120/data2p', exist_ok=True)
DEST_H5_1P = 'datasets/ntu120/data1p/ntu120.h5'
DEST_H5_2P = 'datasets/ntu120/data2p/ntu120.h5'

N_JOINT = 25
C_JOINT = 3
TWO_PERSON_CLASS = list(range(50, 61)) + list(range(106, 121))
LIMBS = [
    (2, 3), (20, 2), (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (1, 20), (0, 1), (0, 16), (16, 17), (17, 18), (18, 19), (0, 12), (12, 13),
    (13, 14), (14, 15), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22)
] # [(parent, child), ...]


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for _ in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for _ in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: str(v)
                    for k, v in zip(body_info_key,
                                    f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for _ in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key,
                                        f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def fill_seq(seq_info, max_body=None):
    n_frame = seq_info['numFrame']
    body_info = {}
    filled_seq = {}
    
    for t, frame in enumerate(seq_info['frameInfo']):
        for n, body in enumerate(frame['bodyInfo']):
            # Load the detected skeleton
            skt = np.zeros((N_JOINT, C_JOINT))
            for j, v in enumerate(body['jointInfo']):
                if j < N_JOINT:
                    skt[j, :] = np.array([v['x'], v['y'], v['z']])
            # First frame for a body detection
            body_id = body['bodyID']
            if body_id not in body_info.keys():
                # Initialize the record
                body_info[body_id] = {'last_det': 0, 'total_det': 0}
                filled_seq[body_id] = np.zeros((n_frame, N_JOINT, C_JOINT))
                # Fill the blank before
                filled_seq[body_id][:t] = np.tile(skt, (t, 1, 1))
            # Fill the blank between last detected frame and this frame by linear interpolation
            last_t = body_info[body_id]['last_det']
            if last_t < (t-1):
                last_skt = filled_seq[body_id][last_t]
                for mid_t in (last_t+1, t):
                    filled_seq[body_id][mid_t] = (mid_t-last_t)/(t-last_t) * skt + (t-mid_t)/(t-last_t) * last_skt
            # Update
            body_info[body_id]['last_det'] = t
            body_info[body_id]['total_det'] += 1
            filled_seq[body_id][t] = skt
            
    # Fill the blank at the end
    for body_id in body_info.keys():
        last_t = body_info[body_id]['last_det']
        if last_t < (n_frame-1):
            last_skt = filled_seq[body_id][last_t]
            filled_seq[body_id][(last_t+1):n_frame] = np.tile(last_skt, (n_frame-1-last_t, 1, 1))

    # Filter out possibly abundant detections
    body_ids = list(body_info.keys())
    if max_body and (len(body_ids) > max_body):
        body_ids = sorted(body_ids, key=lambda bid:body_info[bid]['total_det'], reverse=True)
        for body_id in body_ids[max_body:]:
            del filled_seq[body_id]
    return filled_seq


def smooth_seq(seqs):
    for body_id, seq in seqs.items():
        seqs[body_id] = gaussian_filter1d(seq, sigma=3, axis=0)
    return seqs


def normalize_seq(seqs):
    # Transform absolute joint positions to relative limb vectors and normalize
    roots = []
    for body_id, seq in seqs.items():
        normed_seq = np.zeros_like(seq)
        # Keep the absolute positions of root joint
        normed_seq[:, 0] = seq[:, 0]
        roots.append(seq[:, 0])
        for parent, child in LIMBS:
            limb = seq[:, child] - seq[:, parent]
            normed_seq[:, child] = limb / (np.linalg.norm(limb, axis=-1, keepdims=True) + 1e-4)
        seqs[body_id] = normed_seq
    # Move the motions to be around a base position
    base = np.array([0.0, 0.0, 0.0])
    roots = np.concatenate(roots, axis=0)
    root_bias = np.mean(roots, axis=0) - base
    for body_id, seq in seqs.items():
        seq[:, 0] -= root_bias
        seqs[body_id] = seq

    return seqs


def dict_to_array(seqs):
    seqs = [seq for seq in seqs.values()]
    seqs = np.concatenate(seqs, axis=-1)
    # (T, K, N*3) to (N*3, K, T)
    seqs = np.transpose(seqs, (2, 1, 0))
    return seqs


if __name__ == '__main__':
    all_files = []
    for src_path in SRC_PATHS:
        path_files = sorted(os.listdir(src_path))
        path_files = [os.path.join(src_path, path_file) for path_file in path_files]
        all_files.extend(path_files)

    with open(MISSING_RECORD_FILE, 'r') as f:
        missing_info = f.readlines()
    missing_info = [file_name[:-1] for file_name in missing_info]


    # Load all data
    fout_h5_1p = h5py.File(DEST_H5_1P, 'w')
    fout_h5_2p = h5py.File(DEST_H5_2P, 'w')
    for f in all_files:
        file_name = f.split('/')[-1][:-9]

        # Check if the file is marked as missing
        if file_name in missing_info:
            continue
        seq_info = read_skeleton(f)
        act_class = int(file_name[17:20])

        # Process the data to reduce noise
        if act_class in TWO_PERSON_CLASS:
            seqs = fill_seq(seq_info, max_body=2)
            if len(seqs) < 2:
                continue
        else:
            seqs = fill_seq(seq_info, max_body=1)
            if len(seqs) < 1:
                continue

        # Smooth and transform into normalized limb vector representation
        seqs = smooth_seq(seqs)
        seqs = normalize_seq(seqs)
        seqs = dict_to_array(seqs)

        # Save
        if act_class in TWO_PERSON_CLASS:
            # re-arrange class orders
            if act_class < 106:
                sample_name = file_name[:17] + '%03d'%(act_class-49)
            else:
                sample_name = file_name[:17] + '%03d'%(act_class-94)
            fout_h5_2p.create_dataset(sample_name, data=seqs, dtype='f4')
        else:
            # re-arrange class orders
            if act_class < 61:
                sample_name = file_name
            else:
                sample_name = file_name[:17] + '%03d'%(act_class-11)
            fout_h5_1p.create_dataset(sample_name, data=seqs, dtype='f4')