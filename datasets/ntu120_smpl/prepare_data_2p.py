import os
import pickle
import h5py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('./')
import utils.rotation_conversions as geometry
from utils.misc import to_torch

SRC = 'datasets/NTU120_Mocap/motions' # set to your path
DEST_H5 = 'datasets/ntu120_smpl/ntu_2p.h5'
LOAD_PERSON_NUM = 2
TWO_PERSON_CLASS = list(range(50, 61)) + list(range(106, 121))

def get_rotation(view):
    theta = - view * np.pi/4
    axis = torch.tensor([1, 0, 0], dtype=torch.float)
    axisangle = theta*axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix

if __name__ == '__main__':
    fw = h5py.File(DEST_H5, 'w')
    # Load all samples
    rotation = get_rotation(4)
    for setup in range(1, 33):
        setup_src = os.path.join(SRC, 'S%03d'%(setup))
        motion_files = sorted(os.listdir(setup_src))
        seq_names = sorted(list(set([motion_file[:-10] for motion_file in motion_files])))
        for seq_name in seq_names:
            # Filter out action classes not needed
            action_class = int(seq_name[17:20])
            if LOAD_PERSON_NUM == 1:
                load = (action_class not in TWO_PERSON_CLASS)
            else:
                load = (action_class in TWO_PERSON_CLASS)
            if not load:
                continue
            # Load SMPL params
            pose_seq = []
            scalings = []
            transls = []
            for body_id in range(LOAD_PERSON_NUM):
                motion_file = seq_name + f'_body{body_id}.pkl'
                with open(os.path.join(setup_src, motion_file), 'rb') as f:
                    smpl_params = pickle.load(f)
                poses = smpl_params['smpl_poses']
                trans = smpl_params['smpl_trans']
                scaling = smpl_params['smpl_scaling']
                n_frame = poses.shape[0]
                n_joint = poses.shape[1] // 3

                poses = poses.reshape((-1, n_joint, 3)) # [T, V, C]
                global_matrix = geometry.axis_angle_to_matrix(torch.from_numpy(poses[:, 0]))
                poses[:, 0] = geometry.matrix_to_axis_angle(rotation @ global_matrix).numpy()
                # To rot6d
                poses = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(to_torch(poses)))

                # Rescale the translation
                trans = trans / scaling
                # De-center the translation
                # trans = trans - np.mean(trans, 0)
                pose_seq.append(poses)
                transls.append(trans)
                scalings.append(scaling)

            # De-center the translation
            mean_trans = np.mean(np.concatenate(transls, 0), 0)
            transls[0] = transls[0] - mean_trans
            transls[1] = transls[1] - mean_trans

            # Concat poses with translation
            poses = to_torch(np.concatenate(pose_seq, -1))

            padded_tr = torch.zeros((poses.shape[0], 1, poses.shape[2]), dtype=poses.dtype)
            padded_tr[:,:,:3] = to_torch(transls[0]).reshape(-1, 1, 3)
            padded_tr[:,:,6:9] = to_torch(transls[1]).reshape(-1, 1, 3)
            poses = np.concatenate((poses.numpy(), padded_tr.numpy()), 1)

            # # (T, V, C) to (C, V, T)
            poses = np.transpose(poses, (2, 1, 0))
            fw.create_dataset(seq_name, data=poses, dtype='f4')
