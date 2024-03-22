import os
import json
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.interpolation import zoom
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src_path')
args = parser.parse_args()

DATA_SRC = os.path.join(args.src_path, 'amass/rawdata')
LABEL_SRC = os.path.join(args.src_path,  'babel_v1.0_release')
FPS = 30


if __name__ == '__main__':
    # Load action-idx correspondances
    with open('datasets/babel/action_label_2_idx.json', 'r') as f:
        action2idx = json.load(f)

    for split in ['train', 'val']:
        DEST120_H5 = os.path.join('datasets/babel', split +'120.h5')
        DEST60_H5 = os.path.join('datasets/babel', split +'60.h5')
        fw120 = h5py.File(DEST120_H5, 'w')
        fw60 = h5py.File(DEST60_H5, 'w')

        label_f = os.path.join(LABEL_SRC, split+'.json')
        with open(label_f, 'r') as f:
            annotations = json.load(f)
        # Traverse all sequences
        for babel_sid, seq in annotations.items():
            # Process motion data
            feat_p = seq['feat_p']
            feat_p = ('/').join(feat_p.split('/')[1:])
            # Load the motion data
            motion_f = os.path.join(DATA_SRC, feat_p)
            motions = np.load(motion_f)
            poses = motions['poses']
            n_frame = poses.shape[0]
            poses = np.concatenate((poses[:, :66], np.zeros((n_frame, 6))), axis=1)      # From SMPL-H to SMPL
            trans = motions['trans']
            fps = motions['mocap_framerate']

            # Rescale to a uniform FPS
            scale = FPS / fps
            trans = zoom(trans, [scale, 1])
            poses = zoom(poses, [scale, 1])
            n_frame = poses.shape[0]
            # Transform poses from axis-angle to 6D representations
            n_joint = poses.shape[1] // 3
            poses = poses.reshape((-1, n_joint, 3))
            poses = poses.reshape((-1, 3))
            rot = R.from_rotvec(poses)
            poses = rot.as_matrix().reshape((n_frame, n_joint, 3, 3))
            poses = poses[..., :2, :].reshape((-1, n_joint, 6))

            # Concat poses with translation
            trans = np.concatenate((trans, np.zeros((n_frame, 3))), 1).reshape(-1, 1, 6)
            poses = np.concatenate((poses, trans), 1)
            # (T, V, C) to (C, V, T)
            poses = np.transpose(poses, (2, 1, 0))

            # Parse the action instances
            if seq['frame_ann'] is not None:
                labels = seq['frame_ann']['labels']
                # Parse category label information
                for l, label in enumerate(labels):
                    act_cats = label['act_cat']
                    if act_cats is None:
                        continue
                    act_cats = np.unique(act_cats)
                    for act_cat in act_cats:
                        if act_cat not in action2idx.keys():
                            continue
                        action_id = action2idx[act_cat] + 1     # Make the action id start from 1
                        if action_id < 121:
                            seq_name = label['seg_id'] + '_l%02d'%(l) + '_A%03d'%(action_id)
                            start_t, end_t = label['start_t'], label['end_t']
                            start_frame = max(0, int(start_t*FPS))
                            end_frame = min(n_frame, int(end_t*FPS))
                            seq_poses = poses[..., start_frame:end_frame]
                            # # Eliminate too long or short actions
                            # if (seq_poses.shape[2] > 300) or (seq_poses.shape[2] < 10):
                            #     continue
                            if seq_poses.shape[2] < 1:
                                continue
                            # De-center the translation
                            seq_trans = seq_poses[:, n_joint]
                            seq_poses[:, n_joint] = seq_trans - np.mean(seq_trans, 1, keepdims=True)
                            fw120.create_dataset(seq_name, data=seq_poses, dtype='f4')
                            if action_id < 61:
                                fw60.create_dataset(seq_name, data=seq_poses, dtype='f4')
            else:
                labels = seq['seq_ann']['labels']
                seq_poses = poses
                # # Eliminate too long or short actions
                # if (seq_poses.shape[2] > 300) or (seq_poses.shape[2] < 10):
                #     continue
                if seq_poses.shape[2] < 1:
                    continue
                # De-center the translation
                seq_trans = seq_poses[:, n_joint]
                seq_poses[:, n_joint] = seq_trans - np.mean(seq_trans, 1, keepdims=True)
                # Parse category label information
                for l, label in enumerate(labels):
                    act_cats = label['act_cat']
                    if act_cats is None:
                        continue
                    act_cats = np.unique(act_cats)
                    for act_cat in act_cats:
                        if act_cat not in action2idx.keys():
                            continue
                        action_id = action2idx[act_cat] + 1     # Make the action id start from 1
                        if action_id < 121:
                            seq_name = label['seg_id'] + '_l%02d'%(l) + '_A%03d'%(action_id)
                            fw120.create_dataset(seq_name, data=seq_poses, dtype='f4')
                            if action_id < 61:
                                fw60.create_dataset(seq_name, data=seq_poses, dtype='f4')

        fw120.close()
        fw60.close()