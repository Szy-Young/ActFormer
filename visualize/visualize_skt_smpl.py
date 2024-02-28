import sys, os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

import torch
from rotation_utils import *
from smplx import SMPL


parser = argparse.ArgumentParser()
parser.add_argument('--src_path', help='the sample path')
parser.add_argument('--adj_file', default='ntu_adjacency.h5')
parser.add_argument('--save_path', default='gen_results')
parser.add_argument('--select_class', default=None)
parser.add_argument('--save_video', dest='save_video', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
args = parser.parse_args()


SMPL_MODEL_PATH = 'datasets/babel/smpl_models/SMPL_NEUTRAL.pkl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Different person with highly constrastive color groups
PERSON_COLORS = [
    [(112, 147, 210), (68, 114, 196), (55, 98, 175)],
    [(240, 148, 86), (237, 125, 49), (226, 103, 20)],
    [(197, 224, 180), (173, 211, 149), (152, 200, 122)],
    [(225, 204, 240), (205, 172, 230), (186, 139, 221)],
    [(255, 205, 205), (255, 167, 167), (255, 139, 139)],
]
PERSON_COLORS = np.array(PERSON_COLORS) / 255

HEAD_JOINTS = {
    'smpl': 15,
}


# Different joints / limbs on same person with similar colors.
# Use slight color difference to distinguish left/middle/right body parts.
JOINT_COLORS = {
    'smpl': [
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        1, 2, 1, 2, 1, 2,
    ],
}

LIMB_COLORS = {
    'smpl': [
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        1, 2, 1, 2, 1, 2,
    ],
}

# Config about visualization size
HEAD_RADIUS = 15
JOINT_RADIUS = 3
LIMB_WIDTH = 1.5


def set_ax(debug=True, viewpoint=(0, 0)):
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Set background color
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))

    # Set space range
    ax.set_xlim3d([-2, 2])
    ax.set_ylim3d([-2, 2])
    ax.set_zlim3d([-0.6, 0.6])

    # Set viewpoint
    ax.view_init(25, 45)

    # If debug, maintain grid lines
    if debug:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    # For better visualization effects, remove grid lines
    else:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_zticks([])

        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)
    return ax


def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        if out is not None:
            out.release()
        os._exit(0)


def draw_sample(data,
                parent_matrix,
                person_joint_colors,
                person_limb_colors,
                out_video=None,
                out_path=None,
                img_size=(400, 400),
                debug=True,
                head_radius=9,
                joint_radius=3,
                limb_width=1,
                head_joint=3):

    # Specify the output, priority to save into video
    if out_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video_writer = cv2.VideoWriter(out_video, fourcc, 15.0, img_size)
    # Save into frame-wise images
    elif out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        video_writer = None
    else:
        raise ValueError('Output not specified!')

    # Initialize plt figure
    plt.ion()
    fig = plt.figure(1, figsize=(2, 2), dpi=200)
    fig.canvas.mpl_connect('key_press_event', press)
    fig.clear()

    ax = set_ax(debug=debug, viewpoint=(25, 50))
    parent, child = np.where(parent_matrix==1)
    pair = list(zip(parent, child))

    # Set placeholders
    n_person = data.shape[0] // 3
    n_joints, n_frames = data.shape[1:]
    joints, limbs = [], []
    for n in range(n_person):
        limbs.append([
            ax.plot(np.zeros(2), np.zeros(2), np.zeros(2), color=person_limb_colors[n][l], linewidth=limb_width)[0]
            for l in range(len(parent))
        ])

        # Traverse all joints explicitly, since head needs special processing
        joints_p = []
        for j in range(n_joints):
            if j == head_joint:
                joints_p.append(ax.plot(np.zeros(1), np.zeros(1), np.zeros(1), '.', ms=head_radius, color=person_joint_colors[n][j])[0])
            else:
                joints_p.append(ax.plot(np.zeros(1), np.zeros(1), np.zeros(1), '.', ms=joint_radius, color=person_joint_colors[n][j])[0])
        joints.append(joints_p)

    # Draw animation
    for t in range(n_frames):
        for n in range(n_person):
            for j in range(n_joints):
                joints[n][j].set_data(data[(3*n):(3*n+2), j, t])
                joints[n][j].set_3d_properties(data[3*n+2, j, t])
            for l, (p, c) in enumerate(pair):
                limbs[n][l].set_data(data[(3*n):(3*n+2), [p, c], t])
                limbs[n][l].set_3d_properties(data[3*n+2, [p, c], t])
        fig.canvas.draw()

        if video_writer:
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(int(height), int(width), 3).copy()
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(image)
        else:
            output_file = os.path.join(out_path, '%06d.jpg'%(t))
            plt.savefig(output_file, bbox_inches='tight',pad_inches=0)


if __name__ == '__main__':
    SRC_PATH = args.src_path
    ADJ_FILE = args.adj_file
    SAVE_PATH = args.save_path
    SELECT_CLASS = args.select_class
    SAVE_VIDEO = args.save_video
    DEBUG = args.debug

    # Load skeleton topology information
    with h5py.File(ADJ_FILE, 'r') as f:
        parent = f['parent'][:]
        root = 0

    if parent.shape[0] == 25:
        DATASET = 'smpl'
        parent = parent[:24, :24]
    else:
        raise ValueError('Unrecognized skeleton topology')

    # Prepare colors
    head_joint = HEAD_JOINTS[DATASET]
    joint_color_idx = JOINT_COLORS[DATASET]
    limb_color_idx = LIMB_COLORS[DATASET]
    person_joint_colors, person_limb_colors = [], []
    for p in range(len(PERSON_COLORS)):
        person_joint_color = PERSON_COLORS[p][joint_color_idx]
        person_joint_colors.append(person_joint_color)
        person_limb_color = PERSON_COLORS[p][limb_color_idx]
        person_limb_colors.append(person_limb_color)

    # Create output path
    os.makedirs(SAVE_PATH, exist_ok=True)
    print('Saving visualized results to ', SAVE_PATH)

    # Load data and visualize
    with h5py.File(SRC_PATH, 'r') as f:
        samplenames = list(f.keys())

        for samplename in samplenames:
            # Parse the action category
            i = samplename.rfind('A')
            class_id = int(samplename[(i + 1):(i + 4)])
            if SELECT_CLASS is not None:
                if class_id not in SELECT_CLASS:
                    continue
            classname = samplename[i:(i + 4)]
            class_save_path = os.path.join(SAVE_PATH, classname)
            os.makedirs(class_save_path, exist_ok=True)

            # Parse the motion data
            data_all = f[samplename][:]
            num_p = data_all.shape[0] // 6
            abs_coor_all = []
            for p in range(num_p):
                data = data_all[(6 * p):(6 * p + 6)]

                # Smoothing the data along Time-axis
                data = gaussian_filter1d(data, sigma=1, axis=2)
                
                # Parse motion data
                poses, transl = data[:, :24], data[:3, 24]
                
                # Prepare SMPL models
                batch_size = poses.shape[2]
                smpl = SMPL(
                    model_path=SMPL_MODEL_PATH,
                    gender='NEUTRAL',
                    batch_size=batch_size).to(DEVICE)
                transl = np.transpose(transl, (1, 0))
                transl = torch.from_numpy(transl).to(DEVICE)

                # Transform poses from 6D representations to axis-angles
                poses = np.transpose(poses, (2, 1, 0)).reshape((-1, 6))
                poses = torch.from_numpy(poses).to(DEVICE)
                poses = rotation_6d_to_matrix(poses)
                poses = matrix_to_axis_angle(poses).view(batch_size, -1, 3).view(batch_size, -1)
                global_orient, body_pose = poses[:, :3], poses[:, 3:]

                # From SMPL params to joint positions
                with torch.no_grad():
                    output = smpl.forward(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        transl=transl,
                    )
                joints = output['joints'].detach().cpu().numpy()
                joints = np.transpose(joints, (2, 1, 0))[:, :24]
                abs_coor_all.append(joints)
            abs_coor_all = np.concatenate(abs_coor_all, axis=0)

            # Visualize
            if SAVE_VIDEO:
                draw_sample(abs_coor_all,
                            parent,
                            person_joint_colors,
                            person_limb_colors,
                            out_video=os.path.join(class_save_path, samplename + '.mp4'),
                            debug=DEBUG,
                            head_radius=HEAD_RADIUS,
                            joint_radius=JOINT_RADIUS,
                            limb_width=LIMB_WIDTH,
                            head_joint=head_joint)
            else:
                draw_sample(abs_coor_all,
                            parent,
                            person_joint_colors,
                            person_limb_colors,
                            out_path=os.path.join(class_save_path, samplename),
                            debug=DEBUG,
                            head_radius=HEAD_RADIUS,
                            joint_radius=JOINT_RADIUS,
                            limb_width=LIMB_WIDTH,
                            head_joint=head_joint)