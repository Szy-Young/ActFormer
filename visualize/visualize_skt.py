import sys, os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', help='the sample path')
parser.add_argument('--adj_file', default='ntu_adjacency.h5')
parser.add_argument('--save_path', default='gen_results')
parser.add_argument('--select_class', default=None)
parser.add_argument('--save_video', dest='save_video', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
args = parser.parse_args()


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
    'ntu': 3,
    'gta': 15,
}


# Different joints / limbs on same person with similar colors.
# Use slight color difference to distinguish left/middle/right body parts.
JOINT_COLORS = {
    'ntu': [
        0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2,
        1, 1, 1, 1, 2, 2, 2, 2,
        0, 1, 1, 2, 2,
    ],
    'gta': [
        0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2,
        0, 0, 0, 0, 0,
        1, 1, 1, 2, 2, 2,
    ],
}

LIMB_COLORS = {
    'ntu': [
        0, 0, 1, 2, 0, 0,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        1, 1, 1,
        2, 2, 2,
        0, 1, 2,
    ],
    'gta': [
        0, 0, 1, 2,
        0, 0, 1, 2,
        1, 1, 1,
        2, 2, 2,
        0, 0, 0, 0,
        1, 1, 2, 2,
    ],
}

# Config about visualization size
HEAD_RADIUS = 15
JOINT_RADIUS = 3
LIMB_WIDTH = 1.5


def rel2abs(data, parent, root, avglength):
    assert data.shape[0] == 3
    assert data.shape[1] == parent.shape[0] == parent.shape[1]
    assert root < parent.shape[0]

    abs_data = np.zeros(data.shape)
    abs_data[:, 0] = data[:, 0]
    known_index = [root]
    while(known_index):
        k = known_index.pop()
        children = np.where(parent[k]==1)[0]
        for c in children:
            if c == root:
                continue
            # absolute coordinates = absolute parent + relative child * avglength
            abs_data[:, c, :] = abs_data[:, k, :] + data[:, c, :] * avglength[c]
            known_index.append(c)
    return abs_data


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
                joints[n][j].set_data(data[(3*n):(3*n+1), j, t], data[(3*n+1):(3*n+2), j, t])
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
    SELECT_CLASS = [int(args.select_class)]
    SAVE_VIDEO = args.save_video
    DEBUG = args.debug

    # Load skeleton topology information
    with h5py.File(ADJ_FILE, 'r') as f:
        parent = f['parent'][:]
        root = 0
        avglength = f['avglength'][:]

    if parent.shape[0] == 25:
        DATASET = 'ntu'
    elif parent.shape[0] == 22:
        DATASET = 'gta'
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
    print ('Saving visualized results to ', SAVE_PATH)

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

            # Parse the motion data
            data_all = f[samplename][:] # [6, 25, 74]
            num_p = data_all.shape[0] // 3
            abs_coor_all = []
            for p in range(num_p):
                data = data_all[(3 * p):(3 * p + 3)]
                # Smoothing the data along Time-axis
                data = gaussian_filter1d(data, sigma=1, axis=2)

                # Relative pose to absolute joint locations
                abs_coor = rel2abs(data, parent, root, avglength)

                # Exchange y-axis and z-axis if NTU
                if DATASET == 'ntu':
                    abs_coor = abs_coor[np.array([0, 2, 1], dtype=np.int64)]
                abs_coor_all.append(abs_coor)
            abs_coor_all = np.concatenate(abs_coor_all, axis=0)

            # Visualize
            if SAVE_VIDEO:
                draw_sample(abs_coor_all,
                            parent,
                            person_joint_colors,
                            person_limb_colors,
                            out_video=os.path.join(SAVE_PATH, samplename+'.mp4'),
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
                            out_path=os.path.join(SAVE_PATH, samplename),
                            debug=DEBUG,
                            head_radius=HEAD_RADIUS,
                            joint_radius=JOINT_RADIUS,
                            limb_width=LIMB_WIDTH,
                            head_joint=head_joint)
