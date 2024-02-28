import random
import numpy as np

from utils.torchlight import import_class


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose((0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return data_numpy
    # C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    # begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    _, T, _, _ = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=None,
                scale_candidate=None,
                transform_candidate=None,
                move_time_candidate=None):
    if angle_candidate is None:
        angle_candidate = [-10., -5., 0., 5., 10.]
    if scale_candidate is None:
        scale_candidate = [0.9, 1.0, 1.1]
    if transform_candidate is None:
        transform_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]
    if move_time_candidate is None:
        move_time_candidate = [1]

    # input: C,T,V,M
    _, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def abs2ref_hard(data_numpy, graph='producer.net.st_gcn_unit.graph.ntu_rgb_d.Graph'):
    # input: C,T,V,M
    C, T, _, M = data_numpy.shape
    inward = import_class(graph)().inward
    data_numpy_out = np.zeros((C, T, len(inward), M))
    for i, (n1, n2) in enumerate(inward):
        data_numpy_out[:, :, i, :] = data_numpy[:, :, n1, :] - data_numpy[:, :, n2, :]
    return data_numpy_out


def ref2abs_hard(data_numpy, graph='st_gcn_unit.graph.ntu_rgb_d.Graph'):
    # input: C,T,E,M
    C, T, _, M = data_numpy.shape
    g = import_class(graph)()
    inward = g.inward
    data_numpy_out = np.zeros((C, T, g.num_node, M))

    # the node not the root
    unroot = set([n1 for (n1, n2) in inward])
    # has been recovered
    is_recovered = [n not in unroot for n in range(g.num_node)]

    for _ in range(g.num_node):
        # can not be recovered temply
        is_dirty = [False] * g.num_node
        for (n1, n2) in inward:
            if not is_recovered[n1]:
                if not is_recovered[n2]:
                    is_dirty[n1] = True
        for i, (n1, n2) in enumerate(inward):
            if not is_dirty[n1]:
                data_numpy_out[:, :, n1, :] = data_numpy_out[:, :, n2, :] + data_numpy[:, :, i, :]
                is_recovered[n1] = True
    return data_numpy_out


def abs2ref(data_numpy, graph='producer.net.st_gcn_unit.graph.ntu_rgb_d.Graph'):
    graph = import_class(graph)()
    R = graph.get_abs2ref_matrix()
    return data_numpy.transpose((0, 3, 1, 2)).dot(R).transpose((0, 2, 3, 1))


def ref2abs(data_numpy, graph='producer.net.st_gcn_unit.graph.ntu_rgb_d.Graph'):
    graph = import_class(graph)()
    iR = graph.get_ref2abs_matrix()
    return data_numpy.transpose((0, 3, 1, 2)).dot(iR).transpose((0, 2, 3, 1))


def centralize(data_numpy):
    # input: C,T,V,M
    # C, T, V, M = data_numpy.shape
    data_numpy = data_numpy - data_numpy.mean(axis=2, keepdims=True)
    return data_numpy


def sequence_centralize(data_numpy):
    # input: C,T,V,M
    # C, T, V, M = data_numpy.shape
    data_numpy = data_numpy - data_numpy.mean(axis=2, keepdims=True).mean(axis=1, keepdims=True)
    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    _, T, _, _ = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def frame_snake(data_numpy, TT=256, random_crop=False):
    """
    input C, V, T
    output: make T be TT by snake repeating on the time axis.
    """
    _, _, T = data_numpy.shape

    if T == 1:
        return data_numpy.repeat(TT, axis=-1)
    if T >= TT:
        if random_crop:
            return data_numpy[:, :, random.randint(0, T - TT):][:, :, :TT]
        return data_numpy[:, :, :TT]

    selected_index = np.arange(T)
    selected_index = np.concatenate((selected_index, selected_index[1:-1][::-1]))
    selected_index = np.tile(selected_index, TT // (2 * T - 2) + 1)[:TT]

    return data_numpy[:, :, selected_index]


def frame_snake_actor(data_numpy, ret_tr, TT=256, random_crop=False):
    """
    data_numpy : T, V, C
    ret_tr : T, C
    output: make T be TT by snake repeating on the time axis.
    """
    T, _, _ = data_numpy.shape

    if T == 1:
        return data_numpy.repeat(TT, axis=0), ret_tr.repeat(TT, axis=0)
    if T >= TT:
        if random_crop:
            randi = random.randint(0, T - TT)
            return data_numpy[randi:, :, :][:TT, :, :], ret_tr[randi:, :][:TT, :]
        return data_numpy[:TT, :, :], ret_tr[:TT, :]

    selected_index = np.arange(T)
    selected_index = np.concatenate((selected_index, selected_index[1:-1][::-1]))
    selected_index = np.tile(selected_index, TT // (2 * T - 2) + 1)[:TT]

    return data_numpy[selected_index, :, :], ret_tr[selected_index, :]


def frame_snake_shift(data_numpy, TT=256, random_crop=False, seed=None):
    """
    input C, V, T
    output: make T be TT by snake repeating on the time axis.
    """
    _, _, T = data_numpy.shape

    selected_index = np.arange(T)
    selected_index = np.concatenate((selected_index, selected_index[1:-1][::-1]))
    selected_index = np.tile(selected_index, TT // (2 * T - 2) + 2)

    if seed is not None:
        random.seed(seed)

    if random_crop:
        selected_index = selected_index[random.randint(0, 2 * T - 2):][:TT]
    else:
        selected_index = selected_index[:TT]

    return data_numpy[:, :, selected_index]


def frame_repeat(data_numpy, flip=True):
    # input: C,T,V,M
    _, T, _, _ = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    valid_index = np.arange(begin, end)
    if flip:
        valid_cell = np.concatenate((valid_index, valid_index[1:-1][::-1]))
    else:
        valid_cell = valid_index

    selected_index = np.tile(valid_cell, T // len(valid_cell) + 1)[0:T]

    return data_numpy[:, selected_index, :, :]


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert C == 3
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert np.all(forward_map >= 0)

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
