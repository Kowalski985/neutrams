from compiler.ast import flatten
import logging
import os
import pickle

import itertools
import numpy
import theano
import matplotlib.pyplot as plt


def transfer_data(directory, io_precision):
    # type: (str, str, tuple) -> float, float
    """Transfer the data to io precision

    Args:
        directory: the directory of data
        io_precision: data precision in (low, high, step) form
    Returns:
        k, b: scale parameters
    """

    logging.info("Transfer %s to low precision" % directory)
    in_train_filename = os.path.join(directory, "train.npy")
    in_valid_filename = os.path.join(directory, "valid.npy")
    in_test_filename = os.path.join(directory, "test.npy")
    out_train_filename = os.path.join(directory, "low_precision.train.npy")
    out_valid_filename = os.path.join(directory, "low_precision.valid.npy")
    out_test_filename = os.path.join(directory, "low_precision.test.npy")
    train_data = numpy.load(in_train_filename)
    valid_data = numpy.load(in_valid_filename)
    test_data = numpy.load(in_test_filename)
    io_low, io_high, io_step = io_precision
    data_low = min([train_data.min(), valid_data.min(), test_data.min()])
    data_high = min([train_data.max(), valid_data.max(), test_data.max()])

    k = (io_high - io_low) / (data_high - data_low)
    b = (data_high * io_low - data_low * io_high) / (data_high - data_low)

    train_data = k * train_data + b
    valid_data = k * valid_data + b
    test_data = k * test_data + b
    train_data = (train_data / io_step).round() * io_step
    valid_data = (valid_data / io_step).round() * io_step
    test_data = (test_data / io_step).round() * io_step

    logging.info("Saving %s" % out_train_filename)
    numpy.save(out_train_filename, train_data)
    logging.info("Saving %s" % out_valid_filename)
    numpy.save(out_valid_filename, valid_data)
    logging.info("Saving %s" % out_test_filename)
    numpy.save(out_test_filename, test_data)

    kb_filename = os.path.join(directory, "kb.pkl")
    logging.info("Saving k, b to %s" % kb_filename)
    with open(kb_filename, "wb") as f:
        pickle.dump((k, b), f)
    return k, b

def sparse_weight(w, size):
    # type: (numpy.ndarray, int) -> list, numpy.ndarray
    rows, cols = w.shape
    n_group = int(max(numpy.ceil(rows / float(size)), numpy.ceil(cols / float(size))))
    row_indices = []
    col_indices = []
    for i in range(n_group):
        row_indices.append([j for j in range(rows * i / n_group, rows * (i + 1) / n_group)])
        col_indices.append([j for j in range(cols * i / n_group, cols * (i + 1) / n_group)])
    if n_group <= 1:
        return zip(row_indices, col_indices), numpy.ones(w.shape, dtype=theano.config.floatX)
    total_sum = numpy.sum(abs(w)).item()

    row_group_sum = [
        [
            sum(abs(w[row][col]) for col in col_indices[group])
            for group in range(n_group)
            ]
        for row in range(rows)
        ]
    col_group_sum = [
        [
            sum(abs(w[row][col]) for row in row_indices[group])
            for group in range(n_group)
            ]
        for col in range(cols)
        ]

    def find_max_gain(m_list):
        if type(m_list) is list:
            if len(m_list) == 0:
                return None, []
            max_gain_list = [find_max_gain(e) for e in m_list]
            this_index = numpy.argmax([v for v, idx in max_gain_list])
            value, indices = max_gain_list[this_index]
            return value, [this_index] + indices
        else:
            return m_list, []

    def get_group_sum():
        m_group_sum = 0
        for m_group in range(n_group):
            for m_row, m_col in itertools.product(row_indices[m_group], col_indices[m_group]):
                m_group_sum += abs(w[m_row][m_col])
        return m_group_sum

    def draw_matrix(figure_number):
        order_w = [
            [
                abs(w[m_row][m_col])
                for m_row in flatten(row_indices)
            ]
            for m_col in flatten(col_indices)
        ]
        plt.figure(figure_number)
        plt.clf()
        plt.title("Weight matrix sparsification")
        plt.imshow(order_w)
        plt.colorbar(orientation='horizontal')
        plt.pause(0.001)

    matrix_figure = plt.figure().number
    plt.get_current_fig_manager().window.showMaximized()

    iteration = 0

    while True:
        iteration += 1
        group_sum = get_group_sum()

        # gain if swap row_indices[right_group][right_row_index] with row_indices[left_group][left_row_index]
        swap_row_gain = [
            [
                [
                    [
                        row_group_sum[row_indices[left_group][left_row_index]][right_group] +
                        row_group_sum[row_indices[right_group][right_row_index]][left_group] -
                        row_group_sum[row_indices[left_group][left_row_index]][left_group] -
                        row_group_sum[row_indices[right_group][right_row_index]][right_group]
                        for left_row_index in range(len(row_indices[left_group]))
                        ]
                    for right_row_index in range(len(row_indices[right_group]))
                    ]
                for left_group in range(right_group)
                ]
            for right_group in range(n_group)
            ]

        # gain if swap col_indices[right_group][right_col_index] with col_indices[left_group][left_col_index]
        swap_col_gain = [
            [
                [
                    [
                        col_group_sum[col_indices[left_group][left_col_index]][right_group] +
                        col_group_sum[col_indices[right_group][right_col_index]][left_group] -
                        col_group_sum[col_indices[left_group][left_col_index]][left_group] -
                        col_group_sum[col_indices[right_group][right_col_index]][right_group]
                        for left_col_index in range(len(col_indices[left_group]))
                        ]
                    for right_col_index in range(len(col_indices[right_group]))
                    ]
                for left_group in range(right_group)
                ]
            for right_group in range(n_group)
            ]

        # gain if move row_indices[from_group][from_row_index] to row_indices[to_group]
        move_row_gain = [
            [
                [
                    len(row_indices[to_group]) < size and
                    row_group_sum[row_indices[from_group][from_row_index]][to_group] -
                    row_group_sum[row_indices[from_group][from_row_index]][from_group] or 0
                    for from_row_index in range(len(row_indices[from_group]))
                    ]
                for to_group in range(n_group)
                ]
            for from_group in range(n_group)
            ]

        # gain if move col_indices[from_group][from_col_index] to col_index[to_group]
        move_col_gain = [
            [
                [
                    len(col_indices[to_group]) < size and
                    col_group_sum[col_indices[from_group][from_col_index]][to_group] -
                    col_group_sum[col_indices[from_group][from_col_index]][from_group] or 0
                    for from_col_index in range(len(col_indices[from_group]))
                    ]
                for to_group in range(n_group)
                ]
            for from_group in range(n_group)
            ]

        # get the maximum gain and its indices (information)
        max_gain, op_info = find_max_gain([swap_row_gain, swap_col_gain, move_row_gain, move_col_gain])
        logging.info("iter %i, %03f/%03f ==> +%03f" % (iteration, group_sum, total_sum, max_gain))

        # if gain is less than 0, no more loop
        if max_gain <= 0:
            break

        op_type = op_info[0]
        op_params = op_info[1:]

        # if max gain is swap rows
        if op_type == 0:
            right_group = op_params[0]
            left_group = op_params[1]
            right_row_index = op_params[2]
            left_row_index = op_params[3]

            # apply the swap
            left_row_absolute_index = row_indices[left_group][left_row_index]
            right_row_absolute_index = row_indices[right_group][right_row_index]
            row_indices[left_group][left_row_index] = right_row_absolute_index
            row_indices[right_group][right_row_index] = left_row_absolute_index

            # update sum of cols
            for col in range(cols):
                col_group_sum[col][right_group] += \
                    abs(w[left_row_absolute_index][col]) - abs(w[right_row_absolute_index][col])
                col_group_sum[col][left_group] += \
                    abs(w[right_row_absolute_index][col]) - abs(w[left_row_absolute_index][col])

        # if max gain is swap cols
        elif op_type == 1:
            right_group = op_params[0]
            left_group = op_params[1]
            right_col_index = op_params[2]
            left_col_index = op_params[3]

            # apply the swap
            left_col_absolute_index = col_indices[left_group][left_col_index]
            right_col_absolute_index = col_indices[right_group][right_col_index]
            col_indices[left_group][left_col_index] = right_col_absolute_index
            col_indices[right_group][right_col_index] = left_col_absolute_index

            # update sum of rows
            for row in range(rows):
                row_group_sum[row][right_group] += \
                    abs(w[row][left_col_absolute_index]) - abs(w[row][right_col_absolute_index])
                row_group_sum[row][left_group] += \
                    abs(w[row][right_col_absolute_index]) - abs(w[row][left_col_absolute_index])

        # if max gain is move rows
        elif op_type == 2:
            from_group = op_params[0]
            to_group = op_params[1]
            from_row_index = op_params[2]

            # apply the move
            from_row_absolute_index = row_indices[from_group][from_row_index]
            row_indices[to_group].append(from_row_absolute_index)
            del row_indices[from_group][from_row_index]

            # update sum of cols
            for col in range(cols):
                col_group_sum[col][from_group] -= abs(w[from_row_absolute_index][col])
                col_group_sum[col][to_group] += abs(w[from_row_absolute_index][col])

        # if max gain is move cols
        elif op_type == 3:
            from_group = op_params[0]
            to_group = op_params[1]
            from_col_index = op_params[2]

            # apply the move
            from_col_absolute_index = col_indices[from_group][from_col_index]
            col_indices[to_group].append(from_col_absolute_index)
            del col_indices[from_group][from_col_index]

            # update sum of rows
            for row in range(rows):
                row_group_sum[row][from_group] -= abs(w[row][from_col_absolute_index])
                row_group_sum[row][to_group] += abs(w[row][from_col_absolute_index])
        draw_matrix(matrix_figure)
    w_mask = numpy.zeros(w.shape)
    for group in range(n_group):
        for row, col in itertools.product(row_indices[group], col_indices[group]):
            w_mask[row][col] = 1
    group_indices = zip(row_indices, col_indices)
    return group_indices, w_mask

def scale_parameters(w, leak, threshold, w_precision, leak_precision, threshold_precision):
    def get_scale_factor(param, precision):
        low, high, step = precision
        positive_param = numpy.max([param, numpy.zeros(param.shape)], axis=0)
        negative_param = numpy.min([param, numpy.zeros(param.shape)], axis=0)
        positive_param_scale_factor = numpy.where(positive_param > 0, high / positive_param, numpy.inf)
        negative_param_scale_factor = numpy.where(negative_param < 0, low / negative_param, numpy.inf)
        return numpy.min([positive_param_scale_factor, negative_param_scale_factor])
    w_scale_factor = numpy.min(get_scale_factor(w, w_precision), axis=0)
    leak_scale_factor = get_scale_factor(leak, leak_precision)
    threshold_scale_factor = get_scale_factor(threshold, threshold_precision)
    scale_factor = numpy.min([w_scale_factor, leak_scale_factor, threshold_scale_factor], axis=0)
    return w * scale_factor, leak * scale_factor, threshold * scale_factor,

def check_recognition_error(directory, prefix):
    reference_data = numpy.load(os.path.join(directory, "low_precision.%s.npy" % prefix))
    predict_data = numpy.load(os.path.join(directory, "int.%s.npy" % prefix))
    return numpy.not_equal(reference_data.argmax(axis=1), predict_data.argmax(axis=1)).mean()
