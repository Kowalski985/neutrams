import os
import pickle

import numpy
import matplotlib.pyplot as plt

from net import FPNet, SparseNet, IntNet
from solver import SGDSolver, IntSGDSolver
import transfer


class Transformer(object):
    """Main class of the entire transform algorithm
    """

    def __init__(self, net, io_precision, w_precision, leak_precision, threshold_precision, crossbar_size):
        # type: (train.net.MLPNet, tuple, tuple, tuple, tuple, int) -> Transformer
        self.net = net
        self.directory = net.directory
        self.io_precision = io_precision
        self.w_precision = w_precision
        self.leak_precision = leak_precision
        self.threshold_precision = threshold_precision
        self.crossbar_size = crossbar_size
        data_path = os.path.join(self.directory, "data0")
        self.ks = []
        self.bs = []
        k, b = transfer.transfer_data(data_path, io_precision)
        self.ks.append(k)
        self.bs.append(b)

    def transform_fp(self, index, insert_layers, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                     snapshot):
        directory = os.path.join(self.directory, "layer%d" % index)
        in_directory = os.path.join(self.directory, "data%d" % (index - 1))
        out_directory = os.path.join(self.directory, "data%d" % (index))

        # generate low precision data of this layer
        transfer.transfer_data(out_directory, self.io_precision)

        # load data and train fp
        in_prefix = "low_precision" if index == 1 else "int"
        train_x = numpy.load(os.path.join(in_directory, "%s.train.npy" % in_prefix))
        valid_x = numpy.load(os.path.join(in_directory, "%s.valid.npy" % in_prefix))
        test_x = numpy.load(os.path.join(in_directory, "%s.test.npy" % in_prefix))
        out_prefix = "low_precision"
        train_y = numpy.load(os.path.join(out_directory, "%s.train.npy" % out_prefix))
        valid_y = numpy.load(os.path.join(out_directory, "%s.valid.npy" % out_prefix))
        test_y = numpy.load(os.path.join(out_directory, "%s.test.npy" % out_prefix))

        # load scale factor
        with open(os.path.join(in_directory, "kb.pkl")) as f:
            kx, bx = pickle.load(f)
        with open(os.path.join(out_directory, "kb.pkl")) as f:
            ky, by = pickle.load(f)

        fp_net = FPNet(directory, in_directory, out_directory, kx=kx, bx=bx, ky=ky, by=by) \
            if insert_layers is None or insert_layers == [] else \
            FPNet(directory, in_directory, out_directory, insert_layers, n_x=self.net.layers[index - 1].n_x,
                  n_y=self.net.layers[index - 1].n_y)
        fp_solver = SGDSolver(fp_net, batch_size, max_epoch, learning_rate, momentum, weight_decay, display, snapshot)
        fig_num = plt.figure().number
        plt.get_current_fig_manager().window.showMaximized()

        fp_solver.solve(train_set=(train_x, train_y), valid_set=(valid_x, valid_y), test_set=(test_x, test_y),
                        train_figure=fig_num, title="Floating point tuning")

        fp_net.save()

    def transform_sp(self, index, num_layers, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                     snapshot):
        directory = os.path.join(self.directory, "layer%d" % index)
        in_directory = os.path.join(self.directory, "data%d" % (index - 1))
        out_directory = os.path.join(self.directory, "data%d" % (index))

        # load data and train sp
        in_prefix = "low_precision" if index == 1 else "int"
        train_x = numpy.load(os.path.join(in_directory, "%s.train.npy" % in_prefix))
        valid_x = numpy.load(os.path.join(in_directory, "%s.valid.npy" % in_prefix))
        test_x = numpy.load(os.path.join(in_directory, "%s.test.npy" % in_prefix))
        out_prefix = "low_precision"
        train_y = numpy.load(os.path.join(out_directory, "%s.train.npy" % out_prefix))
        valid_y = numpy.load(os.path.join(out_directory, "%s.valid.npy" % out_prefix))
        test_y = numpy.load(os.path.join(out_directory, "%s.test.npy" % out_prefix))

        sp_net = SparseNet(directory, in_directory, out_directory, num_layers, self.crossbar_size)
        sp_solver = SGDSolver(sp_net, batch_size, max_epoch, learning_rate, momentum, weight_decay, display, snapshot)

        fig_num = plt.figure().number
        plt.get_current_fig_manager().window.showMaximized()

        sp_solver.solve(train_set=(train_x, train_y), valid_set=(valid_x, valid_y), test_set=(test_x, test_y),
                        train_figure=fig_num, title="Sparse tuning")

        sp_net.save()

    def transform_int(self, index, num_layers, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                      snapshot):
        directory = os.path.join(self.directory, "layer%d" % index)
        in_directory = os.path.join(self.directory, "data%d" % (index - 1))
        out_directory = os.path.join(self.directory, "data%d" % (index))

        # load data and train int
        in_prefix = "low_precision" if index == 1 else "int"
        train_x = numpy.load(os.path.join(in_directory, "%s.train.npy" % in_prefix))
        valid_x = numpy.load(os.path.join(in_directory, "%s.valid.npy" % in_prefix))
        test_x = numpy.load(os.path.join(in_directory, "%s.test.npy" % in_prefix))
        out_prefix = "low_precision"
        train_y = numpy.load(os.path.join(out_directory, "%s.train.npy" % out_prefix))
        valid_y = numpy.load(os.path.join(out_directory, "%s.valid.npy" % out_prefix))
        test_y = numpy.load(os.path.join(out_directory, "%s.test.npy" % out_prefix))

        int_net = IntNet(directory, in_directory, out_directory, num_layers, self.w_precision, self.leak_precision,
                         self.threshold_precision, self.io_precision)
        int_solver = IntSGDSolver(int_net, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                                  snapshot)

        fig_num = plt.figure().number
        plt.get_current_fig_manager().window.showMaximized()

        int_solver.solve(train_set=(train_x, train_y), valid_set=(valid_x, valid_y), test_set=(test_x, test_y),
                         train_figure=fig_num, title="Rounding tuning")

        int_net.save()

        int_net.save_output(train_x, "train")
        int_net.save_output(valid_x, "valid")
        int_net.save_output(test_x, "test")
