import logging
import os
import pickle

import numpy
import theano.tensor as tensor

from layer import FPLayer, SparseLayer, IntLayer, LossLayer
import transfer


class FPNet(object):
    """Float-point network for transforming one layer in reference network
    """

    def __init__(self, directory, in_directory, out_directory, insert_layers=None, kx=1, bx=0, ky=1, by=0, n_x=-1,
                 n_y=-1):
        # type: (str, str, str, list, float, float) -> FPNet
        """

        Args:
            directory: Directory of the layer to be transformed.
            in_directory: Directory of input data
            out_directory: Directory of output data
            insert_layers: List of hidden layers' sizes. If no hidden layer is inserted, set this parameter to None
            kx: scale parameters of input layer (used when insert_layers is None or [])
            bx: scale parameters of input layer (used when insert_layers is None or [])
            ky: scale parameters of output layer (used when insert_layers is None or [])
            by: scale parameters of output layer (used when insert_layers is None or [])
            n_x: network input size (used when insert_layers is not None or [])
            n_y: network output size (used when insert_layers is None or [])
        """

        # set and checking directories
        self.directory = directory
        self.in_directory = in_directory
        self.out_directory = out_directory
        if not os.path.exists(directory):
            logging.error("Could not find directory %s" % directory)
        if not os.path.exists(in_directory):
            logging.error("Could not find directory %s" % in_directory)
        if not os.path.exists(out_directory):
            logging.error("Could not find directory %s" % out_directory)

        # define network
        self.x = tensor.matrix("x")
        self.y = tensor.matrix("y")

        self.layers = []
        if insert_layers is None or insert_layers == []:
            # load and scale parameters
            w = numpy.load(os.path.join(self.directory, "w.npy"))
            b = numpy.load(os.path.join(self.directory, "b.npy"))
            w = ky / kx * w
            b = ky * b + by - w.sum(axis=0) * bx
            self.layers.append(FPLayer(directory, 0, self.x, w=w, b=b))
        else:
            layer_sizes = [n_x] + insert_layers + [n_y]
            for i in xrange(len(layer_sizes) - 1):
                layer_x = self.x if i == 0 else self.layers[-1].y
                self.layers.append(FPLayer(directory, i, layer_x, n_x=layer_sizes[i], n_y=layer_sizes[i + 1]))
        self.lossLayer = LossLayer(self.layers[-1].y)

        self.l2 = 0
        self.params = []
        for fp_layer in self.layers:
            self.l2 += tensor.sum(fp_layer.w ** 2)
            self.params += fp_layer.params
        self.loss = self.lossLayer.loss(self.y)
        self.error = self.lossLayer.error(self.y)

    def save(self):
        for fp_layer in self.layers:
            fp_layer.save()

    def save_output(self, x_value, prefix):
        """Save output data of entire network

        Args:
            x_value: input of entire network
            prefix: name of the data set
        """
        filename = os.path.join(self.out_directory, "fp.%s.npy" % prefix)
        y_value = self.layers[-1].y.eval({self.x: x_value})
        logging.info("Saving %s" % filename)
        numpy.save(filename, y_value)


class SparseNet(object):
    """Sparse network for transforming one layer in reference network
    """

    def __init__(self, directory, in_directory, out_directory, n_layers, crossbar_size):
        # type: (str, str, str, int, int) -> SparseNet
        """

        Args:
            directory: Directory of the layer to be transformed.
            in_directory: Directory of input data
            out_directory: Directory of output data
            n_layers: number of layers in the network
            crossbar_size: size of crossbar
        """

        # set and checking directories
        self.directory = directory
        self.in_directory = in_directory
        self.out_directory = out_directory
        if not os.path.exists(directory):
            logging.error("Could not find directory %s" % directory)
        if not os.path.exists(in_directory):
            logging.error("Could not find directory %s" % in_directory)
        if not os.path.exists(out_directory):
            logging.error("Could not find directory %s" % out_directory)

        # define network
        self.x = tensor.matrix("x")
        self.y = tensor.matrix("y")

        self.layers = []
        for i in xrange(n_layers):

            # load parameters
            w = numpy.load(os.path.join(self.directory, "fp.%d.w.npy" % i))
            leak = numpy.load(os.path.join(self.directory, "fp.%d.leak.npy" % i))
            threshold = numpy.load(os.path.join(self.directory, "fp.%d.threshold.npy" % i))

            # load or generate mask and connection information
            group_indices_filename = os.path.join(self.directory, "sp.%d.group_indices.pkl" % i)
            w_mask_filename = os.path.join(self.directory, "sp.%d.w_mask.npy" % i)

            try:
                w_mask = numpy.load(w_mask_filename)
            except IOError:
                group_indices, w_mask = transfer.sparse_weight(w, crossbar_size)
                with open(group_indices_filename, "wb") as f:
                    pickle.dump(group_indices, f)
                numpy.save(w_mask_filename, w_mask)

            # build layers
            layer_x = self.x if i == 0 else self.layers[-1].y
            self.layers.append(SparseLayer(directory, i, layer_x, w, leak, threshold, w_mask))
        self.lossLayer = LossLayer(self.layers[-1].y)

        self.l2 = 0
        self.params = []
        for sp_layer in self.layers:
            self.l2 += tensor.sum((sp_layer.w * sp_layer.w_mask) ** 2)
            self.params += sp_layer.params
        self.loss = self.lossLayer.loss(self.y)
        self.error = self.lossLayer.error(self.y)

    def save(self):
        for sp_layer in self.layers:
            sp_layer.save()

    def save_output(self, x_value, prefix):
        """Save output data of entire network

        Args:
            x_value: input of entire network
            prefix: name of the data set
        """
        filename = os.path.join(self.out_directory, "sp.%s.npy" % prefix)
        y_value = self.layers[-1].y.eval({self.x: x_value})
        logging.info("Saving %s" % filename)
        numpy.save(filename, y_value)


class IntNet(object):
    """Int network for transforming one layer in reference network
    """

    def __init__(self, directory, in_directory, out_directory, n_layers, w_precision, leak_precision,
                 threshold_precision, io_precision):
        # type: (str, str, str, int, tuple, tuple. tuple, tuple) -> IntNet
        """

        Args:
            directory: Directory of the layer to be transformed.
            in_directory: Directory of input data
            out_directory: Directory of output data
            n_layers: number of layers in the network
            w_precision: precision of weight in (low, high, step) form
            leak_precision: precision of leak in (low, high, step) form
            threshold_precision: precision of threshold in (low, high, step) form
            io_precision: precision of input & output data in (low, high, step) form
        """

        # set and checking directories
        self.directory = directory
        self.in_directory = in_directory
        self.out_directory = out_directory
        if not os.path.exists(directory):
            logging.error("Could not find directory %s" % directory)
        if not os.path.exists(in_directory):
            logging.error("Could not find directory %s" % in_directory)
        if not os.path.exists(out_directory):
            logging.error("Could not find directory %s" % out_directory)

        # define network
        self.x = tensor.matrix("x")
        self.y = tensor.matrix("y")

        self.layers = []
        for i in xrange(n_layers):
            # load parameters
            w = numpy.load(os.path.join(self.directory, "sp.%d.w.npy" % i))
            leak = numpy.load(os.path.join(self.directory, "sp.%d.leak.npy" % i))
            threshold = numpy.load(os.path.join(self.directory, "sp.%d.threshold.npy" % i))
            w_mask = numpy.load(os.path.join(self.directory, "sp.%d.w_mask.npy" % i))

            # load or generate mask and connection information
            w = w * w_mask
            w, leak, threshold = transfer.scale_parameters(w, leak, threshold, w_precision, leak_precision,
                                                           threshold_precision)

            # build layers
            layer_x = self.x if i == 0 else self.layers[-1].y
            self.layers.append(IntLayer(directory, i, layer_x, w, leak, threshold, w_mask, w_precision, leak_precision,
                                        threshold_precision, io_precision))
        self.lossLayer = LossLayer(self.layers[-1].y)

        self.l2 = 0
        self.params = []
        self.int_params = []
        for int_layer in self.layers:
            self.l2 += tensor.sum((int_layer.w * int_layer.w_mask) ** 2)
            self.params += int_layer.params
            self.int_params += int_layer.int_params
        self.loss = self.lossLayer.loss(self.y)
        self.error = self.lossLayer.error(self.y)

    def save(self):
        for int_layer in self.layers:
            int_layer.save()

    def save_output(self, x_value, prefix):
        """Save output data of entire network

        Args:
            x_value: input of entire network
            prefix: name of the data set
        """
        filename = os.path.join(self.out_directory, "int.%s.npy" % prefix)
        y_value = self.layers[-1].y.eval({self.x: x_value})
        logging.info("Saving %s" % filename)
        numpy.save(filename, y_value)
