import logging
import os

import numpy
import theano.tensor as tensor

import layer


class MLPNet(object):
    """Multi Layer Perceptron
    """

    def __init__(self, directory, layer_sizes, activation):
        # type: (str, list, function) -> MLPNet
        """

        Args:
            directory: the directory for saving the network
            layer_sizes: the size of input layer, hidden layers and output layer
            activation: activation function used in each layer
        """
        self.directory = directory
        if not os.path.exists(directory):
            logging.info("Creating directory %s" % directory)
            os.mkdir(directory)

        self.x = tensor.matrix("x")
        self.y = tensor.ivector("y")
        self.layers = []
        for i in xrange(len(layer_sizes) - 1):
            layer_directory = os.path.join(self.directory, "layer%d" % (i + 1))
            layer_x = self.x if i == 0 else self.layers[i - 1].y
            self.layers.append(
                layer.FCLayer(
                    directory=layer_directory,
                    x=layer_x,
                    n_x=layer_sizes[i],
                    n_y=layer_sizes[i + 1],
                    activation=activation
                )
            )
        self.lossLayer = layer.SoftMaxLayer(x=self.layers[-1].y)

        self.l2 = 0
        self.params = []
        for fc_layer in self.layers:
            self.l2 += tensor.sum(fc_layer.w ** 2)
            self.params += fc_layer.params
        self.loss = self.lossLayer.loss(self.y)
        self.error = self.lossLayer.error(self.y)

    def save(self):
        """Save the network
        """
        for fc_layer in self.layers:
            fc_layer.save()

    def save_blobs(self, x_value, prefix):
        """Save output data of each layer given network input

        Args:
            x_value: input of entire network
            prefix: name of the data set
        """
        # save input data
        data_directory = os.path.join(self.directory, "data0")
        if not os.path.exists(data_directory):
            logging.info("Creating directory %s" % data_directory)
            os.mkdir(data_directory)
        data_filename = os.path.join(data_directory, "%s.npy" % prefix)
        logging.info("Saving %s" % data_filename)
        numpy.save(data_filename, x_value)

        # save output data of each layer
        for i, fc_layer in enumerate(self.layers):
            data_directory = os.path.join(self.directory, "data%d" % (i + 1))
            if not os.path.exists(data_directory):
                logging.info("Creating directory %s" % data_directory)
                os.mkdir(data_directory)
            data_filename = os.path.join(data_directory, "%s.npy" % prefix)
            y_value = fc_layer.eval(x_value)
            x_value = y_value
            logging.info("Saving %s" % data_filename)
            numpy.save(data_filename, y_value)
