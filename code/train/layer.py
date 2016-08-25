import logging
import os

import numpy
import theano
import theano.tensor as tensor


class FCLayer(object):
    """Full connection layer
    """

    def __init__(self, directory, x, n_x, n_y, activation=tensor.nnet.relu):
        # type: (str, tensor.var.TensorVariable, int, int, function) -> FCLayer
        """Init the full connection layer

        Args:
            directory: directory for save data of this layer
            x: input variable of this layer
            n_x: size of input dimension
            n_y: size of output dimension
            activation: activation function of this layer
        """

        # define file names
        self.directory = directory
        self.n_x = n_x
        self.n_y = n_y
        self.w_filename = os.path.join(directory, "w.npy")
        self.b_filename = os.path.join(directory, "b.npy")
        if not os.path.exists(directory):
            logging.info("Creating directory %s" % directory)
            os.mkdir(directory)

        # load and define parameters
        try:
            self.w = theano.shared(
                value=numpy.asarray(numpy.load(self.w_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.b = theano.shared(
                value=numpy.asarray(numpy.load(self.b_filename), dtype=theano.config.floatX),
                borrow=True
            )
        except IOError:
            self.w = theano.shared(
                value=numpy.asarray(numpy.random.uniform(low=-1., high=1., size=(n_x, n_y)),
                                    dtype=theano.config.floatX),
                borrow=True
            )
            self.b = theano.shared(
                value=numpy.zeros((n_y,), dtype=theano.config.floatX),
                borrow=True
            )
        self.params = [self.w, self.b]

        # computation y = activation(wx + b)
        self.x = x
        self.flatten_x = x.reshape([x.shape[0], -1])  # x.shape[0] is batch size
        self.y = activation(tensor.dot(self.flatten_x, self.w) + self.b)

    def save(self):
        """Save parameters 'w' and 'b'"""
        logging.info("Saving %s" % self.w_filename)
        numpy.save(self.w_filename, self.w.get_value(borrow=True))
        logging.info("Saving %s" % self.b_filename)
        numpy.save(self.b_filename, self.b.get_value(borrow=True))

    def eval(self, x_value):
        # type: (numpy.ndarray) -> numpy.ndarray
        """Evaluate y given x

        Args:
            x_value: the value of x

        Returns:
            the value of y given x
        """
        return self.y.eval({self.x: x_value})


class SoftMaxLayer(object):
    """Softmax loss layer
    """

    def __init__(self, x):
        # type: (tensor.var.TensorVariable) -> SoftMaxLayer
        """Init softmax loss layer

        Args:
            x: the input of the softmax layer
        """
        self.x = x
        self.p_y_given_x = tensor.nnet.softmax(self.x)
        self.y_predict = tensor.argmax(self.p_y_given_x, axis=1)

    def loss(self, y):
        # type: (tensor.var.TensorVariable) -> tensor.var.TensorVariable
        """softmax loss

        Args:
            y : the correct label

        Returns:
            loss
        """
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y])

    def error(self, y):
        # type: (tensor.var.TensorVariable) -> tensor.var.TensorVariable
        """error rate

        Args:
            y: the correct label

        Returns:
            error rate
        """
        return tensor.mean(tensor.neq(self.y_predict, y))
