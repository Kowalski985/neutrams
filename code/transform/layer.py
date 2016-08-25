"""This file defines the layers used in the transformation, the code is based on TIANJI chip.

TIANJI supports both SNN and ANN. Here we use its SNN mode only.

The neuron model is a simplified LIF (leaky integrate and fire) model.
    V(t+1) = V(t) + W * X + L
where V is the membrane potential, W is the weight encoded with 8-bit integer,
X represents the input spikes in time step t. X is a binary vector.
L is the leakage current encoded with 16-bit integer (usually L is negative)
When V reach threshold T , the neuron will issue a spike and V will reset to 0.
T is also encoded with 16-bit integer.

In the firing-rate domain, this model can be written as
    Y = max((W * X + L) / T, 0)
X and Y represent the firing rates of input and output.
"""
import logging
import os

import numpy
import theano
import theano.tensor as tensor


class FPLayer(object):
    """Float-point layer is used in the first step of the transformation

    Float-point layers constrain the activation function, value range of data and weight.
    Encoding of data and weight is still float-point.
    """

    def __init__(self, directory, index, x, n_x=-1, n_y=-1, w=None, b=None):
        # type: (str, int, tensor.var.TensorVariable, int, int, numpy.ndarray, numpy.ndarray) -> FPLayer
        """Init the float-point layer

        Args:
            directory: The directory that contains the layer information.
            index: The layer index of all fp-layers used for the transforming layer.
            x: Input variable.
            n_x: Input dimension. When w and b is provided, it is set to -1
            n_y: Output dimension. When w and b is provided, it is set to -1
            w: Transformed w for initialization.
            b: Transformed b for initialization.
        """

        # define file names
        self.directory = directory
        self.w_filename = os.path.join(directory, "fp.%d.w.npy" % index)
        self.leak_filename = os.path.join(directory, "fp.%d.leak.npy" % index)
        self.threshold_filename = os.path.join(directory, "fp.%d.threshold.npy" % index)
        if not os.path.exists(directory):
            logging.info("Creating directory %s" % directory)
            os.mkdir(directory)

        # load and define parameters
        try:
            self.w = theano.shared(
                value=numpy.asarray(numpy.load(self.w_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.leak = theano.shared(
                value=numpy.asarray(numpy.load(self.leak_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.threshold = theano.shared(
                value=numpy.asarray(numpy.load(self.threshold_filename), dtype=theano.config.floatX),
                borrow=True
            )
        except IOError:
            if w is None or b is None:
                self.w = theano.shared(
                    value=numpy.asarray(
                        numpy.random.uniform(
                            low=0.,
                            high=1. / n_x,
                            size=(n_x, n_y)
                        ),
                        dtype=theano.config.floatX
                    ),
                    borrow=True
                )
                self.leak = theano.shared(
                    value=numpy.zeros((n_y,), dtype=theano.config.floatX),
                    borrow=True
                )
                self.threshold = theano.shared(
                    value=numpy.ones((n_y,), dtype=theano.config.floatX),
                    borrow=True
                )
            else:
                self.w = theano.shared(
                    value=numpy.asarray(w, dtype=theano.config.floatX),
                    borrow=True
                )
                self.leak = theano.shared(
                    value=numpy.asarray(b, dtype=theano.config.floatX),
                    borrow=True
                )
                self.threshold = theano.shared(
                    value=numpy.ones((b.shape[0],), dtype=theano.config.floatX),
                    borrow=True
                )

        self.params = [self.w, self.leak, self.threshold]

        # computation y = max((wx + leak) / threshold, 0)
        self.x = x
        synapse_output = tensor.dot(self.x, self.w)
        neuron_output = (synapse_output + self.leak) / self.threshold
        self.y = tensor.switch(neuron_output > 0, neuron_output, 0)

    def save(self):
        """Save parameters"""
        logging.info("Saving %s" % self.w_filename)
        numpy.save(self.w_filename, self.w.get_value(borrow=True))
        logging.info("Saving %s" % self.leak_filename)
        numpy.save(self.leak_filename, self.leak.get_value(borrow=True))
        logging.info("Saving %s" % self.threshold_filename)
        numpy.save(self.threshold_filename, self.threshold.get_value(borrow=True))


class SparseLayer(object):
    """Sparse layer is used in the second step of the transformation

    Sparse layers constrain connectivity besides the constraints introduced in previous steps.
    The connectivity constraint is implemented with a mask for w.
    """

    def __init__(self, directory, index, x, w, leak, threshold, w_mask):
        """Init the sparse layer

        Args:
            directory: The directory that contains the layer information.
            index: The layer index of all fp-layers used for the transforming layer.
            x: Input variable.
            w: transformed w for initialization
            threshold: transformed threshold for initialization
            leak: transformed leak for initialization
            w_mask: mask for w
        """

        # define file names
        self.directory = directory
        self.w_filename = os.path.join(directory, "sp.%d.w.npy" % index)
        self.leak_filename = os.path.join(directory, "sp.%d.leak.npy" % index)
        self.threshold_filename = os.path.join(directory, "sp.%d.threshold.npy" % index)
        if not os.path.exists(directory):
            logging.info("Creating directory %s" % directory)
            os.mkdir(directory)

        # load and define parameters
        try:
            self.w = theano.shared(
                value=numpy.asarray(numpy.load(self.w_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.leak = theano.shared(
                value=numpy.asarray(numpy.load(self.leak_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.threshold = theano.shared(
                value=numpy.asarray(numpy.load(self.threshold_filename), dtype=theano.config.floatX),
                borrow=True
            )
        except IOError:
            self.w = theano.shared(
                value=numpy.asarray(w, dtype=theano.config.floatX),
                borrow=True
            )
            self.leak = theano.shared(
                value=numpy.asarray(leak, dtype=theano.config.floatX),
                borrow=True
            )
            self.threshold = theano.shared(
                value=numpy.asarray(threshold, dtype=theano.config.floatX),
                borrow=True
            )
        self.w_mask = theano.shared(
            value=numpy.asarray(w_mask, dtype=theano.config.floatX),
            borrow=True
        )
        self.params = [self.w, self.leak, self.threshold]

        # computation y = max((w * w_mask * x + leak) / threshold, 0)
        self.x = x
        synapse_output = tensor.dot(self.x, self.w * self.w_mask)
        neuron_output = (synapse_output + self.leak) / self.threshold
        self.y = tensor.switch(neuron_output > 0, neuron_output, 0)

    def save(self):
        """Save parameters"""
        logging.info("Saving %s" % self.w_filename)
        numpy.save(self.w_filename, self.w.get_value(borrow=True))
        logging.info("Saving %s" % self.leak_filename)
        numpy.save(self.leak_filename, self.leak.get_value(borrow=True))
        logging.info("Saving %s" % self.threshold_filename)
        numpy.save(self.threshold_filename, self.threshold.get_value(borrow=True))


class IntLayer(object):
    """Int layer is used in the third step of the transformation

    Int layers constrain precision besides the constraints introduced in previous steps.
    The parameters are still stored with fp precision.
    During the forward phase, parameters are rounded to target precision and then take into computation.
    During the propagate phase and update phase, fp precision is used.
    Small changes can be accumulated and make the rounding results change.
    """

    def __init__(self, directory, index, x, w, leak, threshold, w_mask, w_precision, leak_precision,
                 threshold_precision, io_precision):
        # type: (str, int, tensor.var.TensorVariable, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, tuple,
        #  tuple, tuple, tuple) -> IntLayer
        """

        Args:
            directory: The directory that contains the layer information.
            index: The layer index of all fp-layers used for the transforming layer.
            x: Input variable.
            w: Transformed w for initialization.
            threshold: Transformed threshold for initialization.
            leak: Transformed leak for initialization.
            w_mask: Mask for w.
            w_precision: Precision of weight in tuple (low, high, step) form.
            leak_precision: Precision of leak in tuple (low, high, step) form.
            threshold_precision: Precision of threshold in tuple (low, high, step) form.
            io_precision: Precision of input & output data in tuple (low, high, step) form.
        """

        # define file names
        self.directory = directory
        self.w_filename = os.path.join(directory, "int.%d.w.npy" % index)
        self.leak_filename = os.path.join(directory, "int.%d.leak.npy" % index)
        self.threshold_filename = os.path.join(directory, "int.%d.threshold.npy" % index)
        if not os.path.exists(directory):
            logging.info("Creating directory %s" % directory)
            os.mkdir(directory)

        # load and define parameters
        try:
            self.w = theano.shared(
                value=numpy.asarray(numpy.load(self.w_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.leak = theano.shared(
                value=numpy.asarray(numpy.load(self.leak_filename), dtype=theano.config.floatX),
                borrow=True
            )
            self.threshold = theano.shared(
                value=numpy.asarray(numpy.load(self.threshold_filename), dtype=theano.config.floatX),
                borrow=True
            )
        except IOError:
            self.w = theano.shared(
                value=numpy.asarray(w, dtype=theano.config.floatX),
                borrow=True
            )
            self.leak = theano.shared(
                value=numpy.asarray(leak, dtype=theano.config.floatX),
                borrow=True
            )
            self.threshold = theano.shared(
                value=numpy.asarray(threshold, dtype=theano.config.floatX),
                borrow=True
            )
        self.w_mask = theano.shared(
            value=numpy.asarray(w_mask, dtype=theano.config.floatX),
            borrow=True
        )
        self.params = [self.w, self.leak, self.threshold]

        # int precision parameters
        def get_int_param(param, precision):
            low, high, step = precision
            int_param = tensor.round(param / step) * step
            return tensor.switch(int_param < low, low, tensor.switch(int_param > high, high, int_param))

        self.int_w = get_int_param(self.w, w_precision)
        self.int_leak = get_int_param(self.leak, leak_precision)
        self.int_threshold = get_int_param(self.threshold, threshold_precision)
        self.int_params = [self.int_w, self.int_leak, self.int_threshold]

        # computation
        self.x = x
        synapse_output = tensor.dot(self.x, self.int_w * self.w_mask)
        neuron_output = (synapse_output + self.int_leak) / self.int_threshold
        io_low, io_high, io_step = io_precision
        int_neuron_output = neuron_output
        #int_neuron_output = tensor.round(neuron_output / io_step) * io_step
        self.y = tensor.switch(int_neuron_output < io_low, io_low, int_neuron_output)
                               #tensor.switch(int_neuron_output > io_high, io_high, int_neuron_output))

    def save(self):
        """Save parameters"""
        logging.info("Saving %s" % self.w_filename)
        numpy.save(self.w_filename, self.w.get_value(borrow=True))
        logging.info("Saving %s" % self.leak_filename)
        numpy.save(self.leak_filename, self.leak.get_value(borrow=True))
        logging.info("Saving %s" % self.threshold_filename)
        numpy.save(self.threshold_filename, self.threshold.get_value(borrow=True))


class LossLayer(object):
    """Mean square loss error"""

    def __init__(self, x):
        # type: (tensor.var.TensorVariable) -> LossLayer
        """
        Args:
            x: input variable
        """
        self.y_predict = x

    def loss(self, y):
        # type: (tensor.var.TensorVariable) -> tensor.var.TensorVariable
        """
        Args:
            y: reference variable
        """
        return tensor.mean((self.y_predict - y) ** 2)

    def error(self, y):
        # type: (tensor.var.TensorVariable) -> tensor.var.TensorVariable
        """
        Args:
            y: reference variable
        """
        return tensor.mean((self.y_predict - y) ** 2)
