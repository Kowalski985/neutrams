import logging
import os

import numpy
import theano.tensor as tensor

from train.net import MLPNet
from transform.transfer import check_recognition_error
from transform.transform import Transformer

logging.basicConfig(format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
numpy.random.seed(1234)

net = MLPNet(
    directory=os.path.join("..", "model", "mnist_mlp"),
    layer_sizes=[784, 50, 10, 10],
    activation=tensor.nnet.relu
)

TIANJI_io_precision = (0., 1., 0.001)
TIANJI_w_precision = (-128., 127., 1.)
TIANJI_leak_precision = (-32768., 32767., 1.)
TIANJI_threshold_precision = (-32768., 32767., 1.)
TIANJI_crossbar_size = 256

transformer = Transformer(net, TIANJI_io_precision, TIANJI_w_precision, TIANJI_leak_precision,
                          TIANJI_threshold_precision, TIANJI_crossbar_size)

# layer 1: 784->50
transformer.transform_fp(
    index=1,
    insert_layers=None,
    batch_size=100,
    max_epoch=100,
    learning_rate=1e-5,
    momentum=0.,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_sp(
    index=1,
    num_layers=1,
    batch_size=100,
    max_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_int(
    index=1,
    num_layers=1,
    batch_size=100,
    max_epoch=100,
    learning_rate=1e3,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

# layer 2: 50->10
transformer.transform_fp(
    index=2,
    insert_layers=None,
    batch_size=100,
    max_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_sp(
    index=2,
    num_layers=1,
    batch_size=100,
    max_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_int(
    index=2,
    num_layers=1,
    batch_size=100,
    max_epoch=10,
    learning_rate=1e3,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

# layer 3: 10->10
transformer.transform_fp(
    index=3,
    insert_layers=None,
    batch_size=100,
    max_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_sp(
    index=3,
    num_layers=1,
    batch_size=100,
    max_epoch=10,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

transformer.transform_int(
    index=3,
    num_layers=1,
    batch_size=100,
    max_epoch=10,
    learning_rate=1,
    momentum=0.9,
    weight_decay=0.00001,
    display=1,
    snapshot=10
)

data_path = os.path.join(net.directory, "data3")
print("Training set error: %f" % check_recognition_error(data_path, "train"))
print("Validation set error: %f" % check_recognition_error(data_path, "valid"))
print("Test set error: %f" % check_recognition_error(data_path, "test"))