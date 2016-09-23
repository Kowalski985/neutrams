import logging
import os
import pickle

import numpy
import theano.tensor as tensor
import matplotlib.pyplot as plt

from train.net import MLPNet
from train.solver import SGDSolver

logging.basicConfig(format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
numpy.random.seed(1234)

net = MLPNet(
    directory=os.path.join("..", "model", "mnist_mlp"),
    layer_sizes=[784, 50, 10, 10],
    activation=tensor.nnet.relu
)

solver = SGDSolver(
    net=net,
    batch_size=100,
    max_epoch=100,
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=0.001,
    display=1,
    snapshot=10
)

# mnist.pkl is available for download here(http://deeplearning.net/data/mnist/mnist.pkl.gz)
with open(os.path.join("..", "data", "mnist.pkl"), "rb") as f:
    train_set, valid_set, test_set = pickle.load(f)

solver.solve(train_set=train_set, valid_set=valid_set, test_set=test_set)
net.save_blobs(train_set[0], "train")
net.save_blobs(valid_set[0], "valid")
net.save_blobs(test_set[0], "test")
