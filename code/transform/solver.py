import logging
import os

import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as tensor


class SGDSolver(object):
    """Stochastic Gradient Descent Solver
    """

    def __init__(self, net, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                 snapshot):
        # type: (MLPNet, int, int, float, float, float, int, int) -> SGDSolver
        """

        Args:
            net: The neural network to be trained.
            batch_size: Batch size.
            max_epoch: number of epoch to run
            learning_rate: Base learning rate.
            momentum: Momentum.
            weight_decay: L2 norm regularization.
            display: Number of iterations to update displayed graph. If it is 0, display will be disabled.
            snapshot: Number of iterations to save a snapshot.
        """
        self.net = net
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.display = display
        self.snapshot = snapshot

    def solve(self, train_set, valid_set, test_set):
        # type: (tuple, tuple, tuple) -> None
        """Solve the optimization.

        Args:
            train_set: Training set in (x, y) form.
            valid_set: Validation set in (x, y) form.
            test_set: Test set in (x, y) form.
        """

        # prepare data set
        def make_shared(data_set):
            data_x, data_y = data_set
            shared_x = theano.shared(
                value=numpy.asarray(data_x, dtype=theano.config.floatX),
                borrow=True
            )
            shared_y = theano.shared(
                value=numpy.asarray(data_y, dtype=theano.config.floatX),
                borrow=True
            )
            return shared_x, shared_y

        logging.info("Preparing data set")
        train_x, train_y = make_shared(train_set)
        valid_x, valid_y = make_shared(valid_set)
        test_x, test_y = make_shared(test_set)

        # define gradient and updates
        logging.info("Building models")
        cost = self.net.loss + self.weight_decay * self.net.l2
        grads = [tensor.grad(cost, param) for param in self.net.params]
        deltas = [theano.shared(param.get_value() * 0., broadcastable=param.broadcastable) for param in self.net.params]
        updates = []
        updates += [(delta, self.momentum * delta + (1 - self.momentum) * grad) for (delta, grad) in zip(deltas, grads)]
        updates += [(param, param - self.learning_rate * delta) for (param, delta) in zip(self.net.params, deltas)]

        # build models
        index = tensor.lscalar()

        def build_model(data_x, data_y, update_dict=None):
            return theano.function(
                inputs=[index],
                outputs=self.net.error,
                updates=update_dict,
                givens={
                    self.net.x: data_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.net.y: data_y[index * self.batch_size: (index + 1) * self.batch_size]
                }
            )

        train_model = build_model(train_x, train_y, updates)
        valid_model = build_model(valid_x, valid_y)
        test_model = build_model(test_x, test_y)

        n_train_batch = train_x.get_value(borrow=True).shape[0] // self.batch_size
        n_valid_batch = valid_x.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batch = valid_x.get_value(borrow=True).shape[0] // self.batch_size

        train_errors = []
        valid_test_index = []
        valid_errors = []
        test_errors = []

        train_figure = -1
        if self.display > 0:
            train_figure = plt.figure().number
            plt.figure(train_figure)
            plt.ion()

        logging.info("Start training")

        for epoch in xrange(self.max_epoch):
            # train model
            train_error = sum([train_model(i) for i in xrange(n_train_batch)]) / n_train_batch
            train_errors.append(train_error)

            logging.info("Epoch %d, training error = %0.5f" % (epoch + 1, train_error))

            valid_test_index.append(epoch)

            valid_error = sum([valid_model(i) for i in xrange(n_valid_batch)]) / n_valid_batch
            valid_errors.append(valid_error)

            logging.info("Epoch %d, validation error = %0.5f" % (epoch + 1, valid_error))

            test_error = sum([test_model(i) for i in xrange(n_test_batch)]) / n_test_batch
            test_errors.append(test_error)

            logging.info("Epoch %d, test error = %0.5f" % (epoch + 1, test_error))

            # display
            if self.display > 0 and (epoch + 1) % self.display == 0:
                plt.figure(train_figure)
                plt.clf()
                plt.plot(train_errors, label="train")
                plt.plot(valid_test_index, valid_errors, label="valid")
                plt.plot(valid_test_index, test_errors, label="test")
                plt.legend()
                plt.pause(0.001)

            # snapshot
            if self.snapshot > 0 and (epoch + 1) % self.snapshot == 0:
                logging.info("Saving net to %s" % self.net.directory)
                self.net.save()

class IntSGDSolver(object):
    """Stochastic Gradient Descent Solver for low precision
    """

    def __init__(self, net, batch_size, max_epoch, learning_rate, momentum, weight_decay, display,
                 snapshot):
        # type: (MLPNet, int, int, float, float, float, int, int) -> SGDSolver
        """

        Args:
            net: The neural network to be trained.
            batch_size: Batch size.
            max_epoch: number of epoch to run
            learning_rate: Base learning rate.
            momentum: Momentum.
            weight_decay: L2 norm regularization.
            display: Number of iterations to update displayed graph. If it is 0, display will be disabled.
            snapshot: Number of iterations to save a snapshot.
        """
        self.net = net
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.display = display
        self.snapshot = snapshot

    def solve(self, train_set, valid_set, test_set):
        # type: (tuple, tuple, tuple) -> None
        """Solve the optimization.

        Args:
            train_set: Training set in (x, y) form.
            valid_set: Validation set in (x, y) form.
            test_set: Test set in (x, y) form.
        """

        # prepare data set
        def make_shared(data_set):
            data_x, data_y = data_set
            shared_x = theano.shared(
                value=numpy.asarray(data_x, dtype=theano.config.floatX),
                borrow=True
            )
            shared_y = theano.shared(
                value=numpy.asarray(data_y, dtype=theano.config.floatX),
                borrow=True
            )
            return shared_x, shared_y

        logging.info("Preparing data set")
        train_x, train_y = make_shared(train_set)
        valid_x, valid_y = make_shared(valid_set)
        test_x, test_y = make_shared(test_set)

        # define gradient and updates
        logging.info("Building models")
        cost = self.net.loss + self.weight_decay * self.net.l2
        grads = [tensor.grad(cost, int_param) for int_param in self.net.int_params]
        deltas = [theano.shared(param.get_value() * 0., broadcastable=param.broadcastable) for param in self.net.params]
        updates = []
        updates += [(delta, self.momentum * delta + (1 - self.momentum) * grad) for (delta, grad) in zip(deltas, grads)]
        updates += [(param, param - self.learning_rate * delta) for (param, delta) in zip(self.net.params, deltas)]

        # build models
        index = tensor.lscalar()

        def build_model(data_x, data_y, update_dict=None):
            return theano.function(
                inputs=[index],
                outputs=self.net.error,
                updates=update_dict,
                givens={
                    self.net.x: data_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.net.y: data_y[index * self.batch_size: (index + 1) * self.batch_size]
                }
            )

        train_model = build_model(train_x, train_y, updates)
        valid_model = build_model(valid_x, valid_y)
        test_model = build_model(test_x, test_y)

        n_train_batch = train_x.get_value(borrow=True).shape[0] // self.batch_size
        n_valid_batch = valid_x.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batch = valid_x.get_value(borrow=True).shape[0] // self.batch_size

        train_errors = []
        valid_test_index = []
        valid_errors = []
        test_errors = []

        train_figure = -1
        if self.display > 0:
            train_figure = plt.figure().number
            plt.figure(train_figure)
            plt.ion()

        logging.info("Start training")

        for epoch in xrange(self.max_epoch):
            # train model
            train_error = sum([train_model(i) for i in xrange(n_train_batch)]) / n_train_batch
            train_errors.append(train_error)

            logging.info("Epoch %d, training error = %0.5f" % (epoch + 1, train_error))

            valid_test_index.append(epoch)

            valid_error = sum([valid_model(i) for i in xrange(n_valid_batch)]) / n_valid_batch
            valid_errors.append(valid_error)

            logging.info("Epoch %d, validation error = %0.5f" % (epoch + 1, valid_error))

            test_error = sum([test_model(i) for i in xrange(n_test_batch)]) / n_test_batch
            test_errors.append(test_error)

            logging.info("Epoch %d, test error = %0.5f" % (epoch + 1, test_error))

            # display
            if self.display > 0 and (epoch + 1) % self.display == 0:
                plt.figure(train_figure)
                plt.clf()
                plt.plot(train_errors, label="train")
                plt.plot(valid_test_index, valid_errors, label="valid")
                plt.plot(valid_test_index, test_errors, label="test")
                plt.legend()
                plt.pause(0.001)

            # snapshot
            if self.snapshot > 0 and (epoch + 1) % self.snapshot == 0:
                logging.info("Saving net to %s" % self.net.directory)
                self.net.save()
