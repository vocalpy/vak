# model class structure adapted from Danijar Hafner
# https://danijar.com/structuring-your-tensorflow-models/
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
from math import ceil
import functools
from collections import namedtuple

import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits

def out_width(in_width, filter_width, stride):
    return ceil(float(in_width - filter_width + 1) / float(stride))


class CNNBiLSTM:
    """hybrid convolutional neural network-bidirectional LSTM
    for segmentation of spectrograms

    Methods
    -------
    __init__ : to make a new model or load a previously trained model
    inference : forward pass through graph, returns predicted probabilities
        for each class for each timebin in a spectrogram
    optimize : given spectrogram and ground truth labels for timebins, computes
        loss and runs optimizer on that loss
    predict : runs tensorflow.nn.top_k on inference to return most likely
        predicted class for each time bin
    error : given spectrogram and ground truth labels, computes error
    saver : instance of tensorflow.saver with the `save` method
    add_summary_writer : adds tensorflow summary writer to model
    """

    def _load(self, sess, meta_file, data_file):
        """load method

        Parameters
        ----------
        sess : tf.Session instance
            session in which this is running
        meta_file : str
            absolute path to meta file saved by CNNBiLSTM.save
        data_file : str
            absolute path to data file saved by CNNBiLSTM.save
        """
        with sess.as_default(graph=self.graph):
            new_saver = tf.train.import_meta_graph(meta_file)
            new_saver.restore(sess, data_file[:-20])
            self.X = self.graph.get_operation_by_name('X')
            self.y = self.graph.get_operation_by_name('y')
            self.lng = self.graph.get_operation_by_name('nSteps')

    def __init__(self,
                 n_syllables=None,
                 batch_size=11,
                 input_vec_size=513,
                 conv1_filters=32,
                 conv2_filters=64,
                 pool1_size=(1, 8),
                 pool1_strides=(1, 8),
                 pool2_size=(1, 8),
                 pool2_strides=(1, 8),
                 learning_rate=0.001,
                 sess=None,
                 meta_file=None,
                 data_file=None,
                 ):
        """__init__ method for CNNBiLSTM
        To instantiate a new CNN-BiLSTM model, call with all of the
        model hyperparameters listed below, i.e. without the parameters
        for loading, `sess`, `meta_file`, and `data_file`.
        To load a previously trained CNN-BiLSTM model, call with
        only the `sess`, `meta_file`, and `data_file` parameters.

        Parameters
        ----------
        n_syllables : int

        batch_size : int

        input_vec_size : int

        conv1_filters : int

        conv2_filters : int

        pool1_size : two element tuple of ints
            Default is (1, 8)

        pool1_strides : two element tuple of ints
            Default is (1, 8)
        pool2_size : two element tuple of ints
            =(1, 8),
        pool2_strides : two element tuple of ints
            =(1, 8),
        learning_rate : float
            Default is 0.001
        sess : tensorflow Session object
            Default is None
        meta_file : str
            Default is None
        data_file : str
            Default is None
        """

        self.graph = tf.Graph()

        load_params = {'sess': sess,
                       'meta_file': meta_file,
                       'data_file': data_file}
        if any(load_params.values()) and not all(load_params.values()):
            # if user passed parameters to load model but not all were
            # specified, throw an error
            specified = [k for k, v in load_params.items()
                             if v is not None]
            not_specified = [k for k, v in load_params.items()
                             if v is None]
            raise ValueError("The arguments {} were specified but the "
                             "arguments {} were not. All are required to "
                             "load a previously saved model."
                             .format(specified, not_specified))

        elif all(load_params.values()):
            # but if all were specified, load the model
            self._load(sess, meta_file, data_file)

        elif not any(load_params.values()):
            # and if none were specified, assume instantiating a new model
            if type(n_syllables) != int:
                raise TypeError('n_syllables must be an integer')
            else:
                if n_syllables < 1:
                    raise ValueError('n_syllables must be a positive integer')

            self.n_syllables = n_syllables
            self.batch_size = batch_size
            self.input_vec_size = input_vec_size
            self.conv1_filters = conv1_filters
            self.conv2_filters = conv2_filters
            self.pool1_size = pool1_size
            self.pool1_strides = pool1_strides
            self.pool2_size = pool2_size
            self.pool2_strides = pool2_strides
            self.learning_rate = learning_rate

            with self.graph.as_default():
                # shape of X is batch_size, time_steps, frequency_bins
                self.X = tf.placeholder(dtype=tf.float32,
                                        shape=[None, None, input_vec_size],
                                        name='X')
                self.y = tf.placeholder(dtype=tf.int32,
                                        shape=[None, None],
                                        name='y')
                # holds the sequence length
                self.lng = tf.placeholder(dtype=tf.int32,
                                          name="nSteps")

                self.inference
                self.optimize
                self.error
                self.saver

                # Merge all summaries into a single op
                self.merged_summary_op = tf.summary.merge_all()

                self.init = tf.global_variables_initializer()

    @define_scope
    def inference(self):
        """inference method, that returns probability of each class
        for each time bin in spectrogram"""
        conv1 = tf.layers.conv2d(
            inputs=tf.reshape(self.X,[self.batch_size, -1,
                                      self.input_vec_size, 1]),
            filters=self.conv1_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name='conv1')

        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=self.pool1_size,
                                        strides=self.pool1_strides,
                                        name='pool1')

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=self.conv2_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name='conv2')

        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=self.pool2_size,
                                        strides=self.pool2_strides,
                                        name='pool2')

        # Determine number of hidden units in bidirectional LSTM:
        # uniquely determined by number of filters and frequency bins
        # in output shape of pool2
        freq_bins_after_pool1 = out_width(self.input_vec_size,
                                          self.pool1_size[1],
                                          self.pool1_strides[1])
        freq_bins_after_pool2 = out_width(freq_bins_after_pool1,
                                          self.pool2_size[1],
                                          self.pool2_strides[1])
        num_hidden = freq_bins_after_pool2 * self.conv2_filters

        # dynamic bi-directional LSTM
        lstm_f1 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0,
                                               state_is_tuple=True, reuse=None)
        lstm_b1 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0,
                                               state_is_tuple=True, reuse=None)
        outputs, _states = tf.nn.bidirectional_dynamic_rnn(lstm_f1,
                                                           lstm_b1,
                                                           tf.reshape(pool2, [
                                                               self.batch_size, -1,
                                                               num_hidden]),
                                                           time_major=False,
                                                           dtype=tf.float32,
                                                           sequence_length=self.lng)

        # projection on the number of syllables creates logits time_steps
        with tf.name_scope('Projection'):
            W_f = tf.Variable(tf.random_normal([num_hidden, self.n_syllables]))
            W_b = tf.Variable(tf.random_normal([num_hidden, self.n_syllables]))
            bias = tf.Variable(tf.random_normal([self.n_syllables]))

        expr1 = tf.unstack(outputs[0],
                           axis=0,
                           num=self.batch_size)
        expr2 = tf.unstack(outputs[1],
                           axis=0,
                           num=self.batch_size)
        logits = tf.concat([tf.matmul(ex1, W_f) + bias + tf.matmul(ex2, W_b)
                            for ex1, ex2 in zip(expr1, expr2)], 0)
        return logits

    @define_scope
    def optimize(self):
        """optimize method, that runs optimizer on one cost value, given
        a spectrogram and ground truth labels for each time bin"""
        xentropy_layer = xentropy(logits=self.inference,
                                  labels=tf.concat(
                                      tf.unstack(self.y,
                                                 axis=0,
                                                 num=self.batch_size),0),
                                  name='xentropy')
        self.cost = tf.reduce_mean(xentropy_layer, name='cost')
        tf.summary.scalar("cost", self.cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(self.cost, global_step=global_step)
        return train_op

    @define_scope
    def predict(self):
        """predict method, that returns argmax(inference) for each time bin,
        i.e. the most likely class"""
        values, indices = tf.nn.top_k(self.inference)
        return indices

    @define_scope
    def error(self):
        """error method, that returns mean error given as input a spectrogram
        and the true and predicted labels"""
        mistakes = tf.not_equal(tf.argmax(self.y, 1),
                                tf.argmax(self.predict, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def saver(self):
        """save method, that uses tf.train.Saver.
        call with session and checkpoint path as arguments"""
        return tf.train.Saver(max_to_keep=10)

    def add_summary_writer(self, logs_path):
        """add summary writer, method that adds as an operation on the graph
        an instance of tf.summary.Filewriter

        Parameters
        ----------
        logs_path : str
            path to which log file is saved"""
        with self.graph.as_default():
            self.summary_writer = tf.summary.FileWriter(logs_path)
