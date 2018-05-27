# model class structure adapted from Danijar Hafner
# https://danijar.com/structuring-your-tensorflow-models/
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
from math import ceil
import functools

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
    for segmentation of spectrograms"""

    def __init__(self,
                 n_syllables,
                 batch_size=11,
                 input_vec_size=513,
                 conv1_filters=32,
                 conv2_filters=64,
                 pool1_size=(1, 8),
                 pool1_strides=(1, 8),
                 pool2_size=(1, 8),
                 pool2_strides=(1, 8),
                 learning_rate=0.001,
                 ):

        if type(n_syllables) != int:
            raise TypeError('n_syllables must be an integer')
        else:
            if n_syllables < 1:
                raise ValueError('n_syllables must be a positive integer')
            else:
                n_syllables = (n_syllables, 1)

        self.graph = tf.Graph()
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
        import pdb;pdb.set_trace()
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

        self.inference
        self.optimize
        self.error

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        self.saver


    @define_scope
    def inference(self):
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
        xentropy_layer = xentropy(logits=self.inference,
                                  labels=tf.concat(
                                      tf.unstack(self.y,
                                                 axis=0,
                                                 num=self.batch_size),0),
                                  name='xentropy')
        cost = tf.reduce_mean(xentropy_layer, name='cost')
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op

    @define_scope
    def error(self):
        eval_op = tf.nn.top_k(self.inference)
        mistakes = tf.not_equal(tf.argmax(self.y, 1),
                                tf.argmax(eval_op, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @define_scope
    def save(self):
        return tf.train.Saver(max_to_keep=10)

    @define_scope
    def load(self, sess, meta_file, data_file):
        loader = tf.train.import_meta_graph(meta_file)
        loader.restore(sess, data_file[:-20])
