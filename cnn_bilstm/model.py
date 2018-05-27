# model class structure adapted from Danijar Hafner
# https://danijar.com/structuring-your-tensorflow-models/
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
import functools
import tensorflow as tf
from .graphs import inference, train


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


class CNNBiLSTM:
    def __init__(self, X, y, lng,
                 batch_size=11,
                 input_vec_size=513,
                 conv1_filters=32,
                 conv2_filters=64,
                 pool1_size=(1, 8),
                 pool1_strides=(1, 8),
                 pool2_size=(1, 8),
                 pool2_strides=(1, 8)
                 ):
        self.X = X
        self.y = y
        self.lng = lng
        self.batch_size = batch_size,
        self.input_vec_size = input_vec_size,
        self.conv1_filters = conv1_filters,
        self.conv2_filters = conv2_filters,
        self.pool1_size = pool1_size,
        self.pool1_strides = pool1_strides,
        self.pool2_size = pool2_size,
        self.pool2_strides = pool2_strides,
        self.graph = tf.Graph()

        self.inference
        self.optimize
        self.error

    @define_scope
    def inference(self):
        return inference(spectrogram=self.X,
                         seq_length=self.lng,
                         n_syllables=self.y.shape[-1],
                         batch_size=self.batch_size,
                         input_vec_size=self.input_vec_size,
                         conv1_filters=self.conv1_filters,
                         conv2_filters=self.conv2_filters,
                         pool1_size=self.pool1_size,
                         pool1_strides=self.pool1_strides,
                         pool2_size=self.pool2_size,
                         pool2_strides=self.pool2_strides)

    def optimize(self):
        train_op, cost = train(logits,
                               lbls,
                               rate,
                               self.batch_size)
        return cost
