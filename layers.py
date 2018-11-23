from abc import ABCMeta, abstractmethod
import tensorflow as tf


class MyLayers(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, inputs):
        pass

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def pre_activation(self):
        return self._pre_activation

    @property
    def activation_fn(self):
        return self._activation_fn

    @property
    def out_activation(self):
        return self._out_activation


def weight_fc(shape, stddev=0.1, initial=None, dtype=None):
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=dtype)
        initial = tf.Variable(initial)
        # initial = tf.Variable(tf.random_normal(shape), dtype=self.dtype_X)
    return initial

def bias_fc(shape, init_bias=0.1, initial=None, dtype=None):
    if initial is None:
        initial = tf.constant(init_bias, shape=shape, dtype=dtype)
        initial = tf.Variable(initial)
        # initial = tf.Variable(tf.zeros(shape) + 0.1, dtype=self.dtype_X)
    return initial


class FC(MyLayers):

    def __init__(self, inputs, out_dims, activation_fn, dtype=tf.float32):

        shape_inputs = inputs.get_shape().as_list()
        assert len(shape_inputs) == 2 # [None, in_dims]
        in_dims = shape_inputs[1]

        shape_W = [in_dims,out_dims]
        shape_b = [1,out_dims]

        self._in_dims = in_dims
        self._out_dims = out_dims # TODO: [None, n_out]?
        self._weights = weight_fc(shape_W, dtype=dtype)
        self._biases = bias_fc(shape_b, dtype=dtype)
        self._pre_activation = tf.add(tf.matmul(inputs, self.weights), self._biases)
        self._activation_fn = activation_fn

        if activation_fn is None:
            self._out_activation = self._pre_activation
        else:
            self._out_activation = activation_fn(self._pre_activation)

class Dropout(MyLayers):

    def __init__(self, inputs, keep_prob):

        shape_inputs = inputs.get_shape().as_list()
        assert len(shape_inputs) == 2 # [None, in_dims]
        in_dims = shape_inputs[1]
        tf_keep_prob = tf.placeholder_with_default(1.0, shape=())
        outputs = tf.nn.dropout(inputs, keep_prob=tf_keep_prob) # TODO: keep_prob should be placeholder? tf.placeholder(tf.float32)

        self._in_dims = in_dims
        self._out_dims = in_dims # TODO: [None, n_out]?
        self._weights = None
        self._biases = None
        self._pre_activation = outputs
        self._activation_fn = None
        self._out_activation = outputs


class Conv2d(MyLayers):

    def __init__(self, inputs, n_filters, activation_fn, dtype=tf.float32):
        pass
        