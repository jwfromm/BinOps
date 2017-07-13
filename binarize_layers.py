import tensorflow as tf
from tensorflow.python.framework import ops
import keras
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.layers.convolutional import Conv2D
from keras.layers import activations, initializers, regularizers, constraints, InputSpec
from keras import backend as K
from keras.regularizers import Regularizer
from binarize_ops import multibit, binarize

### binary sign function with straight through estimator gradient and associated gradient overrides ###
def binarySign(x):
    g = tf.get_default_graph()
    with ops.name_scope("binarySign") as name:
        with g.gradient_override_map({"Sign": "Bin_ST"}):
            return tf.sign(x)

@ops.RegisterGradient("Bin_ST")
def bin_ST(op, grad):
    in_vals = op.inputs[0]
    grad_func = tf.cast(tf.less_equal(tf.abs(in_vals), 1), tf.float32)
    grad_out = tf.multiply(grad_func, grad)
    return [grad_out]

### Keras layer implemented binarization ###
class BinLayer(Layer):
    def call(self, x):
        return binarize(x)

    def compute_output_shape(self, input_shape):
        return input_shape

### Keras multibit layer
class MultibitLayer(Layer):
    def __init__(self, num_bits, **kwargs):
        self.num_bits = num_bits
        super(MultibitLayer, self).__init__(**kwargs)

    def call(self, x):
        return multibit(x, self.num_bits)

    def compute_output_shape(self, input_shape):
        return input_shape
    
### Keras layer for l2 distance between two tensors
class L2DistanceLayer(Layer):
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1][:, 0:x.shape[1], :]
        return tf.norm(tf.abs(tf.subtract(x, y)), ord=2)

    def compute_output_shape(self, input_shape):
        return 1


class BinDense(keras.layers.Dense):
    """
    Just like `Dense`, but binarizes weights before evaluation.
    """
    def build(self, input_shape):
        super(BinDense, self).build(input_shape)
        self.kernel = binarize(self.kernel)

class MultiDense(keras.layers.Dense):
    """
    Just like `Dense`, but quantizes weights to a certain bitdepth using the
    parameter `bit_map`, which is a numeric quantity (scalar, vector, general
    N-D array, etc...) that can be broadcast up to the size of the weights,
    setting the number of bits for each weight accordingly.
    """

    def __init__(self, units, bit_map=1, **kwargs):
        super(MultiDense, self).__init__(units, **kwargs)
        self.bit_map = bit_map

    def build(self, input_shape):
        super(MultiDense, self).build(input_shape)
        self.kernel = multibit(self.kernel, self.bit_map)

    def get_config(self):
        config = {
            'bit_map': self.bit_map,
        }
        base_config = super(MultiDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class BinConv(Conv2D):
    """
    Just like `_Conv`, but binarizes weights before evaluation.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        super(BinConv, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        super(BinConv, self).build(input_shape)
        self.kernel = binarize(self.kernel)

class MultiConv(Conv2D):
    """
    Just like `_Conv`, but quantizes weights to a certain bitdepth using the
    parameter `bit_map`, which is a numeric quantity (scalar, vector, general
    N-D array, etc...) that can be broadcast up to the size of the weights,
    setting the number of bits for each weight accordingly.
    """

    def __init__(self, filters, kernel_size, bit_map=1, **kwargs):
        super(MultiConv, self).__init__(filters, kernel_size, **kwargs)
        self.bit_map = bit_map

    def build(self, input_shape):
        super(MultiConv, self).build(input_shape)
        self.kernel = multibit(self.kernel, self.bit_map)

    def get_config(self):
        config = {
            'bit_map': self.bit_map,
        }
        base_config = super(MultiConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))     
    
class BinReg(Regularizer):
    """
    Regularizer for a binary layer
    # Arguments       
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l2=0.000003):        
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l2:
            regularization += K.sum(K.abs(self.l2 * (1 - K.square(x))))
        return regularization

    def get_config(self):
        return {'l2': float(self.l2)}