import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

binarize_module = tf.load_op_library("libbinarize.so")
multibit_module = tf.load_op_library("libmultibit.so")

@ops.RegisterGradient("Binarize")
def bin_grad(op, grad):
    in_vals = op.inputs[0]
    grad_func = tf.cast(tf.less_equal(tf.abs(in_vals), 1), tf.float32)
    grad_out = tf.multiply(grad_func, grad)
    return [grad_out]

@ops.RegisterGradient("Multibit")
def multibit_grad(op, grad):
    in_vals = op.inputs[0]
    grad_func = tf.cast(tf.less_equal(tf.abs(in_vals), 1), tf.float32)
    grad_out = tf.multiply(grad_func, grad)
    return [grad_out, tf.zeros(tf.shape(op.inputs[1])), tf.zeros(tf.shape(op.inputs[2]))]

def is_wrapped_weights(x):
    if isinstance(x, list) or isinstance(x, tuple):
        if isinstance(x[0], list) or isinstance(x[0], np.ndarray):
            return True
    return False

def binarize(x):
    return binarize_module.binarize(x)

def multibit(x, bit_map):
    # If we have a tuple or list that is a bundle of weights, grab 'em out:
    if is_wrapped_weights(x):
        x = x[0]
    
    # Always treat x as a numpy array
    x = np.asarray(x)

    # Broadcast bit_map up to the same says as the weights shape
    bit_map = np.broadcast_to(bit_map, x.shape)
    max_bit = np.max(bit_map)
    return multibit_module.multibit(x, tf.constant(bit_map, dtype=np.int32), tf.constant(max_bit, dtype=np.int32))