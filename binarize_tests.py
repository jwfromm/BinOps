#!/usr/bin/env python

# Helper function to extract the value of a TensorFlow tensor
def get_tf(x):
    import tensorflow as tf
    return tf.Session().run(x)

# Helper function to figure out how many distinct values are within an array
def distinct_values(x):
    from numpy import asarray
    d = {}
    for z in asarray(x).flat:
        if not z in d:
            d[z] = 1
    return sum(d.values())

# binarize op test
def test_binarize_op():
    from binarize_ops import binarize
    from numpy import array, int32, int64, float32, float64, all, empty

    # Start with the basics
    assert get_tf(binarize([-1])) == -1
    assert get_tf(binarize([1])) == 1
    assert get_tf(binarize([0])) == -1

    # We support all four dtypes, and we return the input dtype
    for dt in [int32, int64, float32, float64]:
        assert get_tf(binarize(array([1], dtype=dt))).dtype == dt

    # Check that we work on lists and vectors
    x_vec = [-.1, .1, -.2, .2, 0, -5, 10]
    x_vec_bin = [-1, 1, -1, 1, -1, -1, 1]
    assert all(get_tf(binarize(x_vec)) == x_vec_bin)
    assert all(get_tf(binarize(array(x_vec))) == x_vec_bin)
    
    # And matrices
    x_mat = array([
        [ -1,  2,  .1],
        [-.1,  0,  10],
        [ 10, 20,  30],
    ])
    x_mat_bin = array([
        [-1,  1, 1],
        [-1, -1, 1],
        [ 1,  1, 1],
    ])
    assert get_tf(binarize(x_mat)).shape == x_mat_bin.shape
    assert all(get_tf(binarize(x_mat)) == x_mat_bin)

    # And tensors
    x_tens = empty((3, 3, 3))
    x_tens_bin = empty((3, 3, 3))
    for idx in range(3):
        x_tens[idx,:,:] = x_mat
        x_tens_bin[idx, :, :] = x_mat_bin
    assert get_tf(binarize(x_tens)).shape == x_tens_bin.shape
    assert all(get_tf(binarize(x_tens)) == x_tens_bin)


# multibit op test
def test_multibit_op():
    from binarize_ops import multibit
    from numpy import array, int32, int64, float32, float64, all, ones
    from scipy import randn

    # Start with the basics
    assert get_tf(multibit([-1], 1)) == -1
    assert get_tf(multibit([1], 1)) == 1
    assert get_tf(multibit([0], 1)) == 0

    # Test that it can binarize
    assert all(get_tf(multibit([-2, -1, 1, 2], 1)) == [-1, -1, 1, 1])

    # Test shape
    x = randn(3, 17, 250)
    assert get_tf(multibit(x, 4)).shape == x.shape

    # Test dtype
    for dt in [int32, int64, float32, float64]:
        x_dt = array(x, dtype=dt)
        assert get_tf(multibit(x_dt, 4)).dtype == dt
    
    # Test the number of unique values is always <= 2**bit_map
    for bit_map in range(6):
        x_multi = get_tf(multibit(x, bit_map))
        nd = distinct_values(x_multi)
        assert nd <= 2**bit_map

    # Test that we can specify a bit_map with variable bit numbers
    bit_map = ones((3, 17, 250))
    bit_map[1,:,:] = 5
    x_multi = get_tf(multibit(x, bit_map))
    nd1 = distinct_values(x_multi[2:,:,:])
    assert nd1 <= 2**1
    nd5 = distinct_values(x_multi[1,:,:])
    assert nd5 <= 2**5
    nd = distinct_values(x_multi)
    assert nd <= 2**1 + 2**5

    # Test that our bit_map gets broadcast up properly
    bit_map = ones((1, 1, 250))
    bit_map[:, :, :50] = 5
    x_multi = get_tf(multibit(x, bit_map))
    nd1 = distinct_values(x_multi[:,:,50:])
    assert nd1 <= 2**1
    nd5 = distinct_values(x_multi[1,:,:50])
    assert nd5 <= 2**5
    nd = distinct_values(x_multi)
    assert nd <= 2**1 + 2**5

