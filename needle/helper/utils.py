import numpy as np
import tensorflow as tf


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def merge_dict(*args, **kwds):
    for d in args:
        for k, v in d.items():
            kwds[k] = v
    return kwds


def decay_cumsum(a, decay, axis=-1):
    assert axis == -1 and len(a.shape) == 2

    b = a.copy()
    for i in range(1, a.shape[axis]):
        b[:, i] += b[:, i - 1] * decay
    return b


def declare_variables(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        # print self

        assert self.scope is not None
        with tf.variable_scope(self.scope, reuse=type(self.scope) != str) as self.scope:
            ret = func(*args, **kwargs)
            self.variables # touch it

        return ret

    return wrapper


def select(a, idx):
    shape = tf.shape(idx)
    # rank = len(idx.get_shape())
    num_elements = tf.reduce_prod(shape)
    base = tf.range(num_elements)
    num_classes = tf.shape(a)[-1]
    offset = tf.reshape(idx, [-1]) + base * num_classes
    # t = 1
    # for i in reversed(range(rank)):
    #     offset += base // t % shape[i] * t * num_classes
    #     t *= shape[i]

    return tf.reshape(tf.gather(tf.reshape(a, [-1]), offset), shape)
