import numpy as np


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def merge_dict(*args, **kwds):
    for d in args:
        for k, v in d.items():
            kwds[k] = v
    return kwds
