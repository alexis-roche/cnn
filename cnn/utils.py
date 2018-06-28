import numpy as np


def balanced_class_weight(y):
    classes = np.unique(y)
    n0 = np.sum(y == classes[0])
    out = {classes[0]: 1.}
    for i in range(1, len(classes)):
        out[classes[i]] = n0 / float(np.sum(y == classes[i]))
    return out
    
