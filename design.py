import sys
import numpy as np


def dim_after_convolution(dim, kernel_size):
    return np.array(dim).astype(np.uint16) - kernel_size + 1


def dim_after_pooling(dim, pool_size):
    return np.ceil(dim_after_convolution(dim, pool_size) / pool_size).astype(np.uint16)


# Backward Python compatibility
def _py2_dim_after_pooling(dim, pool_size):
    return np.ceil(dim_after_convolution(dim, pool_size) / float(pool_size)).astype(np.uint16)

if int(sys.version[0]) < 3:
    dim_after_pooling = _py2_dim_after_pooling

    
def conv_layer_output_dim(dim, kernel_size, pool_size):
    return tuple(dim_after_pooling(dim_after_convolution(dim, kernel_size), pool_size))


def conv_layer_params(filters, kernel_size):
    return filters * (kernel_size ** 2 + 1)


def dense_layer_params(input_size, output_size):
    return (input_size + 1) * output_size


class Design(object):

    def __init__(self, input_size, conv_layers, dense_layers):

        self._conv_layers = conv_layers
        self._dense_layers = dense_layers
        conv_dims = [input_size, ]
        for idx in range(len(conv_layers)):
            conv_dims.append(conv_layer_output_dim(conv_dims[idx][0:2], *conv_layers[idx][1:]) + (conv_layers[idx][0],))
        self._conv_dims = tuple(conv_dims)

    @property
    def dims(self):
        return self._conv_dims + self._dense_layers
    
    def _conv_params(self):
        return tuple((conv_layer_params(*c[0:2]) for c in self._conv_layers))

    def _dense_params(self):
        input_size = (int(np.prod(self._conv_dims[-1])), ) + self._dense_layers[0:-1]
        return tuple((dense_layer_params(s_in, s_out) for s_in, s_out in zip(input_size, self._dense_layers)))

    @property
    def params(self):
        return self._conv_params() + self._dense_params()

