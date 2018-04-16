import sys 
import os
import gc

import numpy as np


from ._utils import (FLOAT_DTYPE,
                     _convolve_image,
                     _multi_convolve_image,
                     _relu_max_pool_image,
                     _get_opencl_device_info,
                     _opencl_test1d,
                     _opencl_convolve_image,
                     _opencl_multi_convolve_image,
                     _opencl_relu_max_pool_image)


def dim_after_convolution(dim, kernel_size):
    return np.array(dim).astype(np.uint16) - kernel_size + 1


def dim_after_pooling(dim, pool_size):
    return np.ceil(dim_after_convolution(dim, pool_size) / pool_size).astype(np.uint16)


def _index_range(x, dim, size):
    hl, hr = size / 2, (size + 1) / 2
    x0, x1 = x - hl, x + hr
    a0, a1 = 0, size
    if x0 < 0:
        a0 = -x0
        x0 = 0
    elif x1 > dim:
        a1 = dim - x1
        x1 = dim
    return a0, a1, x0, x1

    
def patch(data, x, y, size_x, size_y, dtype=float):
    a0, a1, x0, x1 = _index_range(x, data.shape[0], size_x)
    b0, b1, y0, y1 = _index_range(y, data.shape[1], size_y)    
    patch = np.zeros((size_x, size_y, 3), dtype=dtype)
    patch[a0:a1, b0:b1] = data[x0:x1, y0:y1]
    return patch


def softmax(x):
    tmp = np.exp(x)
    tmp /= np.expand_dims(np.sum(tmp, -1), -1)
    return tmp


# Backward Python compatibility
def _py2_dim_after_pooling(dim, pool_size):
    return np.ceil(dim_after_convolution(dim, pool_size) / float(pool_size)).astype(np.uint16)

if int(sys.version[0]) < 3:
    dim_after_pooling = _py2_dim_after_pooling



def configure_cnn(nclasses,
                  image_size,
                  conv_filters,
                  kernel_size,
                  pool_size,
                  dense_units,
                  dropout,
                  learning_rate,
                  decay):
    """
    Returns a sequential Keras model
    """
    import keras
    model = keras.models.Sequential()
    
    # Convolutional layers
    # The same kernel size and pool size are used in each layer.
    input_layer = True
    for f in conv_filters:
        if input_layer:
            kwargs = {'input_shape': list(image_size) + [3]}
        else:
            kwargs = {}
        model.add(keras.layers.Conv2D(f, (kernel_size, kernel_size), **kwargs))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
        input_layer = False

    # Densely connected layers
    model.add(keras.layers.Flatten())
    for n in dense_units:
        model.add(keras.layers.Dense(n))
        model.add(keras.layers.Activation('relu'))

    # Final layer
    if dropout > 0:
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(nclasses))
    model.add(keras.layers.Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=learning_rate, decay=decay)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def shuffle_and_split(x, y, prop_test):
    """
    Returns a tuple:
    x_train, y_train, x_test, y_test
    """
    idx = np.random.permutation(x.shape[0])
    size_train = int((1 - prop_test) * x.shape[0])
    x_test = x[size_train + 1:]
    y_test = y[size_train + 1:]
    x = x[0:size_train]
    y = y[0:size_train]
    return x, y, x_test, y_test


class ImageClassifier(object):

    def __init__(self, image_size, nclasses,
                 conv_filters=(32, 32, 64),
                 kernel_size=3,
                 pool_size=2,
                 dense_units=(64,)):

        self._model = None
        self._layer_index = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._image_size = tuple(image_size)
        self._nclasses = int(nclasses)
        self._conv_filters = tuple(conv_filters)
        self._kernel_size = int(kernel_size)
        self._dense_units = tuple(dense_units)
        self._pool_size = int(pool_size)
        
    def train(self, x, y,
              prop_test=.2,
              batch_size=16,
              epochs=50,
              dropout=.5,
              learning_rate=1e-4,
              decay=1e-6,
              class_weight=None):

        import keras
        y = keras.utils.to_categorical(y, self._nclasses)
        if prop_test == 0:
            self.x_train, self.y_train = x, y
            self.x_test, self.y_test = None, None
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = shuffle_and_split(x, y, prop_test=prop_test)

        self._model = configure_cnn(self._nclasses,
                                    self._image_size,
                                    self._conv_filters,
                                    self._kernel_size,
                                    self._pool_size,
                                    self._dense_units,
                                    dropout,
                                    learning_rate,
                                    decay)

        self._layer_index = tuple([i for i in range(len(self._model.layers)) if type(self._model.layers[i]) == keras.layers.Conv2D]\
                            + [i for i in range(len(self._model.layers)) if type(self._model.layers[i]) == keras.layers.Dense])
        
        if self.x_test is None:
            validation_data = None
        else:
            validation_data = (self.x_test, self.y_test)
        self._model.fit(self.x_train, self.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=class_weight,
                        validation_data=validation_data)
        
    def run(self, x):
        if self._model is None:
            raise ValueError('Model needs be trained first')
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        return self._model.predict(x).squeeze()


    def _label_map_brute_force(self, data):

        def myprint(s):
            sys.stdout.write(s)
            sys.stdout.flush()
            
        out = np.zeros(list(data.shape[0:2]) + [self._nclasses])
        lines = out.shape[1]
        for y in range(lines):
            myprint('%3.2f %% complete.\r' % ((100. * y) / lines))
            patches = np.array([patch(data, x, y, self._image_size[0], self._image_size[1]) for x in range(out.shape[0])])
            out[:, y, :] = self.run(patches)
        return out

    
    def label_map(self, data, device=None, groups=None, brute_force=False):

        if brute_force:
            return self._label_map_brute_force(data)
        
        # Shift the input image 
        sx, sy = self.fcnn_shift
        sdata = np.zeros(data.shape, dtype=FLOAT_DTYPE)
        sdata[sx:, sy:, :] = data[:-sx, :-sy:, :]

        # Run input data through fully convolutional network
        pm = self._label_map(sdata, device=device, groups=groups)

        # Set the borders to zero
        a, b = self.fcnn_borders
        pm[0:a, :, :] = 0
        pm[:, 0:a, :] = 0
        pm[-b:, :, :] = 0
        pm[:, -b:, :] = 0

        return pm

    def _label_map(self, data, device=None, groups=None):
        # Select actual convolution and max pooling routines
        if device is None:
            multi_convolve_image = _multi_convolve_image
            relu_max_pool_image = _relu_max_pool_image
            opencl_args_conv, opencl_args_relu = [], []
        else:
            multi_convolve_image = _opencl_multi_convolve_image
            relu_max_pool_image = _opencl_relu_max_pool_image
            if groups is None:
                groups = [1, 1, 1]
            elif len(groups) != 2:
                raise ValueError('groups need be a sequence of two integers')
            opencl_args_conv = [device] + list(groups)
            opencl_args_relu = opencl_args_conv + [1]

        # Run fully convolutional network
        pool_size, dil = self._pool_size, 1
        for i in range(len(self.layers)):
            kernel, bias = self.get_weights(i, fully_convolutional=True)
            data = multi_convolve_image(data, kernel, bias, dil, dil, *opencl_args_conv)
            # Reset pool size and dilation after convolution with first dense layer
            if i == len(self._conv_filters):
                pool_size = dil = 1
            if i < (len(self.layers) - 1):  # no max activation in last layer
                data = relu_max_pool_image(data, pool_size, pool_size, dil, dil, *opencl_args_relu)
            if i < len(self._conv_filters):
                dil *= 2
        return softmax(data)
    
    def evaluate(self):
        if self._model is None:
            raise ValueError('Model needs be trained first')
        if self.x_test is None:
            return
        return self._model.evaluate(self.x_test, self.y_test, verbose=1)
        
    def accuracy(self):
        loss, acc = self.evaluate()
        return acc

    def save(self, fname):
        if os.path.splitext(fname)[1] != 'h5':
            fname = fname + '.h5'
        self._model.save(fname)

    def purge(self):
        import gc
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        gc.collect()
        
    def get_weights(self, n, dtype=FLOAT_DTYPE, fully_convolutional=False):
        i = self._layer_index[n]
        kernel, bias = self._model.layers[i].get_weights()
        kernel = kernel.astype(dtype)
        bias = bias.astype(dtype)

        # Optionally, reshape the first densely connected layer for
        # use as a fully convolutional network
        if fully_convolutional:
            if n == len(self._conv_filters):
                flat = self._model.layers[i - 1]
                dense = self._model.layers[i]
                kernel = np.reshape(kernel, list(flat.input_shape[1:]) + [dense.output_shape[-1]])
            elif n > len(self._conv_filters):
                kernel = np.reshape(kernel, [1, 1] + list(kernel.shape))

        return kernel, bias
    
    @property
    def image_size(self):
        return self._image_size

    @property
    def nclasses(self):
        return self._nclasses

    @property
    def conv_filters(self):
        return self._conv_filters

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def dense_units(self):
        return self._dense_units

    @property
    def pool_size(self):
        return self._pool_size

    @property
    def layers(self):
        return tuple(['c' for i in range(len(self._conv_filters))] + ['d' for i in range(len(self._dense_units) + 1)])

    @property
    def fcnn_kernel_size(self):
        return self.get_weights(len(self._conv_filters), fully_convolutional=True)[0].shape[0]

    @property
    def fcnn_borders(self):
        steps = len(self._conv_filters)
        formula = lambda ks, fs, ps: (2 ** steps - 1) * (ks // 2 + ps // 2) + (2 ** steps) * (fs // 2)
        left = formula(self._kernel_size - 1, self._pool_size - 1, self.fcnn_kernel_size - 1)
        right = formula(self._kernel_size, self._pool_size, self.fcnn_kernel_size)
        return np.array((left, right))

    @property
    def fcnn_shift(self):
        steps = len(self._conv_filters)
        s = (self._kernel_size - 1) // 2 + (self._pool_size - 1) // 2
        if self._pool_size > 1:
            s *= (self._pool_size ** (steps + 1) - 1) // (self._pool_size - 1)
        else:
            s *= len(self._conv_filters)
        return np.array(self._image_size) // 2 - s


def load_image_classifier(h5file):
    import keras
    model = keras.models.load_model(h5file)
    conv2d_idx = [i for i in range(len(model.layers)) if type(model.layers[i]) == keras.layers.Conv2D]
    dense_idx = [i for i in range(len(model.layers)) if type(model.layers[i]) == keras.layers.Dense]  
    conv_filters = [model.layers[i].filters for i in conv2d_idx]
    dense_units = [model.layers[i].output_shape[-1] for i in dense_idx]
    kernel_size = model.layers[0].kernel_size[0]  # layer 0 has to be a convolution layer
    pool_size = model.layers[2].pool_size[0]  # layer 2 has to be a pooling layer
    nclasses = dense_units[-1]
    dense_units = dense_units[:-1]
    image_size = model.input_shape[1:3]
    C = ImageClassifier(image_size, nclasses, conv_filters, kernel_size, pool_size, dense_units)
    C._model = model
    C._layer_index = tuple(conv2d_idx + dense_idx)
    return C
