import os
import glob
import gc

import numpy as np

import keras

from ._run import FLOAT_DTYPE

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
        if n < len(self._conv_filters):
            i = 3 * n
        else:
            i = 3 * n + 1
        kernel, bias = self._model.layers[i].get_weights()
        kernel = kernel.astype(dtype)
        bias = bias.astype(dtype)

        # Optionally, reshape the first densely connected layer for
        # use as a fully convolutional network
        if fully_convolutional:
            if n == len(self._conv_filters):
                i = 3 * len(self._conv_filters)
                flat = self._model.layers[i]
                dense = self._model.layers[i + 1]
                kernel = np.reshape(kernel, list(flat.input_shape[1:]) + [dense.output_shape[-1]])
            elif n > len(self._conv_filters):
                kernel = np.reshape(kernel, [1, 1] + list(kernel.shape))

        return kernel, bias

    def _get_image_size(self):
        return self._image_size

    def _get_nclasses(self):
        return self._nclasses
    
    def _get_conv_filters(self):
        return self._conv_filters

    def _get_kernel_size(self):
        return self._kernel_size

    def _get_dense_units(self):
        return self._dense_units

    def _get_pool_size(self):
        return self._pool_size

    def _get_layers(self):
        return len(self._conv_filters) + len(self._dense_units) + 1

    image_size = property(_get_image_size)
    nclasses = property(_get_nclasses)
    conv_filters = property(_get_conv_filters)
    kernel_size = property(_get_kernel_size)
    dense_units = property(_get_dense_units)
    pool_size = property(_get_pool_size)
    layers = property(_get_layers)

def load_image_classifier(h5file):
    model = keras.models.load_model(h5file)
    conv_filters = [layer.filters for layer in model.layers if type(layer) == keras.layers.convolutional.Conv2D]
    dense_units = [layer.output_shape[-1] for layer in model.layers if type(layer) == keras.layers.Dense]
    kernel_size = model.layers[0].kernel_size[0]  # layer 0 has to be a convolution layer
    pool_size = model.layers[2].pool_size[0]  # layer 2 has to be a pooling layer
    nclasses = dense_units[-1]
    dense_units = dense_units[:-1]
    image_size = model.input_shape[1:3]
    C = ImageClassifier(image_size, nclasses, conv_filters, kernel_size, pool_size, dense_units)
    C._model = model
    return C


