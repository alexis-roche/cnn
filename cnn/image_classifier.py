import os
import glob
import gc

import numpy as np

import keras



def configure_cnn(nclasses,
                  image_size,
                  conv_filters,
                  kernel_size,
                  pooling_size,
                  dense_units,
                  dropout,
                  learning_rate,
                  decay):
    """
    Returns a sequential Keras model
    """
    model = keras.models.Sequential()
    
    # Convolutional layers
    # The same kernel size and pooling size are used in each layer.
    input_layer = True
    for f in conv_filters:
        if input_layer:
            kwargs = {'input_shape': list(image_size) + [3]}
        else:
            kwargs = {}
        model.add(keras.layers.Conv2D(f, (kernel_size, kernel_size), **kwargs))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(pooling_size, pooling_size)))
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

    def __init__(self, x, y, 
                 conv_filters=(32, 32, 64),
                 kernel_size=3,
                 pooling_size=2,
                 dense_units=(64,),
                 dropout=.5,
                 learning_rate=1e-4,
                 decay=1e-6, 
                 prop_test=0):

        nclasses = int(y.max()) + 1
        image_size = x.shape[1:3]
        self.model = configure_cnn(nclasses,
                                   image_size,
                                   conv_filters,
                                   kernel_size,
                                   pooling_size,
                                   dense_units,
                                   dropout,
                                   learning_rate,
                                   decay)

        y = keras.utils.to_categorical(y, nclasses)
        if prop_test == 0:
            self.x_train, self.y_train = x, y
            self.x_test, self.y_test = None, None
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = shuffle_and_split(x, y, prop_test=prop_test)
        self.trained = False

    def train(self, batch_size, epochs, class_weight=None):
        if self.x_test is None:
            validation_data = None
        else:
            validation_data = (self.x_test, self.y_test)
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       class_weight=class_weight,
                       validation_data=validation_data)
        self.trained = True
        
    def run(self, x):
        return self.model.predict(x)

    def evaluate(self):
        if self.x_test is None:
            return
        return self.model.evaluate(self.x_test, self.y_test, verbose=1)
        
    def accuracy(self):
        loss, acc = self.evaluate()
        return acc

    def save(self, fname):
        if os.path.splitext(fname)[1] != 'h5':
            fname = fname + '.h5'
        self.model.save(fname)

    def purge(self):
        import gc
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        gc.collect()


