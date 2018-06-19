import numpy as np

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator


batch_get_value = lambda param, feed_dict, session: [session.run(p, feed_dict=feed_dict) for p in param]


class Optimizer(object):

    def __init__(self, obj, method='steepest', batch_size=32, shuffle=True, lr=1e-4, decay=1e-6):
        """
        obj is an image classifier object
        """
        x_train = obj.x_train
        y_train = obj.y_train
        self._model = obj._model

        gen = ImageDataGenerator()
        self._batches = gen.flow(x_train, y_train, batch_size=batch_size, shuffle=shuffle)

        self._grad = K.gradients(self._model.total_loss, self._model._collected_trainable_weights)
        self._hess = K.gradients(self._grad, self._model._collected_trainable_weights)

        self._batch_size = batch_size
        self._shuffle = shuffle

        with K.name_scope(self.__class__.__name__):
            self._lr = K.variable(lr, name='lr')
            self._decay = K.variable(decay, name='decay')
            self._iterations = K.variable(0, dtype='int64', name='iterations')
            self._update_function = K.function(
                self._model._feed_inputs + self._model._feed_targets + self._model._feed_sample_weights,
                [self._model.total_loss],
                updates=self.get_updates(),
                name='update_function')

        self._session = K.get_session()
        self._feed_dict = {}

    def _adjusted_lr(self):
        lr = self._lr
        return lr * (1. / (1. + self._decay * K.cast(self._iterations, K.dtype(self._decay))))

    def get_updates(self):
        updates = [K.update_add(self._iterations, 1)]
        new_param = [p - self._adjusted_lr() * g for p, g in zip(self._model._collected_trainable_weights, self._grad)]
        updates += [K.update(p, new_p) for p, new_p in zip(self._model._collected_trainable_weights, new_param)]
        return updates

    def _update_batch(self):
        x, y = next(self._batches)
        self._feed_dict = {self._model._feed_inputs[0]: x,
                           self._model._feed_targets[0]: y,
                           self._model._feed_sample_weights[0]: np.ones(x.shape[0])}
        return x, y
    
    def run(self, epochs=1):
        batches = len(self._batches.x) / self._batch_size + 1
        for e in range(epochs):
            for i in range(batches):
                x, y = self._update_batch()
                out = self._update_function([x] + [y] + [np.ones(x.shape[0])])
                print('Iteration: %d, Epoch: %d, Loss = %f' % (i + 1, e + 1, out[0]))

    @property
    def iterations(self):
        return K.get_value(self._iterations)

    @property
    def decay(self):
        return K.get_value(self._decay)

    @property
    def lr(self):
        return K.get_value(self._lr)
   
    @property
    def batch(self):
        if len(self._feed_dict) < 2:
            return None, None
        return self._feed_dict[self._model._feed_inputs[0]], self._feed_dict[self._model._feed_targets[0]]

    @property
    def param(self):
        return batch_get_value(self._model._collected_trainable_weights, self._feed_dict, self._session)

    @property
    def grad(self):
        if len(self._feed_dict) < 2:
            return None
        return batch_get_value(self._grad, self._feed_dict, self._session)

    @property
    def hess(self):
        if len(self._feed_dict) < 2:
            return None
        return batch_get_value(self._hess, self._feed_dict, self._session)

    @property
    def loss(self):
        return self._session.run(self._model.total_loss, feed_dict=self._feed_dict)


class RMSPropagation(Optimizer):

    def get_updates(self):
        updates = [K.update_add(self._iterations, 1)]
        new_param = [p - self._adjusted_lr() * g / (K.sqrt(K.square(g)) + K.epsilon()) for p, g in zip(self._model._collected_trainable_weights, self._grad)]
        updates += [K.update(p, new_p) for p, new_p in zip(self._model._collected_trainable_weights, new_param)]
        return updates
