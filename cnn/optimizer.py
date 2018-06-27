import numpy as np

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator


batch_get_value = lambda param, feed_dict, session: [session.run(p, feed_dict=feed_dict) for p in param]


class Optimizer(object):

    def __init__(self, obj, method='steepest', batch_size=32, shuffle=True, lr=1e-4, decay=1e-6, **constants):
        """
        obj is an image classifier object
        """
        x_train = obj.x_train
        y_train = obj.y_train
        self._model = obj._model

        self._batch_size = batch_size
        self._shuffle = shuffle
        gen = ImageDataGenerator()
        self._batches = gen.flow(x_train, y_train, batch_size=batch_size, shuffle=shuffle)

        self._init_param(**constants)

        self._grad = K.gradients(self._model.total_loss, self._model._collected_trainable_weights)
        self._hess = K.gradients(self._grad, self._model._collected_trainable_weights)

        with K.name_scope(self.__class__.__name__):
            self._lr = K.variable(lr, name='lr')
            self._decay = K.variable(decay, name='decay')
            self._iterations = K.variable(0, dtype='int64', name='iterations')
            self._update_function = K.function(
                self._model._feed_inputs + self._model._feed_targets + self._model._feed_sample_weights,
                [self._model.total_loss] + self._model.metrics_tensors,
                updates=self.get_updates(),
                name='update_function')

        self._session = K.get_session()
        self._feed_dict = {}

    def _init_param(self, **constants):
        self._constants = constants
        self._param = self._model._collected_trainable_weights

    def _adjusted_lr(self):
        lr = self._lr
        return lr * (1. / (1. + self._decay * K.cast(self._iterations, K.dtype(self._decay))))

    def _update_param(self):
        return [w - self._adjusted_lr() * g for w, g in zip(self._model._collected_trainable_weights, self._grad)]
    
    def get_updates(self):
        updates = [K.update_add(self._iterations, 1)]
        new_param = self._update_param()
        updates += [K.update(p, q) for p, q in zip(self._param, new_param)]
        return updates

    def _update_batch(self):
        x, y = next(self._batches)
        self._feed_dict = {self._model._feed_inputs[0]: x,
                           self._model._feed_targets[0]: y,
                           self._model._feed_sample_weights[0]: np.ones(x.shape[0])}
        return x, y

    def _num_batches(self):
        return len(self._batches.x) / self._batch_size + 1
    
    def run(self, epochs=1):
        for e in range(epochs):
            for i in range(self._num_batches()):
                x, y = self._update_batch()
                out = self._update_function([x] + [y] + [np.ones(x.shape[0])])
                print('Iteration: %d, Epoch: %d, Losses = %s' % (i + 1, e + 1, out))

    def get_config(self):
        config = {'lr': K.get_value(self._lr),
                  'decay': K.get_value(self._decay)}
        config.update(self._constants)
        return config
        
    @property
    def iterations(self):
        return K.get_value(self._iterations)

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

    @property
    def record(self):
        return np.array(self._record)

    
class RMSPropagation(Optimizer):

    def _init_param(self, **constants):
        self._constants = {'epsilon': K.epsilon(), 'rho': 0.9}
        self._constants.update(constants)
        self._param = self._model._collected_trainable_weights +\
                      [K.zeros(K.int_shape(w), dtype=K.dtype(w)) for w in self._model._collected_trainable_weights]
    
    def _update_param(self):
        n = len(self._model._collected_trainable_weights)
        rho, epsilon = self._constants['rho'], self._constants['epsilon']
        new_weights, new_scales = [], []
        for w, s, g in zip(self._param[0:n], self._param[n:], self._grad):
            new_s = rho * s + (1 - rho) * K.square(g)
            new_w = w - self._adjusted_lr() * g / (K.sqrt(new_s) + epsilon)
            new_scales.append(new_s)
            new_weights.append(new_w)
        return new_weights + new_scales


class AverageEP(Optimizer):

    def _init_param(self, **constants):
        self._constants = {'min_var': 1e-2, 'init_var': 1}
        self._constants.update(constants)
        self._param = self._model._collected_trainable_weights +\
                      [K.variable(np.full(K.int_shape(w), 1. / self._constants['init_var'], dtype=K.dtype(w))) for w in self._model._collected_trainable_weights]
    
    def _update_param(self):
        n = len(self._model._collected_trainable_weights)
        min_var = self._constants['min_var']
        n_batches = self._num_batches()
        lr = self._adjusted_lr()
        new_weights, new_scales = [], []
        for w, s, g, h in zip(self._param[0:n], self._param[n:], self._grad, self._hess):
            new_s = (1 - lr / float(n_batches)) * s + lr * K.maximum(h, min_var)
            new_w = w - lr * g / new_s
            new_scales.append(new_s)
            new_weights.append(new_w)
        return new_weights + new_scales

