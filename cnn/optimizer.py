import numpy as np

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator


batch_get_value = lambda param, feed_dict, session: [session.run(p, feed_dict=feed_dict) for p in param]


class Optimizer(object):

    def __init__(self, obj, method='steepest', sample_weight=None, class_weight=None, batch_size=32, shuffle=True, lr=1e-4, decay=1e-6, **constants):
        """
        obj is an image classifier object
        """
        self._model = obj._model

        # Validate user data
        x, y, w = self._model._standardize_user_data(obj.x_train, obj.y_train,
                                                     sample_weight=sample_weight,
                                                     class_weight=class_weight,
                                                     batch_size=batch_size)
        x, y, w = x[0], y[0], w[0]

        self._x = x
        self._y = y
        self._w = w

        self._batch_size = batch_size
        self._shuffle = shuffle
        gen = ImageDataGenerator()
        self._batches = gen.flow(x, y, sample_weight=w, batch_size=batch_size, shuffle=shuffle)

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
        return [x - self._adjusted_lr() * g for x, g in zip(self._model._collected_trainable_weights, self._grad)]
    
    def get_updates(self):
        updates = [K.update_add(self._iterations, 1)]
        new_param = self._update_param()
        updates += [K.update(p, q) for p, q in zip(self._param, new_param)]
        return updates

    def _update_batch(self):
        x, y, w = next(self._batches)
        self._feed_dict = {self._model._feed_inputs[0]: x,
                           self._model._feed_targets[0]: y,
                           self._model._feed_sample_weights[0]: w}
        return x, y, w

    def _num_batches(self):
        return len(self._batches.x) / self._batch_size + 1
    
    def run(self, epochs=1):
        for e in range(epochs):
            for i in range(self._num_batches()):
                x, y, w = self._update_batch()
                out = self._update_function([x] + [y] + [w])
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
        return batch_get_value(self._param, self._feed_dict, self._session)

    @property
    def loc(self):
        return batch_get_value(self._param[0:len(self._model._collected_trainable_weights)], self._feed_dict, self._session)

    @property
    def prec(self):
        n = len(self._model._collected_trainable_weights)
        if len(self._param) > n:
            return batch_get_value(self._param[n:], self._feed_dict, self._session)
        else:
            return None
    
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
                      [K.zeros(K.int_shape(x), dtype=K.dtype(x)) for x in self._model._collected_trainable_weights]
    
    def _update_param(self):
        n = len(self._model._collected_trainable_weights)
        rho, epsilon = self._constants['rho'], self._constants['epsilon']
        new_loc, new_prec = [], []
        for x, p, g in zip(self._param[0:n], self._param[n:], self._grad):
            new_p = rho * p + (1 - rho) * K.square(g)
            new_x = x - self._adjusted_lr() * g / (K.sqrt(new_p) + epsilon)
            new_prec.append(new_p)
            new_loc.append(new_x)
        return new_loc + new_prec


class AverageEP(Optimizer):

    def _init_param(self, **constants):
        self._constants = {'max_var': 1e2, 'init_var': 1}
        self._constants.update(constants)
        self._param = self._model._collected_trainable_weights +\
                      [K.variable(np.full(K.int_shape(w), 1. / self._constants['init_var'], dtype=K.dtype(w))) for w in self._model._collected_trainable_weights]
    
    def _update_param(self):
        n = len(self._model._collected_trainable_weights)
        max_var = self._constants['max_var']
        n_batches = self._num_batches()
        lr = self._adjusted_lr()
        new_loc, new_prec = [], []
        for x, p, g, h in zip(self._param[0:n], self._param[n:], self._grad, self._hess):
            new_p = (1 - lr / float(n_batches)) * p + lr * K.maximum(h, 1. / max_var)
            new_x = x - lr * g / new_p
            new_prec.append(new_p)
            new_loc.append(new_x)
        return new_loc + new_prec

