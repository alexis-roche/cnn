import numpy as np
import vii
import time

import scipy.ndimage as nd
import keras
from keras.backend.tensorflow_backend import _to_tensor, get_value

import cnn

def get_weights(keras_conv):
    kernel, bias = keras_conv.get_weights()
    kernel = kernel.astype(cnn.FLOAT_DTYPE)
    bias = bias.astype(cnn.FLOAT_DTYPE)
    return kernel, bias

def run_layer(keras_layer, x):
    return get_value(keras_layer.call(_to_tensor(np.expand_dims(x, 0), cnn.FLOAT_DTYPE))).squeeze()    


def probe_time(func):
    def timer(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        dt = time.time() - t0
        print(' Time (%s): %f' % (func.__name__, dt))
        return res
    return timer

@probe_time
def convolve_image(*args):
    return cnn._convolve_image(*args)

@probe_time
def multi_convolve_image(*args):
    return cnn._multi_convolve_image(*args)

@probe_time
def relu_max_pool_image(*args):
    return cnn._relu_max_pool_image(*args)


img = vii.load_image('/home/alexis/artisan_data/pizza/item1/con5/pic01.png')


print('Test single convolution...')
src = img.intensity().astype(cnn.FLOAT_DTYPE)
kernel = np.ones((3, 3), dtype=cnn.FLOAT_DTYPE)/9
res = convolve_image(np.expand_dims(src, 2), np.expand_dims(kernel, 2), 0, 1, 1).squeeze()
res_nd = nd.convolve(src, kernel)
print ('Error with scipy.ndimage = %f' % np.max(np.abs(res-res_nd)[1:-1, 1:-1]))

print('Test dilated convolution...')
res_dil = convolve_image(np.expand_dims(src, 2), np.expand_dims(kernel, 2), 0, 2, 2).squeeze()

print('Test computation time for single convolution...')
kernel = np.ones((3, 3, 3), dtype=cnn.FLOAT_DTYPE) / 27
src = img.get_data().astype(cnn.FLOAT_DTYPE)
res = convolve_image(src, kernel, 0, 1, 1)

print('Test multiple convolutions...')
kernel = np.ones((3, 3, 3, 32), dtype=cnn.FLOAT_DTYPE) / 27
res32 = multi_convolve_image(src, kernel, np.zeros(32, dtype=cnn.FLOAT_DTYPE), 1, 1)
for i in range(32):
    assert np.min(res32[..., i] == res)

print('Test ReLU/Max pooling...')
res32bis = relu_max_pool_image(res32, 2, 2, 1, 1)

print('Test consistency with Keras convolution')
patch = img.get_data()[300:350, 400:450, :].astype(cnn.FLOAT_DTYPE) / 255
model = keras.models.load_model('feb2.h5')
conv = model.layers[0]
resK = run_layer(conv, patch)
kernel, bias = get_weights(conv)
res = multi_convolve_image(patch, kernel, bias, 1, 1)[1:-1,1:-1,:]
error = np.max(np.abs(res - resK))
print('Error with Keras/TensorFlow = %f' % error)

print('Test consistency with Keras ReLU max pooling')
jc = relu_max_pool_image(res, 2, 2, 1, 1)[::2,::2,:]
jcK = np.maximum(run_layer(keras.layers.MaxPooling2D(), resK), 0)
error = np.max(np.abs(jc - jcK))
print('Error with Keras/TensorFlow = %f' % error)

print('Test FCNN...')
data = img.get_data().astype(cnn.FLOAT_DTYPE) / 255
data = multi_convolve_image(data, kernel, bias, 1, 1)
data = relu_max_pool_image(data, 2, 2, 1, 1)
jc_fc = data[301:348:2, 401:448:2, :]
print('Error FCNN/CNN = %f' % np.max(np.abs(jc - jc_fc)))


"""
# Flatten shit
flat = model.layers[9]
dense = model.layers[10]
k, bias = get_weights(dense)
kernel = k.reshape(list(flat.input_shape[1:]) + [dense.output_shape[-1]])


### test "unflatten" works
x =  np.random.random(flat.input_shape[1:])
toto = np.sum(k * np.expand_dims(x.flatten(), 1), 0)
titi = np.sum(kernel * np.expand_dims(x, 3), (0, 1, 2))
assert np.min(titi==toto)

# Test RELU max shit
x = np.random.random((10, 12, 3)).astype(cnn.FLOAT_DTYPE)
z1 = relu_max_pool_image(x, 2, 2, 1, 1)

for i in range(3):
    assert np.max(x[0:2,0:2,i]) == z1[0,0,i]
    assert np.max(x[1:3,1:3,i]) == z1[1,1,i]
    assert np.max(x[2:4,1:3,i]) == z1[2,1,i]

z2 = relu_max_pool_image(x, 2, 2, 2, 2)

for i in range(3):
    assert np.max(x[0:4:2,0:4:2,i]) == z2[0,0,i]
    assert np.max(x[1:5:2,1:5:2,i]) == z2[1,1,i]
    assert np.max(x[2:6:2,1:5:2,i]) == z2[2,1,i]


### TEST dimension stuff
zob = lambda dim, kernel_size, pool_size : cnn.dim_after_pooling(cnn.dim_after_convolution(dim, kernel_size), pool_size)

def zobic(dim, kernel_size, pool_size, conv_layers):
    out = dim
    for i in range(conv_layers):
        out = zob(out, kernel_size, pool_size)
    return out
"""
