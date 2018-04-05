import numpy as np
import vii
import time
import scipy.ndimage as nd
import keras
from keras.backend.tensorflow_backend import _to_tensor, get_value

import cnn


def softmax(x):
    tmp = np.exp(x)
    tmp /= np.expand_dims(np.sum(tmp, -1), -1)
    return tmp


def subsample(x, pool_size):
    # Make sure it works with pool size > 2 !!!!
    dx, dy = [int(p) for p in pool_size * (np.array(x.shape[0:2]) // pool_size)]
    return x[:dx:2, :dy:2]


def probe_time(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        dt = time.time() - t0
        print('Time (%s): %f' % (func.__name__, dt))
        return res
    return wrapper


@probe_time
def opencl_multi_convolve_image(*args):
    return cnn._opencl_multi_convolve_image(*args)


@probe_time
def relu_max_pool_image(*args):
    return cnn._relu_max_pool_image(*args)



###########################################################################

img = vii.load_image('/home/alexis/artisan_data/pizza/item1/con5/pic01.png')
classif = cnn.load_image_classifier('feb2.h5')  # 'mar6.h5'

###########################################################################
    
print('CNN test')

x = np.random.randint(img.dims[0] - classif.image_size[0] + 1)
y = np.random.randint(img.dims[1] - classif.image_size[1] + 1)

data = img.get_data().astype(cnn.FLOAT_DTYPE)[x:(x + classif.image_size[0]), y:(y + classif.image_size[1])] / 255

gold = classif.run(data)

zob = data
for i in range(len(classif.conv_filters)):
    kernel, bias = classif.get_weights(i)
    zob = cnn._multi_convolve_image(zob, kernel, bias, 1, 1)[1:-1, 1:-1, :]
    zob = subsample(cnn._relu_max_pool_image(zob, classif.pool_size, classif.pool_size, 1, 1), 2)
zob = zob.flatten()

for i in range(len(classif.conv_filters), len(classif.layers)):
    kernel, bias = classif.get_weights(i)
    zob = np.sum(kernel * np.expand_dims(zob, 1), 0) + bias
    if i < (len(classif.layers) - 1):
        zob = np.maximum(zob, 0)

silver = softmax(zob)

print('error = %f' % np.max(np.abs(gold-silver))) 

############################################################################

print('FCNN test')

sx, sy = classif.fcnn_shift

data = np.zeros(img.get_data().shape, dtype=cnn.FLOAT_DTYPE)
data[sx:, sy:, :] = img.get_data()[:-sx, :-sy:, :]
data /= 255

###data = img.get_data().astype(cnn.FLOAT_DTYPE) / 255

pool_size, dil = classif.pool_size, 1
for i in range(len(classif.layers)):
    print('Processing %d-th convolution layer' % (i + 1))
    kernel, bias = classif.get_weights(i, fully_convolutional=True)
    print('Kernel shape: %d, %d, %d, %d' % kernel.shape)
    data = opencl_multi_convolve_image(data, kernel, bias, dil, dil, 0, 25, 20)
    # Reset pool size and dilation after convolution with first dense layer
    if i == len(classif.conv_filters):
        pool_size = dil = 1
    print('Layer=%d, pool size=%d, dilation=%d' % (i, pool_size, dil))
    if i < (len(classif.layers) - 1):  # no max activation in last layer
        data = relu_max_pool_image(data, pool_size, pool_size, dil, dil)
    if i < len(classif.conv_filters):
        dil *= 2

silver_mask = softmax(data)[..., 1]

###############################

tmp = silver_mask[x + classif.image_size[0] // 2, y + classif.image_size[1] // 2] - gold[1]
print ('FCNN error = %f' % tmp)

gold_mask = np.load('gold_fcnn.npy')

#silver = np.zeros(mask.shape, dtype=cnn.FLOAT_DTYPE)
#silver[10:, 10:] = mask[:-10, :-10]


