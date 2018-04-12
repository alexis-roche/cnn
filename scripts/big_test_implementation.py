import sys
import numpy as np
import time

import vii
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
def cpu_multi_convolve_image(*args):
    return cnn._multi_convolve_image(*args)


@probe_time
def cpu_relu_max_pool_image(*args):
    return cnn._relu_max_pool_image(*args)


@probe_time
def opencl_multi_convolve_image(*args):
    return cnn._opencl_multi_convolve_image(*args)


@probe_time
def opencl_relu_max_pool_image(*args):
    return cnn._opencl_relu_max_pool_image(*args)


###########################################################################

def zobic(steps, kernel_size, pool_size, final_kernel_size):
    formula = lambda steps, ks, ps, fs: (2 ** steps - 1) * (ks // 2 + ps // 2) + (2 ** steps) * (fs // 2)
    return formula(steps, kernel_size - 1, pool_size - 1, final_kernel_size - 1), formula(steps, kernel_size, pool_size, final_kernel_size)

###########################################################################

img = vii.load_image('/home/alexis/artisan_data/pizza/item1/con5/pic01.png')
classif = cnn.load_image_classifier('feb2.h5')  # 'mar6.h5'

OPENCL = False
DEVICE = 0
GROUPS = 25, 20, 1
if len(sys.argv) > 1:
    OPENCL = sys.argv[1] == 'opencl'


def multi_convolve_image(data, kernel, bias, dil_x, dil_y):
    if OPENCL:
        return opencl_multi_convolve_image(data, kernel, bias, dil_x, dil_y, DEVICE, *(GROUPS[0:2]))
    else:
        return cpu_multi_convolve_image(data, kernel, bias, dil_x, dil_y)


def relu_max_pool_image(data, size_x, size_y, dil_x, dil_y):
    if OPENCL:
        return opencl_relu_max_pool_image(data, size_x, size_y, dil_x, dil_y, DEVICE, *GROUPS)
    else:
        return cpu_relu_max_pool_image(data, size_x, size_y, dil_x, dil_y)

###########################################################################
    
print('CNN test')

x = np.random.randint(img.dims[0] - classif.image_size[0] + 1)
y = np.random.randint(img.dims[1] - classif.image_size[1] + 1)

data = img.get_data().astype(cnn.FLOAT_DTYPE)[x:(x + classif.image_size[0]), y:(y + classif.image_size[1])] / 255
gold = classif.run(data)

flow = data
for i in range(len(classif.conv_filters)):
    kernel, bias = classif.get_weights(i)
    flow = multi_convolve_image(flow, kernel, bias, 1, 1)[1:-1, 1:-1, :]
    flow = subsample(relu_max_pool_image(flow, classif.pool_size, classif.pool_size, 1, 1), 2)
flow = flow.flatten()

for i in range(len(classif.conv_filters), len(classif.layers)):
    kernel, bias = classif.get_weights(i)
    flow = np.sum(kernel * np.expand_dims(flow, 1), 0) + bias
    if i < (len(classif.layers) - 1):
        flow = np.maximum(flow, 0)

silver = softmax(flow)

print('error = %f' % np.max(np.abs(gold-silver))) 


############################################################################

print('FCNN test')


data = img.get_data().astype(cnn.FLOAT_DTYPE) / 255

"""
sx, sy = classif.fcnn_shift
data = np.zeros(img.get_data().shape, dtype=cnn.FLOAT_DTYPE)
data[sx:, sy:, :] = img.get_data()[:-sx, :-sy:, :]
data /= 255

pool_size, dil = classif.pool_size, 1
for i in range(len(classif.layers)):
    print('Processing %d-th convolution layer' % (i + 1))
    kernel, bias = classif.get_weights(i, fully_convolutional=True)
    print('Kernel shape: %d, %d, %d, %d' % kernel.shape)
    data = multi_convolve_image(data, kernel, bias, dil, dil)
    # Reset pool size and dilation after convolution with first dense layer
    if i == len(classif.conv_filters):
        pool_size = dil = 1
    print('Layer=%d, pool size=%d, dilation=%d' % (i, pool_size, dil))
    if i < (len(classif.layers) - 1):  # no max activation in last layer
        data = relu_max_pool_image(data, pool_size, pool_size, dil, dil)
    if i < len(classif.conv_filters):
        dil *= 2

label_map = softmax(data)
"""

label_map = classif.label_map(data)
silver_mask = label_map[..., 1]

###############################

tmp = silver_mask[x + classif.image_size[0] // 2, y + classif.image_size[1] // 2] - gold[1]
print ('FCNN error = %f' % tmp)

a, b = zobic(len(classif.conv_filters), classif.kernel_size, classif.pool_size, classif.final_kernel_size)

silver_mask[0:a, :] = 0
silver_mask[:, 0:a] = 0
silver_mask[-b:, :] = 0
silver_mask[:, -b:] = 0


#gold_mask = np.load('gold_fcnn.npy')

#silver = np.zeros(mask.shape, dtype=cnn.FLOAT_DTYPE)
#silver[10:, 10:] = mask[:-10, :-10]

