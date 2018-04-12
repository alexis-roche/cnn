import sys
import time
import numpy as np

import vii
import cnn


GROUPS = 25, 20, 1

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

fimg = '/home/alexis/artisan_data/pizza/item1/con5/pic01.png'
device = 0
brute_force = False
if len(sys.argv) > 1:
    fimg = sys.argv[1]
    if len(sys.argv) > 2:
        device = int(sys.argv[2])
        if device < 0:
            device = None
img = vii.load_image(fimg)
classif = cnn.load_image_classifier('feb2.h5')  # 'mar6.h5'



def multi_convolve_image(data, kernel, bias, dil_x, dil_y):
    if device < 0:
        return cpu_multi_convolve_image(data, kernel, bias, dil_x, dil_y)
    else:
        return opencl_multi_convolve_image(data, kernel, bias, dil_x, dil_y, device, *(GROUPS[0:2]))


def relu_max_pool_image(data, size_x, size_y, dil_x, dil_y):
    if device < 0:
        return cpu_relu_max_pool_image(data, size_x, size_y, dil_x, dil_y)
    else:
        return opencl_relu_max_pool_image(data, size_x, size_y, dil_x, dil_y, device, *GROUPS)


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

silver = cnn.softmax(flow)

print('error = %f' % np.max(np.abs(gold - silver))) 
