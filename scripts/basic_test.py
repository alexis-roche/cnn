import time
import numpy as np

from cnn._utils import (FLOAT_DTYPE,
                        _get_opencl_device_info,
                        _opencl_test1d,
                        _convolve_image,
                        _opencl_convolve_image,
                        _multi_convolve_image,
                        _opencl_multi_convolve_image,
                        _relu_max_pool_image,
                        _opencl_relu_max_pool_image)
                       

dx, dy = 1, 1
kernel_size = 3
nchannels = 10
nfilters = 11
gx, gy = 22, 22


def disc(x, y):
    return np.max(np.abs(x - y))


info = _get_opencl_device_info(1)

##################### Test 1
x = np.random.rand(640).astype(FLOAT_DTYPE)
y = _opencl_test1d(x, 0, 40)
err = disc(x, y - 3)
print('Test1 error = %f' % err)

##################### Test 2
print('Number of work items: %d' % (gx*gy))

src = np.random.rand(640, 480, nchannels).astype(FLOAT_DTYPE)
kernel = np.random.rand(kernel_size, kernel_size, nchannels).astype(FLOAT_DTYPE)
bias = np.random.rand()
"""
src = np.ones((640, 480, 3))
kernel = np.ones((3,3,3))
bias = 0
"""
res = _convolve_image(src, kernel, bias, dx, dy)
res2 = _opencl_convolve_image(src, kernel, bias, dx, dy, 0, gx, gy)

print('res2.max() = %f' % res2.max())
err2 = disc(res, res2)
print('Test2 error = %f' % err2)


##################### Test 3
src = np.random.rand(640, 480, nchannels).astype(FLOAT_DTYPE)
kernels = np.random.rand(kernel_size, kernel_size, nchannels, nfilters).astype(FLOAT_DTYPE)
#biases =  np.ones(nfilters).astype(FLOAT_DTYPE)
biases =  np.random.rand(nfilters).astype(FLOAT_DTYPE)

t0 = time.time()
mres = _multi_convolve_image(src, kernels, biases, dx, dy)
print('Time CPU = %f' % (time.time() - t0))
t0 = time.time()
mres2 = _opencl_multi_convolve_image(src, kernels, biases, dx, dy, 0, gx, gy)
print('Time GPU = %f' % (time.time() - t0))
err3 = disc(mres, mres2)
print('Test3 error = %f' % err3)

##################### Test 4
x = np.random.rand(10, 11, 12).astype(FLOAT_DTYPE)
y = _relu_max_pool_image(x, 2, 2, 1, 1)
z = _opencl_relu_max_pool_image(x, 2, 2, 1, 1, 0, 10, 10, 1)
print('Test4 = %s' % np.min(z == y))
