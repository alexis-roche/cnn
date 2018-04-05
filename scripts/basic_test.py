import numpy as np
import cnn
import time


dx, dy = 1, 1
kernel_size = 3
nchannels = 10
nfilters = 11
gx, gy = 22, 22


def disc(x, y):
    return np.max(np.abs(x - y))


info = cnn._get_opencl_device_info(1)

##################### Test 1
x = np.random.rand(640).astype(cnn.FLOAT_DTYPE)
y = cnn._opencl_test1d(x, 0, 40)
err = disc(x, y - 3)
print('Test1 error = %f' % err)

##################### Test 2
print('Number of work items: %d' % (gx*gy))

src = np.random.rand(640, 480, nchannels).astype(cnn.FLOAT_DTYPE)
kernel = np.random.rand(kernel_size, kernel_size, nchannels).astype(cnn.FLOAT_DTYPE)
bias = np.random.rand()
"""
src = np.ones((640, 480, 3))
kernel = np.ones((3,3,3))
bias = 0
"""
res = cnn._convolve_image(src, kernel, bias, dx, dy)
res2 = cnn._opencl_convolve_image(src, kernel, bias, dx, dy, 0, gx, gy)

print('res2.max() = %f' % res2.max())
err2 = disc(res, res2)
print('Test2 error = %f' % err2)


##################### Test 3

src = np.random.rand(640, 480, nchannels).astype(cnn.FLOAT_DTYPE)
kernels = np.random.rand(kernel_size, kernel_size, nchannels, nfilters).astype(cnn.FLOAT_DTYPE)
#biases =  np.ones(nfilters).astype(cnn.FLOAT_DTYPE)
biases =  np.random.rand(nfilters).astype(cnn.FLOAT_DTYPE)

t0 = time.time()
mres = cnn._multi_convolve_image(src, kernels, biases, dx, dy)
print('Time CPU = %f' % (time.time() - t0))
t0 = time.time()
mres2 = cnn._opencl_multi_convolve_image(src, kernels, biases, dx, dy, 0, gx, gy)
print('Time GPU = %f' % (time.time() - t0))
err3 = disc(mres, mres2)
print('Test3 error = %f' % err3)

