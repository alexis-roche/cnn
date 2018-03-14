# -*- Mode: Python -*-  

__version__ = '0.1'

# Includes
import numpy as np
cimport numpy as np
import os

# Externals
cdef extern from "utils.h":

    ctypedef float FLOAT

    ctypedef struct array1d:
        size_t dim
        size_t off
        FLOAT* data

    ctypedef struct array2d:
        size_t dimx
        size_t dimy
        size_t offx
        size_t offy
        FLOAT* data

    ctypedef struct array3d:
        size_t dimx
        size_t dimy
        size_t dimz
        size_t offx
        size_t offy
        size_t offz
        FLOAT* data

    ctypedef struct array4d:
        size_t dimx
        size_t dimy
        size_t dimz
        size_t dimt
        size_t offx
        size_t offy
        size_t offz
        size_t offt
        FLOAT* data

    void convolve_image(array3d* src, array3d* kernel, FLOAT bias,
                        unsigned int dil_x, unsigned int dil_y,
                        array2d* res)
    void multi_convolve_image(array3d* src, array4d* kernels, array1d* biases,
                              unsigned int dil_x, unsigned int dil_y,
                              array3d* res)
    void relu_max_pool_image(array3d* src,
                             unsigned int size_x, unsigned int size_y,
                             unsigned int dil_x, unsigned int dil_y,
                             array3d* res)

    void basic_test1d(array1d* src,
	              array1d* res,
	              char* fname)

    void gpu_convolve_image(array3d* src,
		            array3d* kernel,
		            FLOAT bias,
		            unsigned int dil_x,
		            unsigned int dil_y,
		            array2d* res,
		            char* fname)

    
    
# Initialize numpy
np.import_array()

# Global variables
FLOAT_DTYPE = 'float%s' % (8 * sizeof(FLOAT))

# Functions
cdef to_array1d(np.ndarray A, array1d* a_ptr):
    a_ptr[0].dim = A.shape[0]
    a_ptr[0].off = A.strides[0] / sizeof(FLOAT)
    a_ptr[0].data = <FLOAT*>A.data

    
cdef to_array2d(np.ndarray A, array2d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].offx = A.strides[0] / sizeof(FLOAT)
    a_ptr[0].offy = A.strides[1] / sizeof(FLOAT)
    a_ptr[0].data = <FLOAT*>A.data

    
cdef to_array3d(np.ndarray A, array3d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].dimz = A.shape[2]
    a_ptr[0].offx = A.strides[0] / sizeof(FLOAT)
    a_ptr[0].offy = A.strides[1] / sizeof(FLOAT)
    a_ptr[0].offz = A.strides[2] / sizeof(FLOAT)
    a_ptr[0].data = <FLOAT*>A.data

    
cdef to_array4d(np.ndarray A, array4d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].dimz = A.shape[2]
    a_ptr[0].dimt = A.shape[3]
    a_ptr[0].offx = A.strides[0] / sizeof(FLOAT)
    a_ptr[0].offy = A.strides[1] / sizeof(FLOAT)
    a_ptr[0].offz = A.strides[2] / sizeof(FLOAT)
    a_ptr[0].offt = A.strides[3] / sizeof(FLOAT)
    a_ptr[0].data = <FLOAT*>A.data


    
def _convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                    np.ndarray[FLOAT, ndim=3] Kernel not None,
                    FLOAT bias,
                    unsigned int fx,
                    unsigned int fy):
    cdef array3d src
    cdef array3d kernel
    cdef array2d res
    #
    # check dimensions!!!
    #
    Res = np.zeros([Src.shape[0], Src.shape[1]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Kernel, &kernel)
    to_array2d(Res, &res)
    convolve_image(&src, &kernel, bias, fx, fy, &res)    
    return Res


def _multi_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                          np.ndarray[FLOAT, ndim=4] Kernels not None,
                          np.ndarray[FLOAT, ndim=1] Biases not None,
                          unsigned int fx,
                          unsigned int fy):
    cdef array3d src
    cdef array4d kernels
    cdef array1d biases
    cdef array3d res
    #
    # check dimensions!!!
    #
    Res = np.zeros([Src.shape[0], Src.shape[1], Kernels.shape[3]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array4d(Kernels, &kernels)
    to_array1d(Biases, &biases)
    to_array3d(Res, &res)
    multi_convolve_image(&src, &kernels, &biases, fx, fy, &res)
    return Res


def _relu_max_pool_image(np.ndarray[FLOAT, ndim=3] Src not None,
                         unsigned int sx,
                         unsigned int sy,
                         unsigned int fx,
                         unsigned int fy):

    cdef array3d src
    cdef array3d res
    Res = np.zeros([Src.shape[0], Src.shape[1], Src.shape[2]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Res, &res)
    relu_max_pool_image(&src, sx, sy, fx, fy, &res)
    return Res


def get_opencl_file():
    return os.path.join(os.path.split(__file__)[0], 'utils.cl')


def _basic_test1d(np.ndarray[FLOAT, ndim=1] Src not None):
    cdef array1d src
    cdef array1d res
    Res = np.zeros(len(Src), dtype=Src.dtype)
    to_array1d(Src, &src)
    to_array1d(Res, &res)
    opencl_file = get_opencl_file()
    basic_test1d(&src, &res, <char*>opencl_file)
    return Res


def _gpu_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                        np.ndarray[FLOAT, ndim=3] Kernel not None,
                        FLOAT bias,
                        unsigned int fx,
                        unsigned int fy):
    cdef array3d src
    cdef array3d kernel
    cdef array2d res
    #
    # check dimensions!!!
    #
    Res = np.zeros([Src.shape[0], Src.shape[1]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Kernel, &kernel)
    to_array2d(Res, &res)
    opencl_file = get_opencl_file()
    gpu_convolve_image(&src, &kernel, bias, fx, fy, &res, <char*>opencl_file) 
    return Res
