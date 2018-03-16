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

    array3d slice3d(array4d* a4d, unsigned int t, FLOAT* data, unsigned char from_buffer)
    array2d slice2d(array3d* a3d, unsigned int z, FLOAT* data, unsigned char from_buffer)
    array1d slice1d(array2d* a2d, unsigned int y, FLOAT* data, unsigned char from_buffer)
        
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

    void gpu_basic_test1d(array1d* src,
	                  array1d* res,
	                  char* fname,
                          unsigned int groups)

    void gpu_convolve_image(array3d* src,
		            array3d* kernel,
		            FLOAT bias,
		            unsigned int dil_x,
		            unsigned int dil_y,
		            array2d* res,
		            char* fname,
			    unsigned int groups_x,
                            unsigned int groups_y)

    void gpu_multi_convolve_image(array3d* src,
				  array4d* kernels,
				  array1d* biases,
				  unsigned int dil_x,
				  unsigned int dil_y,
				  array3d* res,
				  char* fname,
				  unsigned int groups_x,
                                  unsigned int groups_y)

    
    
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


def _slice3d(np.ndarray[FLOAT, ndim=4] Src not None,
             unsigned int t):
    cdef array4d src
    cdef array3d res
    Res = np.zeros([Src.shape[0], Src.shape[1], Src.shape[2]], dtype=Src.dtype)
    to_array4d(Src, &src)
    to_array3d(Res, &res)
    res = slice3d(&src, t, res.data, 0) 
    return Res

def _slice2d(np.ndarray[FLOAT, ndim=3] Src not None,
             unsigned int z):
    cdef array3d src
    cdef array2d res
    Res = np.zeros([Src.shape[0], Src.shape[1]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array2d(Res, &res)
    res = slice2d(&src, z, res.data, 0) 
    return Res

def _slice1d(np.ndarray[FLOAT, ndim=2] Src not None,
             unsigned int y):
    cdef array2d src
    cdef array1d res
    Res = np.zeros(Src.shape[0], dtype=Src.dtype)
    to_array2d(Src, &src)
    to_array1d(Res, &res)
    res = slice1d(&src, y, res.data, 0) 
    return Res


def _convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                    np.ndarray[FLOAT, ndim=3] Kernel not None,
                    FLOAT bias,
                    unsigned int dil_x,
                    unsigned int dil_y):
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
    convolve_image(&src, &kernel, bias, dil_x, dil_y, &res)    
    return Res


def _multi_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                          np.ndarray[FLOAT, ndim=4] Kernels not None,
                          np.ndarray[FLOAT, ndim=1] Biases not None,
                          unsigned int dil_x,
                          unsigned int dil_y):
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
    multi_convolve_image(&src, &kernels, &biases, dil_x, dil_y, &res)
    return Res


def _relu_max_pool_image(np.ndarray[FLOAT, ndim=3] Src not None,
                         unsigned int size_x,
                         unsigned int size_y,
                         unsigned int dil_x,
                         unsigned int dil_y):

    cdef array3d src
    cdef array3d res
    Res = np.zeros([Src.shape[0], Src.shape[1], Src.shape[2]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Res, &res)
    relu_max_pool_image(&src, size_x, size_y, dil_x, dil_y, &res)
    return Res


def _basic_test1d(np.ndarray[FLOAT, ndim=1] Src not None, unsigned int groups):
    cdef array1d src
    cdef array1d res
    Res = np.zeros(len(Src), dtype=Src.dtype)
    to_array1d(Src, &src)
    to_array1d(Res, &res)
    opencl_file = os.path.join(os.path.split(__file__)[0], 'basic_test1d.cl')
    gpu_basic_test1d(&src, &res, <char*>opencl_file, groups)
    return Res


def _gpu_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                        np.ndarray[FLOAT, ndim=3] Kernel not None,
                        FLOAT bias,
                        unsigned int dil_x,
                        unsigned int dil_y,
                        unsigned int groups_x,
                        unsigned int groups_y):
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
    opencl_file = os.path.join(os.path.split(__file__)[0], 'convolve_image.cl')
    gpu_convolve_image(&src, &kernel, bias, dil_x, dil_y, &res, <char*>opencl_file, groups_x, groups_y) 
    return Res


def _gpu_multi_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                              np.ndarray[FLOAT, ndim=4] Kernels not None,
                              np.ndarray[FLOAT, ndim=1] Biases not None,
                              unsigned int dil_x,
                              unsigned int dil_y,
                              unsigned int groups_x,
                              unsigned int groups_y):
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
    opencl_file = os.path.join(os.path.split(__file__)[0], 'convolve_image.cl')
    gpu_multi_convolve_image(&src, &kernels, &biases, dil_x, dil_y, &res, <char*>opencl_file, groups_x, groups_y) 
    return Res
