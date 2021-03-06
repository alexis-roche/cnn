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
        unsigned char owner
        
    ctypedef struct array2d:
        size_t dimx
        size_t dimy
        size_t offx
        size_t offy
        FLOAT* data
        unsigned char owner
        
    ctypedef struct array3d:
        size_t dimx
        size_t dimy
        size_t dimz
        size_t offx
        size_t offy
        size_t offz
        FLOAT* data
        unsigned char owner
        
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
        unsigned char owner

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

    
cdef extern from "opencl_utils.h":

    ctypedef enum opencl_device_type:
        OPENCL_DEVICE_TYPE_DEFAULT = 0,
        OPENCL_DEVICE_TYPE_CPU = 1,
        OPENCL_DEVICE_TYPE_GPU = 2,
        OPENCL_DEVICE_TYPE_ACCELERATOR = 3,
        OPENCL_DEVICE_TYPE_ALL = 4

    ctypedef struct opencl_device_info:
        unsigned int max_work_group_size
        unsigned int max_work_item_dimensions
        unsigned int* max_work_item_sizes

    opencl_device_info* opencl_device_info_new(int type)
    void opencl_device_info_delete(opencl_device_info* thisone)
    void opencl_test1d(array1d* src,
	               array1d* res,
	               char* source_file,
                       opencl_device_type device_type,
                       unsigned int groups)
    void opencl_convolve_image(array3d* src,
		               array3d* kernel,
		               FLOAT bias,
		               unsigned int dil_x,
		               unsigned int dil_y,
		               array2d* res,
		               char* source_file,
                               opencl_device_type device_type,
			       unsigned int groups_x,
                               unsigned int groups_y)
    void opencl_multi_convolve_image(array3d* src,
				     array4d* kernels,
				     array1d* biases,
				     unsigned int dil_x,
				     unsigned int dil_y,
				     array3d* res,
				     char* source_file,
                                     opencl_device_type device_type,
				     unsigned int groups_x,
                                     unsigned int groups_y)
    void opencl_relu_max_pool_image(array3d* src,
				    unsigned int size_x,
				    unsigned int size_y,
				    unsigned int dil_x,
				    unsigned int dil_y,
				    array3d* res,
				    char* source_file,
				    opencl_device_type device_type,
				    unsigned int groups_x,
				    unsigned int groups_y,
				    unsigned int groups_z)
    
    
# Initialize numpy
np.import_array()

# Global variables
_SIZEOF_FLOAT = sizeof(FLOAT)
FLOAT_DTYPE = 'float%s' % (8 * _SIZEOF_FLOAT)

# Functions
cdef to_array1d(np.ndarray A, array1d* a_ptr):
    a_ptr[0].dim = A.shape[0]
    a_ptr[0].off = A.strides[0] / _SIZEOF_FLOAT
    a_ptr[0].data = <FLOAT*>A.data
    a_ptr[0].owner = 0

    
cdef to_array2d(np.ndarray A, array2d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].offx = A.strides[0] / _SIZEOF_FLOAT
    a_ptr[0].offy = A.strides[1] / _SIZEOF_FLOAT
    a_ptr[0].data = <FLOAT*>A.data
    a_ptr[0].owner = 0

    
cdef to_array3d(np.ndarray A, array3d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].dimz = A.shape[2]
    a_ptr[0].offx = A.strides[0] / _SIZEOF_FLOAT
    a_ptr[0].offy = A.strides[1] / _SIZEOF_FLOAT
    a_ptr[0].offz = A.strides[2] / _SIZEOF_FLOAT
    a_ptr[0].data = <FLOAT*>A.data
    a_ptr[0].owner = 0

    
cdef to_array4d(np.ndarray A, array4d* a_ptr):
    a_ptr[0].dimx = A.shape[0]
    a_ptr[0].dimy = A.shape[1]
    a_ptr[0].dimz = A.shape[2]
    a_ptr[0].dimt = A.shape[3]
    a_ptr[0].offx = A.strides[0] / _SIZEOF_FLOAT
    a_ptr[0].offy = A.strides[1] / _SIZEOF_FLOAT
    a_ptr[0].offz = A.strides[2] / _SIZEOF_FLOAT
    a_ptr[0].offt = A.strides[3] / _SIZEOF_FLOAT
    a_ptr[0].data = <FLOAT*>A.data
    a_ptr[0].owner = 0
    

def _convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                    np.ndarray[FLOAT, ndim=3] Kernel not None,
                    FLOAT bias,
                    unsigned int dil_x,
                    unsigned int dil_y):
    cdef array3d src
    cdef array3d kernel
    cdef array2d res
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
    if Kernels.shape[3] != Biases.shape[0]:
        raise ValueError('Inconsistent kernel and bias arrays')
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


def _get_opencl_device_info(opencl_device_type device_type):
    cdef opencl_device_info* info
    cdef unsigned int i
    
    if device_type == OPENCL_DEVICE_TYPE_CPU:
        Device_type = 'cpu'
    elif device_type == OPENCL_DEVICE_TYPE_GPU:
        Device_type = 'gpu'
    elif device_type == OPENCL_DEVICE_TYPE_ACCELERATOR:
        Device_type = 'accelerator'
    elif device_type == OPENCL_DEVICE_TYPE_ALL:
        Device_type = 'all'
    else:
        Device_type = 'default'
    
    info = opencl_device_info_new(device_type)

    Aux = np.zeros(info[0].max_work_item_dimensions, dtype=np.uint)
    for i in range(info[0].max_work_item_dimensions):
        Aux[i] = info[0].max_work_item_sizes[i]
    
    out = {'device_type': Device_type,
           'max_work_group_size': info[0].max_work_group_size,
           'max_work_item_dimensions': info[0].max_work_item_dimensions,
           'max_work_item_sizes': Aux}

    opencl_device_info_delete(info)
    return out


def get_opencl_source_file(name):
    if _SIZEOF_FLOAT == 8:
        fname = name + '_d' + '.cl'
    else:
        fname = name + '.cl'
    return os.path.join(os.path.split(__file__)[0], 'opencl', fname).encode()


def _opencl_test1d(np.ndarray[FLOAT, ndim=1] Src not None,
                   opencl_device_type device_type,
                   unsigned int groups):
    cdef array1d src
    cdef array1d res
    Res = np.zeros(len(Src), dtype=Src.dtype)
    to_array1d(Src, &src)
    to_array1d(Res, &res)
    source_file = get_opencl_source_file('test1d')
    opencl_test1d(&src, &res, <char*>source_file, device_type, groups)
    return Res


def _opencl_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                           np.ndarray[FLOAT, ndim=3] Kernel not None,
                           FLOAT bias,
                           unsigned int dil_x,
                           unsigned int dil_y,
                           opencl_device_type device_type,
                           unsigned int groups_x,
                           unsigned int groups_y):
    cdef array3d src
    cdef array3d kernel
    cdef array2d res
    Res = np.zeros([Src.shape[0], Src.shape[1]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Kernel, &kernel)
    to_array2d(Res, &res)
    source_file = get_opencl_source_file('convolve_image')
    opencl_convolve_image(&src, &kernel, bias, dil_x, dil_y, &res, <char*>source_file, device_type, groups_x, groups_y) 
    return Res


def _opencl_multi_convolve_image(np.ndarray[FLOAT, ndim=3] Src not None,
                                 np.ndarray[FLOAT, ndim=4] Kernels not None,
                                 np.ndarray[FLOAT, ndim=1] Biases not None,
                                 unsigned int dil_x,
                                 unsigned int dil_y,
                                 opencl_device_type device_type,
                                 unsigned int groups_x,
                                 unsigned int groups_y):
    cdef array3d src
    cdef array4d kernels
    cdef array1d biases
    cdef array3d res
    if Kernels.shape[3] != Biases.shape[0]:
        raise ValueError('Inconsistent kernel and bias arrays')
    Res = np.zeros([Src.shape[0], Src.shape[1], Kernels.shape[3]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array4d(Kernels, &kernels)
    to_array1d(Biases, &biases)
    to_array3d(Res, &res)
    source_file = get_opencl_source_file('convolve_image')
    opencl_multi_convolve_image(&src, &kernels, &biases, dil_x, dil_y, &res,
                                <char*>source_file, device_type, groups_x, groups_y) 
    return Res


def _opencl_relu_max_pool_image(np.ndarray[FLOAT, ndim=3] Src not None,
                                unsigned int size_x,
                                unsigned int size_y,
                                unsigned int dil_x,
                                unsigned int dil_y,
                                opencl_device_type device_type,
                                unsigned int groups_x,
                                unsigned int groups_y,
                                unsigned int groups_z):

    cdef array3d src
    cdef array3d res
    Res = np.zeros([Src.shape[0], Src.shape[1], Src.shape[2]], dtype=Src.dtype)
    to_array3d(Src, &src)
    to_array3d(Res, &res)
    source_file = get_opencl_source_file('relu_max_pool_image')
    opencl_relu_max_pool_image(&src, size_x, size_y, dil_x, dil_y, &res,
                               <char*>source_file, device_type, groups_x, groups_y, groups_z)
    return Res
