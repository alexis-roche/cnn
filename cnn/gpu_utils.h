#ifndef GPU_UTILS
#define GPU_UTILS

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "utils.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define OPENCL_DEVICE CL_DEVICE_TYPE_GPU
//#define OPENCL_DEVICE CL_DEVICE_TYPE_DEFAULT

  
  typedef struct {
    cl_kernel kernel;
    cl_context context;
    cl_device_id device_id;
  } opencl_env;
  
  extern opencl_env* opencl_env_new(char* fname, char* kname);
  
  extern void opencl_env_delete(opencl_env* thisone);
  
  extern void gpu_basic_test1d(array1d* src,
			       array1d* res,
			       char* fname,
			       unsigned int batch_size);
  
  extern void gpu_convolve_image(array3d* src,
				 array3d* kernel,
				 FLOAT bias,
				 unsigned int dil_x,
				 unsigned int dil_y,
				 array2d* res,
				 char* fname,
				 unsigned int groups_x,
				 unsigned int groups_y);

  extern void gpu_multi_convolve_image(array3d* src,
				       array4d* kernels,
				       array1d* biases,
				       unsigned int dil_x,
				       unsigned int dil_y,
				       array3d* res,
				       char* fname,
				       unsigned int groups_x,
				       unsigned int groups_y);

#ifdef __cplusplus
}
#endif

#endif

