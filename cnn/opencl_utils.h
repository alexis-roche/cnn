#ifndef OPENCL_UTILS
#define OPENCL_UTILS

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

  typedef enum {
    OPENCL_DEVICE_TYPE_DEFAULT = 0,
    OPENCL_DEVICE_TYPE_CPU = 1,
    OPENCL_DEVICE_TYPE_GPU = 2,
    OPENCL_DEVICE_TYPE_ACCELERATOR = 3,
    OPENCL_DEVICE_TYPE_ALL = 4
  } opencl_device_type;

  typedef struct {
    unsigned int max_work_group_size;
    unsigned int max_work_item_dimensions;
    unsigned int* max_work_item_sizes;
  } opencl_device_info;
  
  typedef struct {
    cl_kernel kernel;
    cl_context context;
    cl_device_id device_id;
  } opencl_env;

  extern opencl_device_info* opencl_device_info_new(opencl_device_type device_type);

  extern void opencl_device_info_delete(opencl_device_info* thisone);
  
  extern opencl_env* opencl_env_new(char* source_file, char* kernel_name, opencl_device_type device_type);
  
  extern void opencl_env_delete(opencl_env* thisone);
  
  extern void opencl_test1d(array1d* src,
			    array1d* res,
			    char* source_file,
			    opencl_device_type device_type,
			    unsigned int batch_size);
  
  extern void opencl_convolve_image(array3d* src,
				    array3d* kernel,
				    FLOAT bias,
				    unsigned int dil_x,
				    unsigned int dil_y,
				    array2d* res,
				    char* source_file,
				    opencl_device_type device_type,
				    unsigned int groups_x,
				    unsigned int groups_y);
  
  extern void opencl_multi_convolve_image(array3d* src,
					  array4d* kernels,
					  array1d* biases,
					  unsigned int dil_x,
					  unsigned int dil_y,
					  array3d* res,
					  char* source_file,
					  opencl_device_type device_type,
					  unsigned int groups_x,
					  unsigned int groups_y);

  extern void opencl_relu_max_pool_image(array3d* src,
					 unsigned int size_x,
					 unsigned int size_y,
					 unsigned int dil_x,
					 unsigned int dil_y,
					 array3d* res,
					 char* source_file,
					 opencl_device_type device_type,
					 unsigned int groups_x,
					 unsigned int groups_y,
					 unsigned int groups_z);
  
  
#ifdef __cplusplus
}
#endif

#endif

