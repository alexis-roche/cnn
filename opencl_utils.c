#include "opencl_utils.h"

#define MAX_SOURCE_SIZE (0x100000)


#define CL_ERROR_CHECK(ret, msg)					\
  {									\
    if (ret != CL_SUCCESS)						\
      fprintf(stderr, "OpenCL error: %s (errcode %i)\n", msg, ret);	\
  }									\
    


static cl_device_type match_device_type(opencl_device_type device_type)
{
  cl_device_type out;

  switch(device_type) {

  case OPENCL_DEVICE_TYPE_CPU:
    out = CL_DEVICE_TYPE_CPU;
    break;

  case OPENCL_DEVICE_TYPE_GPU:
    out = CL_DEVICE_TYPE_GPU;
    break;

  case OPENCL_DEVICE_TYPE_ACCELERATOR:
    out = CL_DEVICE_TYPE_ACCELERATOR;
    break;

  case OPENCL_DEVICE_TYPE_ALL:
    out = CL_DEVICE_TYPE_ALL;
    break;

  case OPENCL_DEVICE_TYPE_DEFAULT:
  default:
    out = CL_DEVICE_TYPE_DEFAULT;
    break;

  }

  return out;  
}

opencl_device_info* opencl_device_info_new(opencl_device_type device_type)
{
  opencl_device_info* thisone = malloc(sizeof(opencl_device_info));
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
  size_t aux1;
  cl_uint aux2;

  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, match_device_type(device_type), 1, &device_id, &ret_num_devices);  

  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &aux1, NULL);
  thisone->max_work_group_size = (unsigned int)aux1;

  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &aux2, NULL);
  thisone->max_work_item_dimensions = (unsigned int)aux2;

  size_t* max_work_item_sizes = malloc((thisone->max_work_item_dimensions) * sizeof(size_t));
  thisone->max_work_item_sizes = malloc((thisone->max_work_item_dimensions) * sizeof(unsigned int));
  ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, (thisone->max_work_item_dimensions) * sizeof(size_t), max_work_item_sizes, NULL);
  unsigned int i;
  for (i=0; i<(thisone->max_work_item_dimensions); i ++)
    thisone->max_work_item_sizes[i] = max_work_item_sizes[i];

  free(max_work_item_sizes);
  return thisone;
}

void opencl_device_info_delete(opencl_device_info* thisone)
{
  free(thisone->max_work_item_sizes);
  free(thisone);
  return;
}


opencl_env* opencl_env_new(char* source_file, char* kernel_name, opencl_device_type device_type)
{
  opencl_env* thisone = (opencl_env*)malloc(sizeof(opencl_env));
  
  /* Load the CL kernel source code into the array source_code */
  FILE *fp;
  char *source_code;
  size_t source_size;
  
  fp = fopen(source_file, "r");
  if (!fp) {
    free(thisone);
    fprintf(stderr, "Error: failed to load kernel.\n");
    exit(1);
  }
  source_code = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_code, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  /* Get platform and device information */
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  CL_ERROR_CHECK(ret, "fail to get platform IDs");

  thisone->device_id = NULL;   
  ret = clGetDeviceIDs(platform_id, match_device_type(device_type), 1, &(thisone->device_id), &ret_num_devices);
  CL_ERROR_CHECK(ret, "fail to get device IDs");

  thisone->context = clCreateContext(NULL, 1, &(thisone->device_id), NULL, NULL, &ret);
  CL_ERROR_CHECK(ret, "fail to create context");

  cl_program program = clCreateProgramWithSource(thisone->context, 1, (const char**)&source_code, (const size_t*)&source_size, &ret);
  CL_ERROR_CHECK(ret, "fail to create program");
 
  ret = clBuildProgram(program, 1, &(thisone->device_id), NULL, NULL, NULL);
  CL_ERROR_CHECK(ret, "fail to build program");

  thisone->kernel = clCreateKernel(program, kernel_name, &ret);
  CL_ERROR_CHECK(ret, "fail to create kernel");

  /* Free memory */
  free(source_code);
  ret = clReleaseProgram(program);

  return thisone;
}


void opencl_env_delete(opencl_env* thisone)
{
  cl_int ret;
  ret = clReleaseKernel(thisone->kernel);
  ret = clReleaseContext(thisone->context);
  free(thisone);  
  return;
}


void opencl_test1d(array1d* src_,
		   array1d* res_,
		   char* source_file,
		   opencl_device_type device_type,
		   unsigned int groups)
{
  /* Create OpenCL environment */
  opencl_env* env = opencl_env_new(source_file, "test1d", device_type);
  
  /* Host variables */
  array1d* src = array1d_new_contiguous_from(src_);
  array1d* res = array1d_new_contiguous_from(res_);

  /* Create a command queue */
  cl_int ret;
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  
  /* Create memory buffers on the device for each vector */
  const int byte_size = sizeof(FLOAT) * src->dim;
  cl_mem src_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
  cl_mem res_data_q = clCreateBuffer(env->context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  
  /* Copy the input vectors to their respective memory buffers */
  ret = clEnqueueWriteBuffer(command_queue, src_data_q, CL_TRUE, 0, byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_data_q, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
    
  /* Set the arguments of the kernel */
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_q);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&res_data_q);
  FLOAT c = 3.0;
  ret = clSetKernelArg(env->kernel, 2, sizeof(FLOAT), (void*)&c);
  
  /* Execute the OpenCL kernel on the list */
  size_t global_item_size = src->dim; 
  size_t local_item_size = groups;
  ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  
  /* Get the result back to host */
  ret = clEnqueueReadBuffer(command_queue, res_data_q, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
  if (res->owner)
    copy1d(res_, res, 1);
  
  /* Clean up */
  array1d_delete(src);
  array1d_delete(res);
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_q);
  ret = clReleaseMemObject(res_data_q);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}

/*
  Assumes src and kernel are contiguous in memory
*/
void opencl_convolve_image(array3d* src_,
			   array3d* kernel_,
			   FLOAT bias,
			   unsigned int dil_x,
			   unsigned int dil_y,
			   array2d* res_,
			   char* source_file,
			   opencl_device_type device_type,
			   unsigned int groups_x,
			   unsigned int groups_y)
{
  /* Create OpenCL environment */
  opencl_env* env = opencl_env_new(source_file, "convolve_image", device_type);

  /* Host variables */
  array3d* src = array3d_new_contiguous_from(src_);
  array3d* kernel = array3d_new_contiguous_from(kernel_);
  array2d* res = array2d_new_contiguous_from(res_);
  
  /* Create memory buffers on the device */
  cl_int ret;
  cl_int src_byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_int kernel_byte_size = sizeof(FLOAT) * kernel->dimx * kernel->dimy * kernel->dimz;
  cl_int res_byte_size = sizeof(FLOAT) * res->dimx * res->dimy;
  cl_mem src_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, src_byte_size, NULL, &ret);
  cl_mem src_dim_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, kernel_byte_size, NULL, &ret);
  cl_mem kernel_dim_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res_data_q = clCreateBuffer(env->context, CL_MEM_READ_WRITE, res_byte_size, NULL, &ret);
  cl_mem res_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
    
  /* Create a command queue */
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);

  /* Copy the input vectors to their respective memory buffers */
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int kernel_dim[3] = {kernel->dimx, kernel->dimy, kernel->dimz};
  unsigned int kernel_off[3] = {kernel->offx, kernel->offy, kernel->offz};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res_off[2] = {res->offx, res->offy};
  ret = clEnqueueWriteBuffer(command_queue, src_data_q, CL_TRUE, 0, src_byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_data_q, CL_TRUE, 0, kernel_byte_size, kernel->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_dim_q, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_q, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_off_q, CL_TRUE, 0, 2 * sizeof(unsigned int), res_off, 0, NULL, NULL);
    
  /* Set the arguments of the kernel */
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_q);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&src_dim_q);
  ret = clSetKernelArg(env->kernel, 2, sizeof(cl_mem), (void*)&src_off_q);
  ret = clSetKernelArg(env->kernel, 3, sizeof(cl_mem), (void*)&kernel_data_q);
  ret = clSetKernelArg(env->kernel, 4, sizeof(cl_mem), (void*)&kernel_dim_q);
  ret = clSetKernelArg(env->kernel, 5, sizeof(cl_mem), (void*)&kernel_off_q);
  ret = clSetKernelArg(env->kernel, 6, sizeof(cl_mem), (void*)&dil_q);
  ret = clSetKernelArg(env->kernel, 7, sizeof(FLOAT), (void*)&bias);
  ret = clSetKernelArg(env->kernel, 8, sizeof(cl_mem), (void*)&res_data_q);
  ret = clSetKernelArg(env->kernel, 9, sizeof(cl_mem), (void*)&res_off_q);
  
  /* Execute the OpenCL kernel on the list */
  size_t global_item_size[2] = {src->dimx, src->dimy}; 
  size_t local_item_size[2] = {groups_x, groups_y}; 
  ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 2, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
  
  /* Get the result back to host */
  ret = clEnqueueReadBuffer(command_queue, res_data_q, CL_TRUE, 0, res_byte_size, res->data, 0, NULL, NULL);
  if (res->owner)
    copy2d(res_, res, 1);
    
  /* Clean up */
  array3d_delete(src);
  array3d_delete(kernel);
  array2d_delete(res);
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_q);
  ret = clReleaseMemObject(src_dim_q);
  ret = clReleaseMemObject(src_off_q);
  ret = clReleaseMemObject(kernel_data_q);
  ret = clReleaseMemObject(kernel_dim_q);
  ret = clReleaseMemObject(kernel_off_q);
  ret = clReleaseMemObject(dil_q);
  ret = clReleaseMemObject(res_data_q);
  ret = clReleaseMemObject(res_off_q);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}



void opencl_multi_convolve_image(array3d* src_,
				 array4d* kernels,
				 array1d* biases,
				 unsigned int dil_x,
				 unsigned int dil_y,
				 array3d* res,
				 char* source_file,
				 opencl_device_type device_type,
				 unsigned int groups_x,
				 unsigned int groups_y)
{  
  /* Create OpenCL environment */
  opencl_env* env = opencl_env_new(source_file, "convolve_image", device_type);

  /* Host variables */
  array3d* src = array3d_new_contiguous_from(src_);
  array3d* kernel = array3d_new(kernels->dimx, kernels->dimy, kernels->dimz);
  array2d* res2d = array2d_new(res->dimx, res->dimy);
  FLOAT* bias = biases->data;
  unsigned int t;
  
  /* Create memory buffers on the device */
  cl_int ret;
  cl_int src_byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_int kernel_byte_size = sizeof(FLOAT) * kernels->dimx * kernels->dimy * kernels->dimz;
  cl_int res2d_byte_size = sizeof(FLOAT) * res->dimx * res->dimy;
  cl_mem src_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, src_byte_size, NULL, &ret);
  cl_mem src_dim_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, kernel_byte_size, NULL, &ret);
  cl_mem kernel_dim_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res2d_data_q = clCreateBuffer(env->context, CL_MEM_READ_WRITE, res2d_byte_size, NULL, &ret);
  cl_mem res2d_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  
  /* Copy the input vectors to their respective memory buffers */
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int kernel_dim[3] = {kernels->dimx, kernels->dimy, kernels->dimz};
  unsigned int kernel_off[3] = {kernel->offx, kernel->offy, kernel->offz};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res2d_off[2] = {res2d->offx, res2d->offy};
  size_t global_item_size[2] = {src->dimx, src->dimy}; 
  size_t local_item_size[2] = {groups_x, groups_y}; 
  
  /* Create a command queue and copy arrays from host to device */
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  ret = clEnqueueWriteBuffer(command_queue, src_data_q, CL_TRUE, 0, src_byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_dim_q, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_q, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res2d_off_q, CL_TRUE, 0, 2 * sizeof(unsigned int), res2d_off, 0, NULL, NULL);
    
  /* Set the arguments of the kernel
     The bias argument (position 7) needs be set in the loop as is not
     a cl_mem and changes depending on the convolution */
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_q);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&src_dim_q);
  ret = clSetKernelArg(env->kernel, 2, sizeof(cl_mem), (void*)&src_off_q);
  ret = clSetKernelArg(env->kernel, 3, sizeof(cl_mem), (void*)&kernel_data_q);
  ret = clSetKernelArg(env->kernel, 4, sizeof(cl_mem), (void*)&kernel_dim_q);
  ret = clSetKernelArg(env->kernel, 5, sizeof(cl_mem), (void*)&kernel_off_q);
  ret = clSetKernelArg(env->kernel, 6, sizeof(cl_mem), (void*)&dil_q);
  ret = clSetKernelArg(env->kernel, 8, sizeof(cl_mem), (void*)&res2d_data_q);
  ret = clSetKernelArg(env->kernel, 9, sizeof(cl_mem), (void*)&res2d_off_q);

  /* Loop over sequence of kernels and output array
     Copy kernel data to contiguous array `kernel`, then to device
     Run the kernel, copy the result back to the `res2d` array,
     then to the `res` array */
  for(t=0; t<kernels->dimt; t++) {
    slice3d(kernels, kernel, t, 0);
    ret = clEnqueueWriteBuffer(command_queue, kernel_data_q, CL_TRUE, 0, kernel_byte_size, kernel->data, 0, NULL, NULL);
    ret = clSetKernelArg(env->kernel, 7, sizeof(FLOAT), (void*)bias);
    ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 2, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, res2d_data_q, CL_TRUE, 0, res2d_byte_size, res2d->data, 0, NULL, NULL);
    slice2d(res, res2d, t, 1);
    bias += biases->off;
  }
  
  /* Clean up */
  array3d_delete(src);
  array3d_delete(kernel);
  array2d_delete(res2d);
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_q);
  ret = clReleaseMemObject(src_dim_q);
  ret = clReleaseMemObject(src_off_q);
  ret = clReleaseMemObject(kernel_data_q);
  ret = clReleaseMemObject(kernel_dim_q);
  ret = clReleaseMemObject(kernel_off_q);
  ret = clReleaseMemObject(dil_q);
  ret = clReleaseMemObject(res2d_data_q);
  ret = clReleaseMemObject(res2d_off_q);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}




void opencl_relu_max_pool_image(array3d* src_,
				unsigned int size_x,
				unsigned int size_y,
				unsigned int dil_x,
				unsigned int dil_y,
				array3d* res_,
				char* source_file,
				opencl_device_type device_type,
				unsigned int groups_x,
				unsigned int groups_y,
				unsigned int groups_z)

{
  /* Create OpenCL environment */
  opencl_env* env = opencl_env_new(source_file, "relu_max_pool_image", device_type);

  /* Host variables */
  array3d* src = array3d_new_contiguous_from(src_);
  array3d* res = array3d_new_contiguous_from(res_);
  
  /* Create memory buffers on the device */
  cl_int ret;
  cl_int byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_mem src_data_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
  cl_mem src_dim_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem size_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res_data_q = clCreateBuffer(env->context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  cl_mem res_off_q = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  
  /* Copy the input vectors to their respective memory buffers */
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int size[2] = {size_x, size_y};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res_off[3] = {res->offx, res->offy, res->offz};
  size_t global_item_size[3] = {src->dimx, src->dimy, src->dimz}; 
  size_t local_item_size[3] = {groups_x, groups_y, groups_z}; 
  
  /* Create a command queue and copy arrays from host to device */
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  ret = clEnqueueWriteBuffer(command_queue, src_data_q, CL_TRUE, 0, byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, size_q, CL_TRUE, 0, 2 * sizeof(unsigned int), size, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_q, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL); 
  ret = clEnqueueWriteBuffer(command_queue, res_off_q, CL_TRUE, 0, 3 * sizeof(unsigned int), res_off, 0, NULL, NULL);
    
  /* Set the arguments of the kernel */
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_q);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&src_dim_q);
  ret = clSetKernelArg(env->kernel, 2, sizeof(cl_mem), (void*)&src_off_q);
  ret = clSetKernelArg(env->kernel, 3, sizeof(cl_mem), (void*)&size_q);
  ret = clSetKernelArg(env->kernel, 4, sizeof(cl_mem), (void*)&dil_q);
  ret = clSetKernelArg(env->kernel, 5, sizeof(cl_mem), (void*)&res_data_q);
  ret = clSetKernelArg(env->kernel, 6, sizeof(cl_mem), (void*)&res_off_q);

  /* Execute the OpenCL kernel on the list */
  ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 3, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
  
  /* Get the result back to host */
  ret = clEnqueueReadBuffer(command_queue, res_data_q, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
  if (res->owner)
    copy3d(res_, res, 1);
  
  /* Clean up */
  array3d_delete(src);
  array3d_delete(res);
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_q);
  ret = clReleaseMemObject(src_dim_q);
  ret = clReleaseMemObject(src_off_q);
  ret = clReleaseMemObject(size_q);
  ret = clReleaseMemObject(dil_q);
  ret = clReleaseMemObject(res_data_q);
  ret = clReleaseMemObject(res_off_q);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}
