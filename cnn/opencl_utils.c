#include "opencl_utils.h"

#define MAX_SOURCE_SIZE (0x100000)


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
  
  // Load the CL kernel source code into the array source_code
  FILE *fp;
  char *source_code;
  size_t source_size;
  
  fp = fopen(source_file, "r");
  if (!fp) {
    free(thisone);
    fprintf(stderr, "WARNING: failed to load kernel.\n");
    exit(1);
  }
  source_code = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_code, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  if (ret != 0)
    fprintf(stderr, "WARNING: fail to get platform IDs\n");
  
  thisone->device_id = NULL;   
  ret = clGetDeviceIDs(platform_id, match_device_type(device_type), 1, &(thisone->device_id), &ret_num_devices);
  if (ret != 0)
    fprintf(stderr, "WARNING: fail to get device IDs\n");

  thisone->context = clCreateContext(NULL, 1, &(thisone->device_id), NULL, NULL, &ret);
  if (ret != 0) 
    fprintf(stderr, "WARNING: fail to create context\n");

  cl_program program = clCreateProgramWithSource(thisone->context, 1, (const char **)&source_code, (const size_t *)&source_size, &ret);
  if (ret != 0)
    fprintf(stderr, "WARNING: fail to create program\n");
 
  ret = clBuildProgram(program, 1, &(thisone->device_id), NULL, NULL, NULL);
  if (ret != 0)
    fprintf(stderr, "WARNING: fail to build program\n");

  thisone->kernel = clCreateKernel(program, kernel_name, &ret);
  if (ret != 0) 
    fprintf(stderr, "WARNING: fail to create kernel\n");

  // Free memory
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


void opencl_test1d(array1d* src,
		   array1d* res,
		   char* source_file,
		   opencl_device_type device_type,
		   unsigned int groups)
{
  // Create OpenCL environment
  opencl_env* env = opencl_env_new(source_file, "test1d", device_type);
  
  // Create a command queue
  cl_int ret;
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  
  // Create memory buffers on the device for each vector 
  const int byte_size = sizeof(FLOAT) * src->dim;
  cl_mem src_data_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
  cl_mem res_data_dev = clCreateBuffer(env->context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  
  // Copy the input vectors to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, src_data_dev, CL_TRUE, 0, byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_data_dev, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
    
  // Set the arguments of the kernel
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_dev);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&res_data_dev);
  FLOAT c = 3.0;
  ret = clSetKernelArg(env->kernel, 2, sizeof(FLOAT), (void*)&c);
  
  // Execute the OpenCL kernel on the list
  size_t global_item_size = src->dim; // Process the entire lists
  size_t local_item_size = groups; // Divide work items into groups
  ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  
  // Get the result back to host
  ret = clEnqueueReadBuffer(command_queue, res_data_dev, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_dev);
  ret = clReleaseMemObject(res_data_dev);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}


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
{
  // Create OpenCL environment
  opencl_env* env = opencl_env_new(source_file, "convolve_image", device_type);

  // Create memory buffers on the device
  cl_int ret;
  cl_int src_byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_int kernel_byte_size = sizeof(FLOAT) * kernel->dimx * kernel->dimy * kernel->dimz;
  cl_int res_byte_size = sizeof(FLOAT) * res->dimx * res->dimy;
  cl_mem src_data_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, src_byte_size, NULL, &ret);
  cl_mem src_dim_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_data_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, kernel_byte_size, NULL, &ret);
  cl_mem kernel_dim_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res_data_dev = clCreateBuffer(env->context, CL_MEM_READ_WRITE, res_byte_size, NULL, &ret);
  cl_mem res_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
    
  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);

  // Copy the input vectors to their respective memory buffers
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int kernel_dim[3] = {kernel->dimx, kernel->dimy, kernel->dimz};
  unsigned int kernel_off[3] = {kernel->offx, kernel->offy, kernel->offz};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res_off[2] = {res->offx, res->offy};
  ret = clEnqueueWriteBuffer(command_queue, src_data_dev, CL_TRUE, 0, src_byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_data_dev, CL_TRUE, 0, kernel_byte_size, kernel->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_dim_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_off_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_dev, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_data_dev, CL_TRUE, 0, res_byte_size, res->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_off_dev, CL_TRUE, 0, 2 * sizeof(unsigned int), res_off, 0, NULL, NULL);
    
  // Set the arguments of the kernel
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_dev);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&src_dim_dev);
  ret = clSetKernelArg(env->kernel, 2, sizeof(cl_mem), (void*)&src_off_dev);
  ret = clSetKernelArg(env->kernel, 3, sizeof(cl_mem), (void*)&kernel_data_dev);
  ret = clSetKernelArg(env->kernel, 4, sizeof(cl_mem), (void*)&kernel_dim_dev);
  ret = clSetKernelArg(env->kernel, 5, sizeof(cl_mem), (void*)&kernel_off_dev);
  ret = clSetKernelArg(env->kernel, 6, sizeof(cl_mem), (void*)&dil_dev);
  ret = clSetKernelArg(env->kernel, 7, sizeof(FLOAT), (void*)&bias);
  ret = clSetKernelArg(env->kernel, 8, sizeof(cl_mem), (void*)&res_data_dev);
  ret = clSetKernelArg(env->kernel, 9, sizeof(cl_mem), (void*)&res_off_dev);
  
  // Execute the OpenCL kernel on the list
  size_t global_item_size[2] = {src->dimx, src->dimy}; 
  size_t local_item_size[2] = {groups_x, groups_y}; 
  ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 2, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
  
  // Get the result back to host
  ret = clEnqueueReadBuffer(command_queue, res_data_dev, CL_TRUE, 0, res_byte_size, res->data, 0, NULL, NULL);
  
  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_dev);
  ret = clReleaseMemObject(kernel_data_dev);
  ret = clReleaseMemObject(res_data_dev);
  ret = clReleaseCommandQueue(command_queue);

  opencl_env_delete(env);
}



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
{  
  array3d kernel;
  array2d res2d;
  unsigned int t;
  FLOAT* bias;

  opencl_env* env = opencl_env_new(source_file, "convolve_image", device_type);
  FLOAT* kernel_data = (FLOAT*)malloc(kernels->dimx * kernels->dimy * kernels->dimz * sizeof(FLOAT));
  FLOAT* res2d_data = (FLOAT*)malloc(res->dimx * res->dimy * sizeof(FLOAT));

  // Create memory buffers on the device
  cl_int ret;
  cl_int src_byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_int kernel_byte_size = sizeof(FLOAT) * kernels->dimx * kernels->dimy * kernels->dimz;
  cl_int res2d_byte_size = sizeof(FLOAT) * res->dimx * res->dimy;
  cl_mem src_data_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, src_byte_size, NULL, &ret);
  cl_mem src_dim_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_data_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, kernel_byte_size, NULL, &ret);
  cl_mem kernel_dim_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res2d_data_dev = clCreateBuffer(env->context, CL_MEM_READ_WRITE, res2d_byte_size, NULL, &ret);
  cl_mem res2d_off_dev = clCreateBuffer(env->context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  
  // Copy the input vectors to their respective memory buffers
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int kernel_dim[3] = {kernels->dimx, kernels->dimy, kernels->dimz};
  unsigned int kernel_off[3] = {kernels->dimy * kernels->dimz, kernels->dimz, 1};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res2d_off[2] = {res->dimy, 1};
  size_t global_item_size[2] = {src->dimx, src->dimy}; 
  size_t local_item_size[2] = {groups_x, groups_y}; 
  
  // Create a command queue and copy arrays from host to device 
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  ret = clEnqueueWriteBuffer(command_queue, src_data_dev, CL_TRUE, 0, src_byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_dim_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_off_dev, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_dev, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res2d_off_dev, CL_TRUE, 0, 2 * sizeof(unsigned int), res2d_off, 0, NULL, NULL);
    
  // Set the arguments of the kernel
  ret = clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void*)&src_data_dev);
  ret = clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void*)&src_dim_dev);
  ret = clSetKernelArg(env->kernel, 2, sizeof(cl_mem), (void*)&src_off_dev);
  ret = clSetKernelArg(env->kernel, 3, sizeof(cl_mem), (void*)&kernel_data_dev);
  ret = clSetKernelArg(env->kernel, 4, sizeof(cl_mem), (void*)&kernel_dim_dev);
  ret = clSetKernelArg(env->kernel, 5, sizeof(cl_mem), (void*)&kernel_off_dev);
  ret = clSetKernelArg(env->kernel, 6, sizeof(cl_mem), (void*)&dil_dev);
  ret = clSetKernelArg(env->kernel, 7, sizeof(FLOAT), (void*)&bias);
  ret = clSetKernelArg(env->kernel, 8, sizeof(cl_mem), (void*)&res2d_data_dev);
  ret = clSetKernelArg(env->kernel, 9, sizeof(cl_mem), (void*)&res2d_off_dev);

  // Loop over sequence of kernels and output array
  bias = biases->data;
  for(t=0; t<kernels->dimt; t++) {
    kernel = slice3d(kernels, t, kernel_data, 0);
    ret = clEnqueueWriteBuffer(command_queue, kernel_data_dev, CL_TRUE, 0, kernel_byte_size, kernel_data, 0, NULL, NULL);
    res2d = slice2d(res, t, res2d_data, 0);
    ret = clEnqueueWriteBuffer(command_queue, res2d_data_dev, CL_TRUE, 0, res2d_byte_size, res2d_data, 0, NULL, NULL);    
    ret = clEnqueueNDRangeKernel(command_queue, env->kernel, 2, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, res2d_data_dev, CL_TRUE, 0, res2d_byte_size, res2d_data, 0, NULL, NULL);
    res2d = slice2d(res, t, res2d_data, 1);
    bias += biases->off;
  }
  
  // Clean up
  free(kernel_data);
  free(res2d_data);
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseMemObject(src_data_dev);
  ret = clReleaseMemObject(kernel_data_dev);
  ret = clReleaseMemObject(res2d_data_dev);
  ret = clReleaseCommandQueue(command_queue);
  opencl_env_delete(env);
}

