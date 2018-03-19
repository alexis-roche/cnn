#include "utils.h"

#include <math.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

#define OPENCL_DEVICE CL_DEVICE_TYPE_GPU
//#define OPENCL_DEVICE CL_DEVICE_TYPE_DEFAULT



array3d slice3d(array4d* a4d, unsigned int t, FLOAT* data, unsigned char from_buffer)
{
  array3d a3d;
  FLOAT *buf_a3d, *buf_a4d;
  unsigned int x, y, z;
  size_t pos_x, pos_xy;
  
  a3d.dimx = a4d->dimx;
  a3d.dimy = a4d->dimy;
  a3d.dimz = a4d->dimz;
  a3d.offx = a3d.dimy * a3d.dimz;
  a3d.offy = a3d.dimz;
  a3d.offz = 1; 
  a3d.data = data;
  
  buf_a3d = data;
  pos_x = t * a4d->offt;
  for(x=0; x<a3d.dimx; x++) {
    pos_xy = pos_x;
    for(y=0; y<a3d.dimy; y++) {
      buf_a4d = a4d->data + pos_xy;
      for(z=0; z<a3d.dimz; z++) {
	if (from_buffer)
	  *buf_a4d = *buf_a3d;
	else
	  *buf_a3d = *buf_a4d;
	buf_a3d ++;
	buf_a4d += a4d->offz;
      }
      pos_xy += a4d->offy;
    }
    pos_x += a4d->offx;
  }
  return a3d;
}

array2d slice2d(array3d* a3d, unsigned int z, FLOAT* data, unsigned char from_buffer)
{
  array2d a2d;
  FLOAT *buf_a2d, *buf_a3d;
  unsigned int x, y;
  size_t pos_x;
  
  a2d.dimx = a3d->dimx;
  a2d.dimy = a3d->dimy;
  a2d.offx = a2d.dimy;
  a2d.offy = 1;
  a2d.data = data;
  
  buf_a2d = data;
  pos_x = z * a3d->offz;
  for(x=0; x<a2d.dimx; x++) {
      buf_a3d = a3d->data + pos_x;
      for(y=0; y<a2d.dimy; y++) {
	if (from_buffer)
	  *buf_a3d = *buf_a2d;
	else
	  *buf_a2d = *buf_a3d;
	buf_a2d ++;
	buf_a3d += a3d->offy;
      }
      pos_x += a3d->offx;
  }
  return a2d;
}


array1d slice1d(array2d* a2d, unsigned int y, FLOAT* data, unsigned char from_buffer)
{
  array1d a1d;
  FLOAT *buf_a1d, *buf_a2d;
  unsigned int x;
  
  a1d.dim = a2d->dimx;
  a1d.off = 1;
  a1d.data = data;
  
  buf_a1d = data;
  buf_a2d = a2d->data + y * a2d->offy;
  for(x=0; x<a1d.dim; x++) {
    if (from_buffer)
      *buf_a2d = *buf_a1d;
    else
      *buf_a1d = *buf_a2d;
    buf_a1d ++;
    buf_a2d += a2d->offx;
  }
  return a1d;
}


static inline unsigned int half_dimension(unsigned int dim)
{
  return (dim - 1) / 2;
}


static inline unsigned int unsigned_ceil(FLOAT a)
{
  unsigned int n = (unsigned int) a;
  if (n == a)
    return n;
  else
    return n + 1;
}


static FLOAT _convolve_image(unsigned int xc,
			     unsigned int yc,
			     array3d* src,
			     array3d* kernel,
			     unsigned int dil_x,
			     unsigned int dil_y,
			     FLOAT bias)
{
  FLOAT out = bias;
  unsigned int x, y, z;
  size_t pos_x_kernel = 0, pos_y_kernel = 0, pos_xy_kernel, pos_x_src, pos_y_src, pos_xy_src;
  FLOAT *buf_kernel, *buf_src;
  size_t inc_src_x = dil_x * src->offx;
  size_t inc_src_y = dil_y * src->offy;
  int alpha, beta;
  
  /* Return zero if kernel and source do not fully overlap */
  alpha = xc - dil_x * half_dimension(kernel->dimx);
  beta = src->dimx - alpha;
  if ((alpha < 0) || (beta < (dil_x * kernel->dimx)))
    return 0;
  pos_x_src = alpha * src->offx;
  
  alpha = yc - dil_y * half_dimension(kernel->dimy);
  beta = src->dimy - alpha;
  if ((alpha < 0) || (beta < (dil_y * kernel->dimy)))
    return 0;
  pos_y_src = alpha * src->offy;

  
  /* Joint 3D loop over kernel and source */
  for (x=0; x<kernel->dimx; x++) {

    pos_xy_kernel = pos_x_kernel + pos_y_kernel;
    pos_xy_src = pos_x_src + pos_y_src;
    
    for (y=0; y<kernel->dimy; y++) {

      buf_kernel = kernel->data + pos_xy_kernel;
      buf_src = src->data + pos_xy_src;
      
      for (z=0; z<kernel->dimz; z++) {
	out += (*buf_kernel) * (*buf_src);
	buf_kernel += kernel->offz;
	buf_src += src->offz;
      }
      pos_xy_kernel += kernel->offy;
      pos_xy_src += inc_src_y;
      
    }
    pos_x_kernel += kernel->offx;
    pos_x_src += inc_src_x;
  }

  return out; 
}

/*
  Assume:
  
  kernel and src have same dimension z
  src and res have same x and y dimensions
  result in res, needs be pre-allocated

 */
void convolve_image(array3d* src,
		    array3d* kernel,
		    FLOAT bias,
		    unsigned int dil_x,
		    unsigned int dil_y,
		    array2d* res)
{
  unsigned int x, y;
  size_t pos_x;
  FLOAT *buf_res;
  
  pos_x = 0;
  for (x=0; x<res->dimx; x++) {
    buf_res = res->data + pos_x;
    for (y=0; y<res->dimy; y++) {
      *buf_res = _convolve_image(x, y, src, kernel, dil_x, dil_y, bias);
      buf_res += res->offy;
    }
    pos_x += res->offx; 
  }
}


void multi_convolve_image(array3d* src,
			  array4d* kernels,
			  array1d* biases,
			  unsigned int dil_x,
			  unsigned int dil_y,
			  array3d* res)

{
  array3d kernel;
  array2d res2d;
  unsigned int t;
  FLOAT *bias;
  
  kernel.dimx = kernels->dimx;
  kernel.dimy = kernels->dimy;
  kernel.dimz = kernels->dimz;
  kernel.offx = kernels->offx;
  kernel.offy = kernels->offy;
  kernel.offz = kernels->offz;
  kernel.data = kernels->data;

  bias = biases->data;
  
  res2d.dimx = res->dimx;
  res2d.dimy = res->dimy;
  res2d.offx = res->offx;
  res2d.offy = res->offy;
  res2d.data = res->data;

  for(t=0; t<kernels->dimt; t++) {
    convolve_image(src, &kernel, *bias, dil_x, dil_y, &res2d);
    kernel.data += kernels->offt;
    bias += biases->off;
    res2d.data += res->offz;
  }
}




static FLOAT _relu_max_pool_image(unsigned int xc,
				  unsigned int yc,
				  size_t pos_zc,
				  array3d* src,
				  unsigned int size_x,
				  unsigned int size_y,
				  unsigned int dil_x,
				  unsigned int dil_y)
{
  FLOAT out = 0, tmp;
  unsigned int x, y, z;
  size_t pos_x, pos_y, pos_xy;
  FLOAT *buf;
  size_t inc_x = dil_x * src->offx;
  size_t inc_y = dil_y * src->offy;
  int alpha, beta;
  
  /* Return zero if kernel and source do not fully overlap */
  alpha = xc - dil_x * half_dimension(size_x);
  beta = src->dimx - alpha;
  if ((alpha < 0) || (beta < (dil_x * size_x)))
    return out;
  pos_x = alpha * src->offx;
  
  alpha = yc - dil_y * half_dimension(size_y);
  beta = src->dimy - alpha;
  if ((alpha < 0) || (beta < (dil_y * size_y)))
    return out;
  pos_y = alpha * src->offy;
  
  /* 2D loop over source slice to find max value (if positive), and
     subsitute output if applicable */
  pos_x += pos_zc;
  for (x=0; x<size_x; x++) {
    pos_xy = pos_x + pos_y;
    buf = src->data + pos_xy;
    for (y=0; y<size_y; y++) {
      tmp = *buf;
      if (tmp > out)
	out = tmp;
      pos_xy += inc_y;
      buf += inc_y;
    }
    pos_x += inc_x;
  }

  return out; 
}



void relu_max_pool_image(array3d* src,
			 unsigned int size_x,
			 unsigned int size_y,
			 unsigned int dil_x,
			 unsigned int dil_y,
			 array3d* res)
{
  unsigned int x, y, z;
  size_t pos_res_z, pos_res_xz, pos_src_z;
  FLOAT *buf_res;
  
  pos_res_z = 0;
  pos_src_z = 0;
  
  for (z=0; z<src->dimz; z++) {

    pos_res_xz = pos_res_z;

    for (x=0; x<src->dimx; x++) {
      buf_res = res->data + pos_res_xz;

      for (y=0; y<src->dimy; y++) {
	*buf_res = _relu_max_pool_image(x, y, pos_src_z, src, size_x, size_y, dil_x, dil_y);
	buf_res += res->offy; 
      }
      pos_res_xz += res->offx; 
    }

    pos_res_z += res->offz;
    pos_src_z += src->offz;
  }
  
}



typedef struct {
  cl_kernel kernel;
  cl_context context;
  cl_device_id device_id;
} opencl_env;



static opencl_env* opencl_env_new(char* fname, char* kname)
{

  opencl_env* thisone = (opencl_env*)malloc(sizeof(opencl_env));
  
  // Load the CL kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;
  fp = fopen(fname, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

  fprintf(stderr, "clGetPlatforms: ret=%d\n", ret);

  thisone->device_id = NULL;   
  ret = clGetDeviceIDs(platform_id, OPENCL_DEVICE, 1, &(thisone->device_id), &ret_num_devices);
  if (ret != 0)
    fprintf(stderr, "COULD NOT REACH GPU\n");

  fprintf(stderr, "clGetDeviceIDs: ret=%d\n", ret);
  
  // Load the kernel source code into the array source_str
  thisone->context = clCreateContext(NULL, 1, &(thisone->device_id), NULL, NULL, &ret);

  fprintf(stderr, "clCreateContext: ret=%d\n", ret);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(thisone->context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &(thisone->device_id), NULL, NULL, NULL);
  
  // Create the OpenCL kernel
  thisone->kernel = clCreateKernel(program, kname, &ret);

  free(source_str);
  ret = clReleaseProgram(program);
  
  return thisone;
}


static void opencl_env_delete(opencl_env* thisone)
{
  cl_int ret;
  ret = clReleaseKernel(thisone->kernel);
  ret = clReleaseContext(thisone->context);
  free(thisone);  
  return;
}


void gpu_basic_test1d(array1d* src, array1d* res, char* fname, unsigned int groups)
{

  // Create OpenCL environment
  opencl_env* env = opencl_env_new(fname, "basic_test1d");
  
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

  unsigned int max_work_items = CL_DEVICE_MAX_WORK_GROUP_SIZE;  
  fprintf(stderr, "Max number of work items = %d\n", max_work_items);
  
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



static void _gpu_convolve_image(array3d* src,
				array3d* kernel,
				FLOAT bias,
				unsigned int dil_x,
				unsigned int dil_y,
				array2d* res,
				opencl_env* env,
				unsigned int groups_x,
				unsigned int groups_y)
{  

  // Create a command queue
  cl_int ret;
  cl_command_queue command_queue = clCreateCommandQueue(env->context, env->device_id, 0, &ret);
  
  // Create memory buffers on the device
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
}


void gpu_convolve_image(array3d* src,
			array3d* kernel,
			FLOAT bias,
			unsigned int dil_x,
			unsigned int dil_y,
			array2d* res,
			char* fname,
			unsigned int groups_x,
			unsigned int groups_y)
{
  opencl_env* env; 

  env = opencl_env_new(fname, "convolve_image");
  _gpu_convolve_image(src, kernel, bias, dil_x, dil_y, res, env, groups_x, groups_y);
  opencl_env_delete(env);
}


void gpu_multi_convolve_image(array3d* src,
			      array4d* kernels,
			      array1d* biases,
			      unsigned int dil_x,
			      unsigned int dil_y,
			      array3d* res,
			      char* fname,
			      unsigned int groups_x,
			      unsigned int groups_y)
{
  array3d kernel;
  array2d res2d;
  unsigned int t;
  FLOAT *kernel_data, *res2d_data, *bias;
  opencl_env* env;

  env = opencl_env_new(fname, "convolve_image");
  kernel_data = (FLOAT*)malloc(kernels->dimx * kernels->dimy * kernels->dimz * sizeof(FLOAT));
  res2d_data = (FLOAT*)malloc(res->dimx * res->dimy * sizeof(FLOAT));
  
  bias = biases->data;
  for(t=0; t<kernels->dimt; t++) {
    kernel = slice3d(kernels, t, kernel_data, 0);
    res2d = slice2d(res, t, res2d_data, 0);
    _gpu_convolve_image(src, &kernel, *bias, dil_x, dil_y, &res2d, env, groups_x, groups_y);
    res2d = slice2d(res, t, res2d_data, 1);
    bias += biases->off;
  }

  free(kernel_data);
  free(res2d_data);
  opencl_env_delete(env);
}
