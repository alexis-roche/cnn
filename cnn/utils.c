#include "utils.h"

#include <math.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

#if 1
#define OPENCL_DEVICE CL_DEVICE_TYPE_GPU
#else
#define OPENCL_DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


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
			     unsigned int dil_y)
{
  FLOAT out = 0;
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
    return out;
  pos_x_src = alpha * src->offx;
  
  alpha = yc - dil_y * half_dimension(kernel->dimy);
  beta = src->dimy - alpha;
  if ((alpha < 0) || (beta < (dil_y * kernel->dimy)))
    return out;
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
      *buf_res = _convolve_image(x, y, src, kernel, dil_x, dil_y) + bias;
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



void gpu_basic_test1d(array1d* src, array1d* res, char* fname, unsigned int batch_size)
{

  // Create the two input vectors
  const int byte_size = sizeof(FLOAT) * src->dim;
  
  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;
  
  fp = fopen(fname, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);
  
  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, OPENCL_DEVICE, 1, &device_id, &ret_num_devices);
  
  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  
  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  
  // Create memory buffers on the device for each vector 
  cl_mem src_data_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
  cl_mem res_data_cp = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  
  // Copy the input vectors to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, src_data_cp, CL_TRUE, 0, byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_data_cp, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
  
  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
  
  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  
  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "basic_test1d", &ret);
  
  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&src_data_cp);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&res_data_cp);
  FLOAT c = 3.0;
  ret = clSetKernelArg(kernel, 2, sizeof(FLOAT), (void*)&c);
  
  // Execute the OpenCL kernel on the list
  size_t global_item_size = src->dim; // Process the entire lists
  size_t local_item_size = batch_size; // Divide work items into groups
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  
  // Get the result back to host
  ret = clEnqueueReadBuffer(command_queue, res_data_cp, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(src_data_cp);
  ret = clReleaseMemObject(res_data_cp);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
}


void gpu_convolve_image(array3d* src,
			array3d* kernel,
			FLOAT bias,
			unsigned int dil_x,
			unsigned int dil_y,
			array2d* res,
			char* fname,
			unsigned int batch_size)
{

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
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, OPENCL_DEVICE, 1, &device_id, &ret_num_devices);
  
  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  
  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  
  // Create memory buffers on the device
  cl_int src_byte_size = sizeof(FLOAT) * src->dimx * src->dimy * src->dimz;
  cl_int kernel_byte_size = sizeof(FLOAT) * kernel->dimx * kernel->dimy * kernel->dimz;
  cl_int res_byte_size = sizeof(FLOAT) * res->dimx * res->dimy;
  cl_mem src_data_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, src_byte_size, NULL, &ret);
  cl_mem src_dim_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem src_off_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_data_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, kernel_byte_size, NULL, &ret);
  cl_mem kernel_dim_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem kernel_off_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned int), NULL, &ret);
  cl_mem dil_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  cl_mem res_data_cp = clCreateBuffer(context, CL_MEM_READ_WRITE, res_byte_size, NULL, &ret);
  cl_mem res_off_cp = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(unsigned int), NULL, &ret);
  
  // Copy the input vectors to their respective memory buffers
  unsigned int src_dim[3] = {src->dimx, src->dimy, src->dimz};
  unsigned int src_off[3] = {src->offx, src->offy, src->offz};
  unsigned int kernel_dim[3] = {kernel->dimx, kernel->dimy, kernel->dimz};


  printf("Kernel dim = %d, %d, %d\n", kernel_dim[0], kernel_dim[1], kernel_dim[2]);
  
  unsigned int kernel_off[3] = {kernel->offx, kernel->offy, kernel->offz};
  unsigned int dil[2] = {dil_x, dil_y};
  unsigned int res_off[2] = {res->offx, res->offy};
  
  ret = clEnqueueWriteBuffer(command_queue, src_data_cp, CL_TRUE, 0, src_byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_dim_cp, CL_TRUE, 0, 3 * sizeof(unsigned int), src_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, src_off_cp, CL_TRUE, 0, 3 * sizeof(unsigned int), src_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_data_cp, CL_TRUE, 0, kernel_byte_size, kernel->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_dim_cp, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_dim, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, kernel_off_cp, CL_TRUE, 0, 3 * sizeof(unsigned int), kernel_off, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, dil_cp, CL_TRUE, 0, 2 * sizeof(unsigned int), dil, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_data_cp, CL_TRUE, 0, res_byte_size, res->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, res_off_cp, CL_TRUE, 0, 2 * sizeof(unsigned int), res_off, 0, NULL, NULL);
  
  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
  
  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  fprintf(stderr, "Ret = %d\n", ret);

  
  // Create the OpenCL kernel
  cl_kernel k_conv = clCreateKernel(program, "convolve_image", &ret);
  
  // Set the arguments of the kernel
  ret = clSetKernelArg(k_conv, 0, sizeof(cl_mem), (void*)&src_data_cp);
  ret = clSetKernelArg(k_conv, 1, sizeof(cl_mem), (void*)&src_dim_cp);
  ret = clSetKernelArg(k_conv, 2, sizeof(cl_mem), (void*)&src_off_cp);
  ret = clSetKernelArg(k_conv, 3, sizeof(cl_mem), (void*)&kernel_data_cp);
  ret = clSetKernelArg(k_conv, 4, sizeof(cl_mem), (void*)&kernel_dim_cp);
  ret = clSetKernelArg(k_conv, 5, sizeof(cl_mem), (void*)&kernel_off_cp);
  ret = clSetKernelArg(k_conv, 6, sizeof(cl_mem), (void*)&dil_cp);  
  ret = clSetKernelArg(k_conv, 7, sizeof(cl_mem), (void*)&res_data_cp);
  ret = clSetKernelArg(k_conv, 8, sizeof(cl_mem), (void*)&res_off_cp);
  
  // Execute the OpenCL kernel on the list
  size_t global_item_size[2] = {src->dimx, src->dimy}; // Process the entire lists
  size_t local_item_size[2] = {batch_size, batch_size}; // Divide work items into groups 
  ret = clEnqueueNDRangeKernel(command_queue, k_conv, 2, NULL, (const size_t*)&global_item_size, (const size_t*)&local_item_size, 0, NULL, NULL);
  
  // Get the result back to host
  ret = clEnqueueReadBuffer(command_queue, res_data_cp, CL_TRUE, 0, res_byte_size, res->data, 0, NULL, NULL);
  
  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(k_conv);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(src_data_cp);
  ret = clReleaseMemObject(kernel_data_cp);
  ret = clReleaseMemObject(res_data_cp);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
}


