#include "utils.h"

#include <math.h>
#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)


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

static inline void adjust_kernel(unsigned int c,
				 unsigned int dil,
				 size_t kernel_dim,
				 size_t kernel_offset,
				 size_t src_dim,
				 size_t src_offset,
				 unsigned int* c0,
				 unsigned int* c1,
				 size_t* pos_kernel,
				 size_t* pos_src)
{
  int alpha, beta;
  
  alpha = dil * half_dimension(kernel_dim) - c;
  beta = alpha + src_dim;
    
  if (alpha > 0) {
    *c0 = unsigned_ceil(alpha / (FLOAT)dil);
    if (pos_kernel != NULL)
      *pos_kernel = (*c0) * kernel_offset;
  }

  *pos_src = (dil * (*c0) - alpha) * src_offset;

  if ( (dil * kernel_dim) > beta)
    *c1 = unsigned_ceil(beta / (FLOAT)dil);
 
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
  unsigned int x0 = 0, y0 = 0, x1 = kernel->dimx, y1 = kernel->dimy;
  size_t pos_x_kernel = 0, pos_y_kernel = 0, pos_xy_kernel, pos_x_src, pos_y_src, pos_xy_src;
  FLOAT *buf_kernel, *buf_src;
  size_t inc_src_x = dil_x * src->offx;
  size_t inc_src_y = dil_y * src->offy;
  
  /* Adjust x- and y-coordinate ranges for partial overlap between
     kernel and source */
  adjust_kernel(xc, dil_x, kernel->dimx,  kernel->offx,
		src->dimx, src->offx, 
		&x0, &x1, &pos_x_kernel, &pos_x_src);
  
  adjust_kernel(yc, dil_y, kernel->dimy, kernel->offy,
		src->dimy, src->offy, 
		&y0, &y1, &pos_y_kernel, &pos_y_src);

  /* Joint 3D loop over kernel and source */
  
  for (x=x0; x<x1; x++) {

    pos_xy_kernel = pos_x_kernel + pos_y_kernel;
    pos_xy_src = pos_x_src + pos_y_src;
    
    for (y=y0; y<y1; y++) {

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
  unsigned int x0 = 0, y0 = 0, x1 = size_x, y1 = size_y;
  size_t pos_x = 0, pos_y = 0, pos_xy;
  FLOAT *buf;
  size_t inc_x = dil_x * src->offx;
  size_t inc_y = dil_y * src->offy;
  
  /* Adjust x- and y-coordinate ranges for partial overlap between
     kernel and source */
  adjust_kernel(xc, dil_x, size_x,  0,
		src->dimx, src->offx, 
		&x0, &x1, NULL, &pos_x);

  adjust_kernel(yc, dil_y, size_y, 0, 
		src->dimy, src->offy, 
		&y0, &y1, NULL, &pos_y);
  
  /* 2D loop over source slice to find max value (if positive), and
     subsitute output if applicable */
  pos_x += pos_zc;
  for (x=x0; x<x1; x++) {
    pos_xy = pos_x + pos_y;
    buf = src->data + pos_xy;
    for (y=y0; y<y1; y++) {
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



void irina(array1d* src,
	   array1d* res,
	   char* fname) {

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
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  
  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  
  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  
  // Create memory buffers on the device for each vector 
  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, byte_size, NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  
  // Copy the input vectors to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, byte_size, src->data, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
  
  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
  
  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  
  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
  
  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  static const float zob = 3.0;
  ret = clSetKernelArg(kernel, 2, sizeof(float), &zob);
  
  // Execute the OpenCL kernel on the list
  size_t global_item_size = src->dim; // Process the entire lists
  size_t local_item_size = 10; // Divide work items into groups of 10
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  
  // Get the result back to host
  ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0, byte_size, res->data, 0, NULL, NULL);
  
  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
}
