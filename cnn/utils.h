#ifndef UTILS
#define UTILS

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

  typedef float FLOAT;
  
  typedef struct {
    size_t dim;
    size_t off;
    FLOAT* data;
  } array1d;

  typedef struct {
    size_t dimx;
    size_t dimy;
    size_t offx;
    size_t offy;
    FLOAT* data;
  } array2d;

  typedef struct {
    size_t dimx;
    size_t dimy;
    size_t dimz;
    size_t offx;
    size_t offy;
    size_t offz;
    FLOAT* data;
  } array3d;

  typedef struct {
    size_t dimx;
    size_t dimy;
    size_t dimz;
    size_t dimt;
    size_t offx;
    size_t offy;
    size_t offz;
    size_t offt;
    FLOAT* data;
  } array4d;


  extern array3d slice3d(array4d* a4d, unsigned int t, FLOAT* data, unsigned char from_buffer);
  extern array2d slice2d(array3d* a3d, unsigned int z, FLOAT* data, unsigned char from_buffer);
  extern array1d slice1d(array2d* a2d, unsigned int y, FLOAT* data, unsigned char from_buffer);
  
  extern void convolve_image(array3d* src,
			     array3d* kernel,
			     FLOAT bias,
			     unsigned int dil_x,
			     unsigned int dil_y,
			     array2d* res);
  extern void multi_convolve_image(array3d* src,
				   array4d* kernels,
				   array1d* biases,
				   unsigned int dil_x,
				   unsigned int dil_y,
				   array3d* res);
  extern void relu_max_pool_image(array3d* src,
				  unsigned int size_x,
				  unsigned int size_y,
				  unsigned int dil_x,
				  unsigned int dil_y,
				  array3d* res);

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

