#ifndef RUN_UTILS
#define RUN_UTILS

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
  
#ifdef __cplusplus
}
#endif

#endif

