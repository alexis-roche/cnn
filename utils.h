#ifndef UTILS
#define UTILS

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

#ifdef FLOAT64
  typedef double FLOAT;
#else
  typedef float FLOAT;
#endif
  
  typedef struct {
    size_t dim;
    size_t off;
    FLOAT* data;
    unsigned char owner;
  } array1d;

  typedef struct {
    size_t dimx;
    size_t dimy;
    size_t offx;
    size_t offy;
    FLOAT* data;
    unsigned char owner;
  } array2d;

  typedef struct {
    size_t dimx;
    size_t dimy;
    size_t dimz;
    size_t offx;
    size_t offy;
    size_t offz;
    FLOAT* data;
    unsigned char owner;
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
    unsigned char owner;
  } array4d;

  extern array1d* array1d_new(unsigned int dim);
  extern void array1d_delete(array1d* thisone);
  extern array1d* array1d_new_contiguous_from(array1d* src);
  
  extern array2d* array2d_new(unsigned int dimx, unsigned int dimy);
  extern void array2d_delete(array2d* thisone);
  extern array2d* array2d_new_contiguous_from(array2d* src);
  
  extern array3d* array3d_new(unsigned int dimx, unsigned int dimy, unsigned int dimz);
  extern void array3d_delete(array3d* thisone);
  extern array3d* array3d_new_contiguous_from(array3d* src);
  
  extern array4d* array4d_new(unsigned int dimx, unsigned int dimy, unsigned int dimz, unsigned int dimt);
  extern void array4d_delete(array4d* thisone);
  extern array4d* array4d_new_contiguous_from(array4d* src);

  extern void copy1d(array1d* src, array1d* res, unsigned char to_left);
  extern void copy2d(array2d* src, array2d* res, unsigned char to_left);
  extern void copy3d(array3d* src, array3d* res, unsigned char to_left);
  extern void copy4d(array4d* src, array4d* res, unsigned char to_left);

  extern void slice1d(array2d* a2d, array1d* a1d, unsigned int y, unsigned char from_buffer);
  extern void slice2d(array3d* a3d, array2d* a2d, unsigned int z, unsigned char from_buffer);
  extern void slice3d(array4d* a4d, array3d* a3d, unsigned int t, unsigned char from_buffer);
  
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

