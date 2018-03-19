#include "utils.h"


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
