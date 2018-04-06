#include "utils.h"


array1d* array1d_new(unsigned int dim)
{
  array1d* thisone = malloc(sizeof(array1d));
  thisone->dim = dim;
  thisone->off = 1;
  thisone->data = (FLOAT*)calloc(dim, sizeof(FLOAT));
  thisone->owner = 1;
  return thisone;
}

void array1d_delete(array1d* thisone)
{
  if ((thisone->owner) && (thisone->data != NULL))
    free(thisone->data);
  free(thisone);
}

array1d* array1d_new_contiguous_from(array1d* src)
{
  array1d* thisone = malloc(sizeof(array1d));
  thisone->dim = src->dim;

  if (src->off == 1) {
    thisone->off = 1;
    thisone->data = src->data;
    thisone->owner = 0;
  }
  else {
    thisone->off = 1;
    thisone->data = (FLOAT*)calloc(src->dim, sizeof(FLOAT));
    thisone->owner = 1;
    copy1d(src, thisone, 0); 
  }
  
  return thisone;
}


array2d* array2d_new(unsigned int dimx, unsigned int dimy)
{
  array2d* thisone = malloc(sizeof(array2d));
  thisone->dimx = dimx;
  thisone->dimy = dimy;
  thisone->offy = 1;
  thisone->offx = dimy;
  thisone->data = (FLOAT*)calloc(dimx * dimy, sizeof(FLOAT));
  thisone->owner = 1;
  return thisone;
}

void array2d_delete(array2d* thisone)
{
  if ((thisone->owner) && (thisone->data != NULL))
    free(thisone->data);
  free(thisone);
}


array2d* array2d_new_contiguous_from(array2d* src)
{
  array2d* thisone = malloc(sizeof(array2d));
  thisone->dimx = src->dimx;
  thisone->dimy = src->dimy;

  size_t a = src->dimx * src->dimy - 1;
  size_t b = src->offx * (src->dimx - 1) + src->offy * (src->dimy - 1);

  if (a == b) {
    thisone->offy = src->offy;
    thisone->offx = src->offx;
    thisone->data = src->data;
    thisone->owner = 0;
  }
  else {
    thisone->offy = 1;
    thisone->offx = src->dimy;
    thisone->data = (FLOAT*)calloc(src->dimx * src->dimy, sizeof(FLOAT));
    thisone->owner = 1;
    copy2d(src, thisone, 0);
  }
  
  return thisone;
}


array3d* array3d_new(unsigned int dimx, unsigned int dimy, unsigned int dimz)
{
  array3d* thisone = malloc(sizeof(array3d));
  thisone->dimx = dimx;
  thisone->dimy = dimy;
  thisone->dimz = dimz;
  thisone->offz = 1;
  thisone->offy = dimz;
  thisone->offx = dimy * thisone->offy;
  thisone->data = (FLOAT*)calloc(dimx * dimy * dimz, sizeof(FLOAT));
  thisone->owner = 1;
  return thisone;
}

void array3d_delete(array3d* thisone)
{
  if ((thisone->owner) && (thisone->data != NULL))
    free(thisone->data);
  free(thisone);
}

array3d* array3d_new_contiguous_from(array3d* src)
{
  array3d* thisone = malloc(sizeof(array3d));
  thisone->dimx = src->dimx;
  thisone->dimy = src->dimy;
  thisone->dimz = src->dimz;

  size_t a = src->dimx * src->dimy * src->dimz - 1;
  size_t b = src->offx * (src->dimx - 1) + src->offy * (src->dimy - 1) + src->offz * (src->dimz - 1);

  if (a == b) {
    thisone->offz = src->offz;
    thisone->offy = src->offy;
    thisone->offx = src->offx;
    thisone->data = src->data;
    thisone->owner = 0;
  }
  else {
    thisone->offz = 1;
    thisone->offy = src->dimz;
    thisone->offx = src->dimy * thisone->offy;
    thisone->data = (FLOAT*)calloc(src->dimx * src->dimy * src->dimz, sizeof(FLOAT));
    thisone->owner = 1;
    copy3d(src, thisone, 0);
  }
  
  return thisone;
}


array4d* array4d_new(unsigned int dimx, unsigned int dimy, unsigned int dimz, unsigned int dimt)
{
  array4d* thisone = malloc(sizeof(array4d));
  thisone->dimx = dimx;
  thisone->dimy = dimy;
  thisone->dimz = dimz;
  thisone->dimt = dimt;
  thisone->offt = 1;
  thisone->offz = dimt;
  thisone->offy = dimz * thisone->offt;
  thisone->offx = dimy * thisone->offy;
  thisone->data = (FLOAT*)calloc(dimx * dimy * dimz * dimt, sizeof(FLOAT));
  thisone->owner = 1;
  return thisone;
}

void array4d_delete(array4d* thisone)
{
  if ((thisone->owner) && (thisone->data != NULL))
    free(thisone->data);
  free(thisone);
}


array4d* array4d_new_contiguous_from(array4d* src)
{
  array4d* thisone = malloc(sizeof(array4d));
  thisone->dimx = src->dimx;
  thisone->dimy = src->dimy;
  thisone->dimz = src->dimz;
  thisone->dimt = src->dimt;

  size_t a = src->dimx * src->dimy * src->dimz * src->dimt - 1;
  size_t b = src->offx * (src->dimx - 1) + src->offy * (src->dimy - 1) + src->offz * (src->dimz - 1) + src->offt * (src->dimt - 1);

  if (a == b) {
    thisone->offt = src->offt;
    thisone->offz = src->offz;
    thisone->offy = src->offy;
    thisone->offx = src->offx;
    thisone->data = src->data;
    thisone->owner = 0;
  }
  else {
    thisone->offt = 1;
    thisone->offz = src->dimt;
    thisone->offy = src->dimz * thisone->offz;
    thisone->offx = src->dimy * thisone->offy;
    thisone->data = (FLOAT*)calloc(src->dimx * src->dimy * src->dimz * src->dimt, sizeof(FLOAT));
    thisone->owner = 1;
    copy4d(src, thisone, 0);
  }

  return thisone;
}


void copy1d(array1d* src, array1d* res, unsigned char to_left)
{
  FLOAT *buf = res->data, *buf_src = src->data;
  unsigned int x;

  if (res->off != 1)
    return;
  
  for(x=0; x<src->dim; x++, buf++, buf_src+=src->off) {
    if (to_left)
      *buf_src = *buf;
    else
      *buf = *buf_src;
  }
}

void copy2d(array2d* src, array2d* res, unsigned char to_left)
{
  FLOAT *buf = res->data, *buf_src;
  unsigned int x, y;
  size_t pos_x;

  if (res->offy != 1)
    return;

  for(x=0, pos_x=0; x<src->dimx; x++, pos_x+=src->offx) {
    buf_src = src->data + pos_x;
    for(y=0; y<src->dimy; y++, buf++, buf_src+=src->offy) {
      if (to_left)
	*buf_src = *buf;
      else
	*buf = *buf_src;
    }
  }
}


void copy3d(array3d* src, array3d* res, unsigned char to_left)
{
  FLOAT *buf = res->data, *buf_src;
  unsigned int x, y, z;
  size_t pos_x, pos_xy;

  if (res->offz != 1)
    return;

  for(x=0, pos_x=0; x<src->dimx; x++, pos_x+=src->offx) {
    for(y=0, pos_xy=pos_x; y<src->dimy; y++, pos_xy+=src->offy) {	
      buf_src = src->data + pos_xy;
      for(z=0; z<src->dimz; z++, buf++, buf_src+=src->offz) {
	if (to_left)
	  *buf_src = *buf;
	else
	  *buf = *buf_src;
      }
    }
  }
}


void copy4d(array4d* src, array4d* res, unsigned char to_left)
{
  FLOAT *buf = res->data, *buf_src;
  unsigned int x, y, z, t;
  size_t pos_x, pos_xy, pos_xyz;

  if (res->offt != 1)
    return;
  
  for(x=0, pos_x=0; x<src->dimx; x++, pos_x+=src->offx) {
    for(y=0, pos_xy=pos_x; y<src->dimy; y++, pos_xy+=src->offy) {	
      for(z=0, pos_xyz=pos_xy; z<src->dimz; z++, pos_xyz+=src->offz) {
	buf_src = src->data + pos_xyz;
	for(t=0; t<src->dimt; t++, buf++, buf_src+=src->offt) {
	  if (to_left)
	    *buf_src = *buf;
	  else
	    *buf = *buf_src;
	}
      }
    }
  }    
}


void slice3d(array4d* a4d, array3d* a3d, unsigned int t, unsigned char to_left)
{
  FLOAT *buf_a3d, *buf_a4d;
  unsigned int x, y, z;
  size_t pos_x, pos_xy;
    
  buf_a3d = a3d->data;
  pos_x = t * a4d->offt;
  for(x=0; x<a3d->dimx; x++) {
    pos_xy = pos_x;
    for(y=0; y<a3d->dimy; y++) {
      buf_a4d = a4d->data + pos_xy;
      for(z=0; z<a3d->dimz; z++) {
	if (to_left)
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
}

void slice2d(array3d* a3d, array2d* a2d, unsigned int z, unsigned char to_left)
{
  FLOAT *buf_a2d, *buf_a3d;
  unsigned int x, y;
  size_t pos_x;
  
  buf_a2d = a2d->data;
  pos_x = z * a3d->offz;
  for(x=0; x<a2d->dimx; x++) {
      buf_a3d = a3d->data + pos_x;
      for(y=0; y<a2d->dimy; y++) {
	if (to_left)
	  *buf_a3d = *buf_a2d;
	else
	  *buf_a2d = *buf_a3d;
	buf_a2d ++;
	buf_a3d += a3d->offy;
      }
      pos_x += a3d->offx;
  }
}


void slice1d(array2d* a2d, array1d* a1d, unsigned int y, unsigned char to_left)
{
  FLOAT *buf_a1d, *buf_a2d;
  unsigned int x;
    
  buf_a1d = a1d->data;
  buf_a2d = a2d->data + y * a2d->offy;
  for(x=0; x<a1d->dim; x++) {
    if (to_left)
      *buf_a2d = *buf_a1d;
    else
      *buf_a1d = *buf_a2d;
    buf_a1d ++;
    buf_a2d += a2d->offx;
  }
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
  size_t inc_x = dil_x * src->offx;
  size_t inc_y = dil_y * src->offy;
  int alpha, beta;
  FLOAT *buf;
  
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
