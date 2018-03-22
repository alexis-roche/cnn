__kernel void test1d(__global float *A, __global float *B, float c) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    B[i] = A[i] + c;
}


__kernel void convolve_image(__global float* src_data,
                             __global unsigned int* src_dim,
			     __global unsigned int* src_off,
                             __global float* kernel_data,
                             __global unsigned int* kernel_dim,
			     __global unsigned int* kernel_off,
			     __global unsigned int* dil,
			     float bias,
			     __global float* res_data,
			     __global unsigned int* res_off) {

  unsigned int xc = get_global_id(0);
  unsigned int yc = get_global_id(1);
  float out = bias;
  size_t inc_src_x = dil[0] * src_off[0];
  size_t inc_src_y = dil[1] * src_off[1];
  unsigned int x, y, z;
  size_t pos_x_kernel = 0, pos_y_kernel = 0, pos_xy_kernel, pos_x_src, pos_y_src, pos_xy_src, pos_res;
  int alpha, beta;
  __global float *buf_kernel, *buf_src;

  alpha = xc - dil[0] * ((kernel_dim[0] - 1) / 2);
  beta = src_dim[0] - alpha;
  if ((alpha < 0) || (beta < (dil[0] * kernel_dim[0])))
    return;
  pos_x_src = alpha * src_off[0];

  alpha = yc - dil[1] * ((kernel_dim[1] - 1) / 2);
  beta = src_dim[1] - alpha;
  if ((alpha < 0) || (beta < (dil[1] * kernel_dim[1])))
    return;
  pos_y_src = alpha * src_off[1];

  for (x=0; x<kernel_dim[0]; x++) {
    pos_xy_kernel = pos_x_kernel + pos_y_kernel;
    pos_xy_src = pos_x_src + pos_y_src;

    for (y=0; y<kernel_dim[1]; y++) {
      buf_kernel = kernel_data + pos_xy_kernel;
      buf_src = src_data + pos_xy_src;

      for (z=0; z<kernel_dim[2]; z++) {
	out += (*buf_kernel) * (*buf_src);
	buf_kernel += kernel_off[2];
	buf_src += src_off[2];
      }
      pos_xy_kernel += kernel_off[1];
      pos_xy_src += inc_src_y;    
    }
    pos_x_kernel += kernel_off[0];
    pos_x_src += inc_src_x;
  }

  pos_res = xc * res_off[0] + yc * res_off[1];
  res_data[pos_res] = out;
}