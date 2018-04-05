__kernel void relu_max_pool_image(__global float* src_data,
    	                       	  __global unsigned int* src_dim,
			          __global unsigned int* src_off,
                             	  __global unsigned int* size,
			     	  __global unsigned int* dil,
			     	  __global float* res_data,
			     	  __global unsigned int* res_off) {
  unsigned int xc = get_global_id(0);
  unsigned int yc = get_global_id(1);
  unsigned int zc = get_global_id(2);
  float out = 0, tmp;
  unsigned int x, y, z;
  size_t pos_x, pos_y, pos_xy, pos_zc, pos_res;
  size_t inc_x = dil[0] * src_off[0];
  size_t inc_y = dil[1] * src_off[1];
  int alpha, beta;
  __global float *buf;

  alpha = xc - dil[0] * ((size[0] - 1) / 2);
  beta = src_dim[0] - alpha;
  if ((alpha < 0) || (beta < (dil[0] * size[0])))
    return;
  pos_x = alpha * src_off[0];

  alpha = yc - dil[1] * ((size[1] - 1) / 2);
  beta = src_dim[1] - alpha;
  if ((alpha < 0) || (beta < (dil[1] * size[1])))
    return;
  pos_y = alpha * src_off[1];

  pos_zc = zc * src_off[2];

  pos_x += pos_zc;
  for (x=0; x<size[0]; x++) {
    pos_xy = pos_x + pos_y;
    buf = src_data + pos_xy;

    for (y=0; y<size[1]; y++) {
      tmp = *buf;
      if (tmp > out)
	out = tmp;
      pos_xy += inc_y;
      buf += inc_y;
    }

    pos_x += inc_x;
  }

  pos_res = xc * res_off[0] + yc * res_off[1] + zc * res_off[2];
  res_data[pos_res] = out;
}
