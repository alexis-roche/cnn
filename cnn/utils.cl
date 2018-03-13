__kernel void vector_add(__global const float *A, __global float *B, float c) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    B[i] = A[i] + c;
}


__kernel void convolve_image(__global const float *src_data, __global const float *kernel_data, unsigned int dil_x, unsigned int dil_y, __global float *res_data) {
 
    int i = get_global_id(0);
    int j = get_global_id(1);
    res_data[i] = src_data[i];

}

