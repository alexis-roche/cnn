__kernel void vector_add(__global const float *A, __global float *B, float c) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    B[i] = A[i] + c;
}