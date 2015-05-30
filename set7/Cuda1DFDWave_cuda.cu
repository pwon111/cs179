/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


__global__
void update_nodes_kernel(float *old, float *current, int length,
                         float courant_squared) {
    float current_val;
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
         n < length - 1; n += blockDim.x * gridDim.x) {
        current_val = current[n];
        old[n] = 2 * current_val - old[n]
                     + courant_squared * (current[n + 1]
                                          - 2 * current_val + current[n - 1]);
    }
}

__global__
void update_boundaries_kernel(float *current, float left_boundary_value,
                              int length) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n == 0)
        current[0] = left_boundary_value;
    else if (n == 1)
        current[length - 1] = 0.0;
}

void call_update_nodes_kernel(unsigned int grid_size, unsigned int block_size,
                              float *old, float *current, int length,
                              float courant_squared) {
    update_nodes_kernel<<<grid_size, block_size>>>(old, current, length,
                                                   courant_squared);
}

void call_update_boundaries_kernel(float *current, float left_boundary_value,
                                   int length) {
    update_boundaries_kernel<<<2, 1>>>(current, left_boundary_value, length);
}