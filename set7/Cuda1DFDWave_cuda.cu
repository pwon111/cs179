/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"

/**
 * This kernel uses each node's (and its neighbors') y-value(s) at times t and
 * t - 1 to calculate it at t + 1, for nodes 1 through length - 2 (where the
 * nodes are 0-indexed).
 */
__global__
void update_nodes_kernel(float *old, float *current, int length,
                         float courant_squared) {
    float current_val;
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
         n < length - 1; n += blockDim.x * gridDim.x) {
        // The current y-value gets used a couple times, so make sure we don't
        // have to dereference it more than once
        current_val = current[n];
        // Use the wave equation to apply the update rule, storing the new value
        // in the "old" array so that we can just swap it with the "current" one
        // once the kernel has finished
        old[n] = 2 * current_val - old[n] +
                 courant_squared * (current[n + 1] - 2 * current_val +
                                    current[n - 1]);
    }
}

/**
 * This kernel updates the boundary nodes, namely those at indices 0 and
 * length - 1. The former's value is calculated on the CPU and passed in as an
 * argument, while the latter's is always 0.0.
 */
__global__
void update_boundaries_kernel(float *current, float left_boundary_value,
                              int length) {
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n == 0)
        current[0] = left_boundary_value;
    else if (n == 1)
        current[length - 1] = 0.0;
}

/**
 * This function calls the general update kernel with the desired grid and block
 * size. The kernel itself supports arbitrary grid and block sizes.
 */
void call_update_nodes_kernel(unsigned int grid_size, unsigned int block_size,
                              float *old, float *current, int length,
                              float courant_squared) {
    update_nodes_kernel<<<grid_size, block_size>>>(old, current, length,
                                                   courant_squared);
}

/**
 * This function calls the boundary node update kernel with two blocks of one
 * thread each; we only need two threads, but if they were in one block they
 * be de facto in the same warp, and would diverge, leading to poorer
 * performance than just putting them each in their own block. The kernel
 * itself supports arbitrary grid and block sizes.
 */
void call_update_boundaries_kernel(float *current, float left_boundary_value,
                                   int length) {
    update_boundaries_kernel<<<2, 1>>>(current, left_boundary_value, length);
}