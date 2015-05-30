/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH

void call_update_nodes_kernel(unsigned int grid_size, unsigned int block_size,
                              float *old, float *current, int length,
                              float courant_squared);

void call_update_boundaries_kernel(float *current, float left_boundary_value,
                                   int length);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH
