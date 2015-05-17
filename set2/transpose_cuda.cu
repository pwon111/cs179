#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

/**
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
*/


/**
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  // Writing to output isn't coalesced, since threads in a warp will all have
  // the same j but different i's, so each thread will be writing to cache lines
  // n indices apart, and n is way bigger than 32. Reading from input is,
  // though, since n is a multiple of 32 so n * j puts us at the start of a
  // cache line, and then thread i reads the ith consecutive member of the line
  for (; j < end_j; j++) {
    output[j + n * i] = input[i + n * j];
  }
}

/**
 * If we make our columns 65 members long, then values in the same row but
 * different columns are in different banks since a stride of 65 loops around
 * twice and puts us one bank over from the previous one. We can just put our 64
 * matrix elements in the first 64 indices and use the last one for padding.
 * Thus, writing to our shared memory avoids bank conflicts, and reading from it
 * does simply by the order we do it in, as k and threadIdx.y will be constant
 * across a warp and so it'll just read 32 banks with stride 1.
 *
 * Additionally, all of the transactions with global memory are coalesced, as
 * described in my comment in the naive transpose kernel.
 */
__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
  __shared__ float data[64 * 65];

  int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  int k = 0;

  for (k = 0; k < 4; k++) {
    data[(4 * threadIdx.y + k) + 65 * threadIdx.x] = input[i + n * (j + k)];
  }

  // Make sure we don't start writing output before we're done reading input
  __syncthreads();

  // Swap the blockIdx's since the 64x64 section of matrix flips across the
  // diagonal
  i = threadIdx.x + 64 * blockIdx.y;
  j = 4 * threadIdx.y + 64 * blockIdx.x;

  for (k = 0; k < 4; k++) {
    output[i + n * (j + k)] = data[threadIdx.x + 65 * (4 * threadIdx.y + k)];
  }
}

/**
 * To improve the performance of the shmem kernel, I unrolled the two for loops
 * so that the program doesn't have to check if the loop's breakout condition
 * has been met.
 */
__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  // TODO: This should be based off of your shmemTransposeKernel.
  // Use any optimization tricks discussed so far to improve performance.
  // Consider ILP and loop unrolling.

  // See the shmem kernel for explanation of stride 65 + why there are no bank
  // conflicts or non-coalesced memory accesses
  __shared__ float data[64 * 65];

  int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;

  // Unroll the loop for performance
  data[(4 * threadIdx.y) + 65 * threadIdx.x]     = input[i + n * j];
  data[(4 * threadIdx.y + 1) + 65 * threadIdx.x] = input[i + n * (j + 1)];
  data[(4 * threadIdx.y + 2) + 65 * threadIdx.x] = input[i + n * (j + 2)];
  data[(4 * threadIdx.y + 3) + 65 * threadIdx.x] = input[i + n * (j + 3)];

  __syncthreads();

  i = threadIdx.x + 64 * blockIdx.y;
  j = 4 * threadIdx.y + 64 * blockIdx.x;

  // Unroll the loop for performance
  output[i + n * j]       = data[threadIdx.x + 65 * (4 * threadIdx.y)];
  output[i + n * (j + 1)] = data[threadIdx.x + 65 * (4 * threadIdx.y + 1)];
  output[i + n * (j + 2)] = data[threadIdx.x + 65 * (4 * threadIdx.y + 2)];
  output[i + n * (j + 3)] = data[threadIdx.x + 65 * (4 * threadIdx.y + 3)];
}

void cudaTranspose(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type) {
  if (type == NAIVE) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == SHMEM) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == OPTIMAL) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else {
    // unknown type
    assert(false);
  }
}
