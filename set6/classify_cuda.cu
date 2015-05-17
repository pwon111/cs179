#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(float *data, int batch_size, float step_size,
		       float *weights, float *spare_weights, float *errors) {
  __shared__ float old_weights[REVIEW_DIM];

  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int d = blockDim.x * gridDim.x;
  unsigned int i, j;
  for (i = threadIdx.x; i < REVIEW_DIM; i += d) {
    old_weights[i] = weights[i];
  }

  __syncthreads();
  float grad_n_divisor, y_n, sum = 0.0;
  for (i = n; i < batch_size; i += d) {
    y_n = data[i + REVIEW_DIM * batch_size];
    
    for (j = 0; j < REVIEW_DIM; j++)
      sum += old_weights[j] * data[i + j * batch_size];

    grad_n_divisor = 1.0 + expf(y_n * sum);

    if (grad_n_divisor <= 2.0)
      atomicAdd(errors, 1);
    
    for (j = 0; j < REVIEW_DIM; j++)
      atomicAdd(spare_weights + j, y_n * data[i + j * batch_size] / grad_n_divisor);
  }

  __syncthreads();
  for (i = n; i < REVIEW_DIM; i+= d) {
    weights[i] = old_weights[i] + step_size * spare_weights[i] / batch_size;
  }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(float *data, int batch_size,
                   float step_size, float *weights, float *spare_weights) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = 0;

  float *d_errors;
  cudaMalloc(&d_errors, sizeof(float));
  cudaMemset(d_errors, 0, sizeof(float));

  trainLogRegKernel<<<grid_size, block_size, shmem_bytes>>>(data,
                                                            batch_size,
                                                            step_size,
                                                            weights,
                                                            spare_weights,
                                                            d_errors);

  float h_errors = -1.0;
  cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
  cudaFree(d_errors);
  return h_errors;
}
