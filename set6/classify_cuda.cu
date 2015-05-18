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
 * data_stride: Stride of the review vectors in the input data, since they are
                transposed for coalescing purposes.
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(float *data, int batch_size, int data_stride,
                       float step_size, float *weights, float *errors) {
  // Shared memory since we're going to be reading the old weights and updating
  // the new values a lot
  __shared__ float old_weights[REVIEW_DIM];
  __shared__ float new_weights[REVIEW_DIM];

  unsigned int i, j;

  // For each block, copy the current weights to shared memory and zero out the
  // new ones
  for (i = threadIdx.x; i < REVIEW_DIM; i += blockDim.x) {
    old_weights[i] = weights[i];
    new_weights[i] = 0.0;
  }

  // Make sure each block's weights are set up before starting any calculations
  __syncthreads();

  // Have each thread compute its contribution to the gradient
  float grad_n_divisor, y_n, sum = 0.0;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size;
       i += blockDim.x * gridDim.x) {
    // Get whether the review is actually a restaurant or not
    y_n = data[i + REVIEW_DIM * data_stride];
    
    // Compute w^T x_n, i.e. the dot product of w and x_n
    for (j = 0; j < REVIEW_DIM; j++)
      sum += old_weights[j] * data[i + j * data_stride];

    // Compute the 1.0 + exp(y_n w^T x_n) divisor component of the gradient
    grad_n_divisor = 1.0 + expf(y_n * sum);

    // We know we misclassified the review when y_n w^T x_n is negative, which
    // we can check for by comparing 1 + expf(y_n w^T x_n) to 2, since e^k where
    // k < 0 is less than 1
    if (grad_n_divisor <= 2.0)
      atomicAdd(errors, 1);
    
    // Add to each component of the gradient vector, without dividing by
    // batch_size or multiplying by -1
    for (j = 0; j < REVIEW_DIM; j++)
      atomicAdd(new_weights + j, y_n * data[i + j * data_stride] / grad_n_divisor);
  }

  // Make sure all of our new weights are updated
  __syncthreads();

  // Divide the gradient vector by N, multiply it by the step size, and subtract
  // it from the weights (I know this is addition, but I never multiply my
  // gradient vector by -1); it's okay to compute part of the gradient
  // separately in each block since multiplication and addition are linear
  for (i = threadIdx.x; i < REVIEW_DIM; i+= blockDim.x) {
    atomicAdd(weights + i, step_size * new_weights[i] / batch_size);
  }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(float *data, int batch_size, int data_stride,
                   float step_size, float *weights) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = 0;

  float *d_errors;
  cudaMalloc(&d_errors, sizeof(float));
  cudaMemset(d_errors, 0, sizeof(float));

  trainLogRegKernel<<<grid_size, block_size, shmem_bytes>>>(data,
                                                            batch_size,
                                                            data_stride,
                                                            step_size,
                                                            weights,
                                                            d_errors);

  float h_errors = -1.0;
  cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
  cudaFree(d_errors);
  return h_errors;
}
