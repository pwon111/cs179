#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "cluster_cuda.cuh"

// This assumes address stores the average of n elements atomically updates
// address to store the average of n + 1 elements (the n elements as well as
// val). This might be useful for updating cluster centers.
// modified from http://stackoverflow.com/a/17401122
__device__ 
float atomicUpdateAverage(float* address, int n, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    float next_val = (n * __int_as_float(assumed) + val) / (n + 1);
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(next_val));
  } while (assumed != old);
  return __int_as_float(old);
}

// computes the distance squared between vectors a and b where vectors have
// length size and stride a_stride and b_stride respectively.
__device__ 
float squared_distance(float *a, float *b, int a_stride, int b_stride,
                       int size) {
  float dist = 0.0;
  for (int i=0; i < size; i++) {
    float diff = a[a_stride * i] - b[b_stride * i];
    dist += diff * diff;
  }
  return dist;
}

/*
 * Notationally, all matrices are column majors, so if I say that matrix Z is
 * of size m * n, then the stride in the m axis is 1. For purposes of
 * optimization (particularly coalesced accesses), you can change the format of
 * any array.
 *
 * clusters is a REVIEW_DIM * k array containing the location of each of the k
 * cluster centers.
 *
 * cluster_counts is a k element array containing how many data points are in 
 * each cluster.
 *
 * k is the number of clusters.
 *
 * data is a REVIEW_DIM * batch_size array containing the batch of reviews to
 * cluster. Note that each review is contiguous (so elements 0 through 49 are
 * review 0, ...)
 *
 * output is a batch_size array that contains the index of the cluster to which
 * each review is the closest to.
 *
 * batch_size is the number of reviews this kernel must handle.
 */
__global__
void sloppyClusterKernel(float *clusters, int *cluster_counts, int k, 
                         float *data, int *output, int batch_size) {
  // Stack variables
  float dist, best_dist;
  int best, old;
  // For each input review, go through all the cluster locations and find the
  // closest one, then adjust its location to account for this review
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size;
       i += blockDim.x * gridDim.x) {
    best_dist = -1.0;
    for (int j = 0; j < k; j++) {
      // Both the input data and the cluster locations are transposed and stored
      // in row-major format, so that when reading from and writing to data,
      // each warp will make a coalesced set of 32 consecutive accesses
      dist = squared_distance(clusters + j, data + i, k, batch_size,
                              REVIEW_DIM);

      // Update closest cluster information accordingly
      if (j == 0 || dist < best_dist) {
        best_dist = dist;
        best = j;
      }
    }

    // Indicate which cluster each review in the batch was assigned to
    output[i] = best;

    // Update the cluster count to include this new review
    old = atomicAdd(cluster_counts + best, 1);
    // Go through the components of the cluster location vector and adjust them
    // to reflect the new average position of the reviews in the cluster
    for (int j = 0; j < REVIEW_DIM; j++) {
      atomicUpdateAverage(clusters + best + j * k, old,
                          data[i + j * batch_size]);
    }
  }
}


void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = sizeof(float) * (k * (REVIEW_DIM + 1));

  sloppyClusterKernel<<<grid_size, block_size, shmem_bytes, stream>>>
                     (clusters, cluster_counts, k, data, output, batch_size);
}
