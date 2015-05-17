/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {
    cufftComplex i, j;
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
         n < padded_length; n += blockDim.x * gridDim.x) {
        i = raw_data[n];
        j = impulse_v[n];
        // Complex multiplication, scale by length
        out_data[n].x = (i.x * j.x - i.y * j.y) / padded_length;
        // Would be more efficient to just set to 0, but I wasn't sure if that's
        // what you (the graders) want or not
        out_data[n].y = (i.x * j.y + i.y * j.x) / padded_length;
    }
}

/**
 * My implementation of this kernel is as follows: first, each block gets its
 * own float array in shared memory of length blockDim.x, and each thread in the
 * block fills in index threadIdx.x by taking the maximum real component of all
 * values at indices n + k * blockDim.x * gridDim.x, where n is the thread's
 * unique identifier blockIdx.x * blockDim.x + threadIdx.x, and k is an integer
 * starting from 0. As long as blockDim.x is a nice multiple of 32, these reads
 * will be coalesced (since they're consecutive and would start on cache lines),
 * and all the access to shared memory are sequential so there's no bank
 * conflicts there. Next, the first blockDim.x / 2 threads in each block replace
 * the float at index threadIdx.x in shared memory with the max between it and
 * that at threadIdx.x + blockDim.x / 2, which doesn't cause a bank conflict
 * since it's an atomic operation. The first blockDim.x / 4 threads in the block
 * then repeat this process on the resulting blockDim.x / 2 values, and then
 * again with blockDim.x / 8 threads on blockDim.x / 4 values, et cetera until
 * the first index of shared memory contains the highest value in the block.
 * This is then atomically compared to the value at max_abs_val (which starts at
 * 0), which is updated if necessary. This is computationally inefficient since
 * the gridDim.x blocks must wait for other ones to do atomicMax() first, but
 * as I explain below, resulted in significantly faster running times than an
 * alternate, entirely parallel solution I also developed.
 *
 * I also wrote another version of this process, which instead of letting each
 * block atomically compare to the stored absolute value uses the same process
 * as this kernel to reduce the blockDim.x values produced by it to an
 * increasingly smaller number of maximum values until only one is left, but
 * this actually led to a 50% performance hit on my machine, which I suspect is
 * due to the I/O operations required to repeatedly store the new, reduced set
 * of possible maximum values in global memory, and then read it back out again
 * on the kernel's next iteration. This kernel is included in the file
 * fft_convolve_cuda_bad.cu.
 */
__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    // Shared list of blockDim.x floats
    extern __shared__ float pass[];

    // Populate all the shared arrays with one value per thread, looping as
    // necessary to deal with arrays longer than the number of threads
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    pass[threadIdx.x] = 0;
    for (; n < padded_length; n += blockDim.x * gridDim.x)
        atomicMax(pass + threadIdx.x, fabs(out_data[n].x));
    __syncthreads();

    n = blockIdx.x * blockDim.x + threadIdx.x;

    // Iteratively fold the list in half until there's just one max value at
    // the first index ((slots + 1) / 2 rounds down, slots / 2 rounds up)
    for (int slots = blockDim.x; slots > 1; slots = (slots + 1) / 2) {
        if (threadIdx.x < slots / 2) {
            atomicMax(pass + threadIdx.x, pass[threadIdx.x + (slots + 1) / 2]);
        }
        __syncthreads();
    }

    // Atomically compare to current max value and update if necessary
    if (threadIdx.x == 0)
        atomicMax(max_abs_val, pass[0]);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    float max_mag = *max_abs_val;
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
         n < padded_length; n += blockDim.x * gridDim.x) {
        out_data[n].x /= max_mag;
        // Would be more efficient to just set to 0, but I wasn't sure if that's
        // what you (the graders) want or not
        out_data[n].y /= max_mag;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v,
                                                     out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaMaximumKernel<<<blocks, threadsPerBlock,
                        min((unsigned int) (threadsPerBlock * sizeof(float)),
                            padded_length)>>>
                     (out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val,
                                                  padded_length);
}
