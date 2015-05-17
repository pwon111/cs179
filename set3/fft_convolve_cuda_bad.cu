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
 * This kernel is very similar to the one I submitted in fft_convolve_cuda.cu,
 * in that it breaks the list into up to gridDim.x chunks of blockDim.x floats,
 * then folds these in half repeatedly using atomicMax. However, at the end it
 * puts the results into global memory at index blockIdx.x of work_area,
 * producing a list greatly reduced in size that can be iterated over again with
 * the same "folding" method, using the second kernel.
 */
__global__
void
cudaMaximumFirstKernel(cufftComplex *out_data, float *work_area,
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

    int next;
    // Iteratively fold the list in half until there's just one max value at
    // the first index ((slots + 1) / 2 rounds down, slots / 2 rounds up)
    for (int slots = blockDim.x; slots > 1; slots = (slots + 1) / 2) {
        if (threadIdx.x < slots / 2) {
            atomicMax(pass + threadIdx.x, pass[threadIdx.x + (slots + 1) / 2]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        work_area[blockIdx.x] = pass[0];
}

/**
 * After the first kernel has reduced the list to a much smaller one in global
 * memory, this kernel retrieves it and passes over it again, storing the
 * further-reduced list back into work_area if it was updated with more than one
 * value, or just storing the single maximum magnitude if that's all that's
 * left. Since everything here is a linear array and we're only ever accessing
 * consecutive elements, all memory transactions are again bank conflict-free
 * or coalesced as long as the number of threads per block is a multiple of 32.
 */
__global__
void
cudaMaximumSecondKernel(float *work_area, float *max_abs_val, int new_length) {
    // Shared list of blockDim.x floats
    extern __shared__ float pass[];

    pass[threadIdx.x] = 0;
    unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < new_length) {
        // Read in maximums from previous pass
        pass[threadIdx.x] = work_area[n];
        __syncthreads();

        int next;
        // Repeat the iterative "folding" process where the first half is
        // compared to the second, then the first quarter to the second, etc.
        for (int slots = blockDim.x; slots > 1; slots = (slots + 1) / 2) {
            if (threadIdx.x < slots / 2) {
                atomicMax(pass + threadIdx.x,
                          pass[threadIdx.x + (slots + 1) / 2]);
            }
            __syncthreads();
        }

        // Store results in work area if not done, otherwise store max magnitude
        if (threadIdx.x == 0)
            if (new_length <= blockDim.x)
                *max_abs_val = pass[0];
            else
                work_area[blockIdx.x] = pass[0];
    }
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
        float *work_area,
        const unsigned int padded_length) {
    // Fill work area with up to gridDim.x maximum values from first pass
    cudaMaximumFirstKernel<<<blocks, threadsPerBlock,
                             threadsPerBlock * sizeof(float)>>>
                          (out_data, work_area, padded_length);

    // Pass over those values again, then again, etc. until there's only one
    for (int slots = min(blocks, (padded_length - 1) / threadsPerBlock + 1);
         slots > 1; slots = (slots - 1) / threadsPerBlock + 1) {
        cudaMaximumSecondKernel<<<blocks, threadsPerBlock,
                                  threadsPerBlock * sizeof(float)>>>
                               (work_area, max_abs_val, slots);
    }
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val,
                                                  padded_length);
}
