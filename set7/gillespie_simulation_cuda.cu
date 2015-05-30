#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>

#include "gillespie_simulation_cuda.cuh"

__global__
void simulation_step_kernel(float *samples, int *concentrations, int *states,
                            float *times, int sim_count, float b, float g,
                            float K_on, float K_off) {
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; n < sim_count;
         n += blockDim.x * gridDim.x) {
        int state = states[n];
        float event_sample = samples[n];
        float time_sample = samples[sim_count + n];
        float propensity_sum;

        if (state == 0) {
            propensity_sum = K_on + concentrations[n] * g;
            if (event_sample <= K_on / propensity_sum)
                states[n] = 1;
            else
                concentrations[n]--;
        
            times[n] -= logf(time_sample) / propensity_sum;
        }
        else if (state == 1) {
            propensity_sum = K_off + b + concentrations[n] * g;
            float first_bin = K_off / propensity_sum;
            if (event_sample <= first_bin)
                states[n] = 0;
            else if (event_sample <= first_bin + b / propensity_sum) {
                concentrations[n]++;
            }
            else
                concentrations[n]--;
        
            times[n] -= logf(time_sample) / propensity_sum;
        }
    }
}

__global__
void timestep_update_kernel(int *concentrations, int *states, float *times,
                            int *timestep_samples, int sim_count,
                            int timestep_count, int *done) {
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; n < sim_count;
         n += blockDim.x * gridDim.x) {
        if (states[n] == -1)
            break;

        unsigned int actual_i;
        for (unsigned int i = 0; i < timestep_count; i++) {
            actual_i = i * sim_count + n;
            if (timestep_samples[actual_i] == -1) {
                if ((unsigned int) times[n] >= i * 10) {
                    timestep_samples[actual_i] = concentrations[n];
                    if (i + 1 == timestep_count) {
                        states[n] = -1;
                        break;
                    }
                    atomicExch(done, 0);
                }
                else {
                    atomicExch(done, 0);
                    break;
                }
            }
        }
    }
}

__global__
void expected_means_kernel(int *timestep_samples, float *means, int sim_count,
                           int timestep_count) {
    extern __shared__ float values[];

    int odd_sim_count = sim_count & 1;
    unsigned int slots = (sim_count + 1) / 2;
    for (unsigned int j = blockIdx.x; j < timestep_count; j += gridDim.x) {
        unsigned int offset = j * sim_count;
        for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x) {
            if (odd_sim_count && i == slots - 1)
                values[i] = (float) timestep_samples[offset + i];
            else
                values[i] = (float) (timestep_samples[offset + i] +
                                     timestep_samples[offset + i + slots]);
        }

        odd_sim_count = slots & 1;

        for (slots = (slots + 1) / 2; slots > 1; slots = (slots + 1) / 2) {
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x) {
                if (odd_sim_count && i == slots - 1)
                    values[i] = values[i];
                else
                    values[i] = values[i] + values[i + slots];
            }

            odd_sim_count = slots & 1;
        }

        if (threadIdx.x == 0)
            means[j] = values[0] / sim_count;
    }
}

__global__
void expected_variances_kernel(int *timestep_samples, float *means,
                               float *variances, int sim_count,
                               int timestep_count) {
    extern __shared__ float values[];

    for (unsigned int j = blockIdx.x; j < timestep_count; j += gridDim.x) {
        unsigned int offset = j * sim_count;
        float difference;
        for (unsigned int i = threadIdx.x; i < sim_count; i += blockDim.x) {
            difference = (float) timestep_samples[offset + i] - means[j];
            values[i] = difference * difference;
        }

        int odd_sim_count = sim_count & 1;

        for (unsigned int slots = (sim_count + 1) / 2; slots > 1;
             slots = (slots + 1) / 2) {
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x) {
                if (odd_sim_count && i == slots - 1)
                    values[i] = values[i];
                else
                    values[i] = values[i] + values[i + slots];
            }

            odd_sim_count = slots & 1;
        }

        if (threadIdx.x == 0)
            variances[j] = values[0] / sim_count;
    }
}

void call_simulation_step_kernel(float *samples, int *concentrations, int *states,
                                 float *times, int sim_count, float b, float g,
                                 float K_on, float K_off) {
    simulation_step_kernel<<<1, sim_count>>>(samples, concentrations, states,
                                             times, sim_count, b, g, K_on,
                                             K_off);
}

void call_timestep_update_kernel(int *concentrations, int *states, float *times,
                                 int *timestep_samples, int sim_count,
                                 int timestep_info_sim_count, int *done) {
    cudaMemset(done, 1, sizeof(int));
    timestep_update_kernel<<<1, sim_count>>>(concentrations, states, times,
                                             timestep_samples, sim_count,
                                             timestep_info_sim_count, done);
}

void call_expected_means_kernel(int *timestep_samples, float *means,
                                int sim_count, int timestep_count) {
    expected_means_kernel<<<timestep_count,
                            sim_count,
                            sim_count * sizeof(float)>>>
                         (timestep_samples, means, sim_count, timestep_count);
}

void call_expected_variances_kernel(int *timestep_samples, float *means,
                                    float *variances, int sim_count,
                                    int timestep_count) {
    expected_variances_kernel<<<timestep_count,
                                sim_count,
                                sim_count * sizeof(float)>>>
                             (timestep_samples, means, variances, sim_count,
                              timestep_count);
}