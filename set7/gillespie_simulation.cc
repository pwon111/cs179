#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <curand.h>

#include "gillespie_simulation_cuda.cuh"

#define SIM_COUNT 100
#define TIMESTEP_COUNT 1000
#define B 10
#define G 1
#define K_ON 0.1
#define K_OFF 0.9

int main(int argc, char **argv) {
    cudaSetDevice(1);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    int *h_done, *d_done;
    h_done = (int *) malloc(sizeof(int));
    cudaMalloc((void **) &d_done, sizeof(int));
    *h_done = 0;
    cudaMemset(d_done, 1, sizeof(int));

    int *d_concentrations, *d_states, *d_timestep_samples;
    float *d_times, *d_samples;
    size_t sim_info_size_i = SIM_COUNT * sizeof(int);
    size_t sim_info_size_f = SIM_COUNT * sizeof(float);
    size_t timestep_info_size = SIM_COUNT * TIMESTEP_COUNT * sizeof(int);
    
    cudaMalloc((void **) &d_concentrations, sim_info_size_i);
    cudaMalloc((void **) &d_states, sim_info_size_i);
    cudaMalloc((void **) &d_times, sim_info_size_f);
    cudaMalloc((void **) &d_samples, 2 * sim_info_size_f);
    cudaMalloc((void **) &d_timestep_samples, timestep_info_size);

    cudaMemset(d_concentrations, 0, sim_info_size_i);
    cudaMemset(d_states, 0, sim_info_size_i);
    cudaMemset(d_times, 0.0, sim_info_size_f);
    cudaMemset(d_timestep_samples, -1, timestep_info_size);

    while (*h_done == 0) {
        curandGenerateUniform(gen, d_samples, 2 * sim_info_size_f);
        call_simulation_step_kernel(d_samples, d_concentrations, d_states,
                                    d_times, SIM_COUNT, B, G, K_ON, K_OFF);

        call_timestep_update_kernel(d_concentrations, d_states, d_times,
                                    d_timestep_samples, SIM_COUNT,
                                    TIMESTEP_COUNT, d_done);

        cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_concentrations);
    cudaFree(d_states);
    cudaFree(d_times);
    cudaFree(d_samples);
    cudaFree(d_done);
    curandDestroyGenerator(gen);

    float *h_means, *d_means, *h_variances, *d_variances;
    size_t output_size = TIMESTEP_COUNT * sizeof(float);

    h_means = (float *) malloc(output_size);
    cudaMalloc((void **) &d_means, output_size);
    h_variances = (float *) malloc(output_size);
    cudaMalloc((void **) &d_variances, output_size);

    call_expected_means_kernel(d_timestep_samples, d_means, SIM_COUNT,
                               TIMESTEP_COUNT);

    call_expected_variances_kernel(d_timestep_samples, d_means, d_variances,
                                   SIM_COUNT, TIMESTEP_COUNT);

    cudaMemcpy(h_means, d_means, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, output_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < TIMESTEP_COUNT; i++)
        printf("Timestep %d: Mean = %f, Variance = %f\n", i, h_means[i],
                                                          h_variances[i]);

    cudaFree(d_means);
    cudaFree(d_variances);

    free(h_done);
    free(h_means);
    free(h_variances);
}
