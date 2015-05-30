#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <curand.h>

#include "gillespie_simulation_cuda.cuh"

#define SIM_COUNT 1024      /* Number of simulations to run in parallel */
#define TIMESTEP_COUNT 1000 /* Number of timesteps to record */
#define TOTAL_TIME 100      /* Total time (in seconds) to run each simulation */
#define B 10        /* Propensity to produce the chemical */
#define G 1         /* Factor in the propensity to consume the chemical */
#define K_ON 0.1    /* Propensity of the system to turn on */
#define K_OFF 0.9   /* Propensity of the system to turn off */

/**
 * This program uses the Gillespie algorithm to simulate SIM_COUNT runs of a
 * chemical-producing system for TOTAL_TIME seconds, in parallel. It records
 * the concentration of each system at each of TIMESTEP_COUNT evenly-spaced
 * points in time, choosing the concentration immediately following the closest
 * event to happen after said timestep in each simulation. It then calculates
 * and prints the expected concentration and variance at each timestep.
 */
int main(int argc, char **argv) {
    // Use the GPU that's not driving my display
    cudaSetDevice(1);

    // Create the PRNG
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Allocate host and device memory for a boolean indicating whether at least
    // one simulation is still running, setting the host to indicate that one is
    int *h_done, *d_done;
    h_done = (int *) malloc(sizeof(int));
    cudaMalloc((void **) &d_done, sizeof(int));
    *h_done = 0;
    cudaMemset(d_done, 1, sizeof(int));

    // Device memory arrays storing the current concentration in each
    // simulation, the state of each simulation (on = 1, off = 0, done = -1),
    // the values at each timestep produced by the resampling kernel, the
    // current elapsed time of each simulation, and the PRNG output needed to
    // advance to the next step of each simulation, as well as size_t variables
    // to allocate, set, copy, etc. all of them
    int *d_concentrations, *d_states, *d_timestep_samples;
    float *d_times, *d_rng_samples;
    size_t sim_info_size_i = SIM_COUNT * sizeof(int);
    size_t sim_info_size_f = SIM_COUNT * sizeof(float);
    size_t timestep_info_size = SIM_COUNT * TIMESTEP_COUNT * sizeof(int);
    
    // Allocate all the GPU memory needed for running the simulations themselves
    cudaMalloc((void **) &d_concentrations, sim_info_size_i);
    cudaMalloc((void **) &d_states, sim_info_size_i);
    cudaMalloc((void **) &d_times, sim_info_size_f);
    cudaMalloc((void **) &d_rng_samples, 2 * sim_info_size_f);
    cudaMalloc((void **) &d_timestep_samples, timestep_info_size);

    // Set all concentrations to start at 0, all simulations to start in the off
    // (0) state, all elapsed times to start at 0.0, and all the timestep
    // concentration samples to -1 so we can tell if it hasn't been updated yet
    cudaMemset(d_concentrations, 0, sim_info_size_i);
    cudaMemset(d_states, 0, sim_info_size_i);
    cudaMemset(d_times, 0.0, sim_info_size_f);
    cudaMemset(d_timestep_samples, -1, timestep_info_size);

    // The done boolean gets set to 0 by every thread whose simulation is still
    // running, so we just keep calling the simulation and timestep sample
    // kernels until all simulations are done.
    while (!(*h_done)) {
        // Generate a set of random numbers in [0, 1.0) for both choosing the
        // next event in each simulation, and how long it takes to happen
        curandGenerateUniform(gen, d_rng_samples, 2 * SIM_COUNT);

        // Call the kernel to advance each simulation based on the predefined
        // propensities and the just-generated random values, using the
        // Gillespie algorithm
        call_simulation_step_kernel(d_rng_samples, d_concentrations, d_states,
                                    d_times, SIM_COUNT, B, G, K_ON, K_OFF);

        // Call the kernel to update the concentration values for each
        // simulation in a set array of timesteps, sampled after the closest
        // event in the simulation to happen after each step
        call_timestep_update_kernel(d_concentrations, d_states, d_times,
                                    d_timestep_samples, SIM_COUNT,
                                    TIMESTEP_COUNT, TOTAL_TIME, d_done);

        // Copy the done boolean to the host so we can check it
        cudaMemcpy(h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Free all the device memory we no longer need and destroy the PRNG
    cudaFree(d_concentrations);
    cudaFree(d_states);
    cudaFree(d_times);
    cudaFree(d_rng_samples);
    cudaFree(d_done);
    curandDestroyGenerator(gen);

    // Allocate host and device arrays to store the concentration mean and
    // variance at each timestep
    float *h_means, *d_means, *h_variances, *d_variances;
    size_t output_size = TIMESTEP_COUNT * sizeof(float);

    h_means = (float *) malloc(output_size);
    cudaMalloc((void **) &d_means, output_size);
    h_variances = (float *) malloc(output_size);
    cudaMalloc((void **) &d_variances, output_size);

    // Calculate the expected (mean) concentration for each timestep
    call_expected_means_kernel(d_timestep_samples, d_means, SIM_COUNT,
                               TIMESTEP_COUNT);

    // Calculate the expected variance for each timestep
    call_expected_variances_kernel(d_timestep_samples, d_means, d_variances,
                                   SIM_COUNT, TIMESTEP_COUNT);

    // Copy the results to the host and print them out
    cudaMemcpy(h_means, d_means, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, output_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < TIMESTEP_COUNT; i++)
        printf("Timestep %d: Mean = %f, Variance = %f\n", i, h_means[i],
                                                          h_variances[i]);

    // Free the remaining device memory
    cudaFree(d_means);
    cudaFree(d_variances);
    cudaFree(d_timestep_samples);

    // Free the remaining host memory
    free(h_done);
    free(h_means);
    free(h_variances);
}
