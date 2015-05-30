#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>

#include "gillespie_simulation_cuda.cuh"

/**
 * This kernel advances each simulation by one event. It does this by modeling
 * each simulation as independent Poisson processes and using the Gillespie
 * algorithm to randomly choose an event and a timespan until it happens,
 * updating the arrays used to keep track of each simulation's state.
 */
__global__
void simulation_step_kernel(float *samples, int *concentrations, int *states,
                            float *times, int sim_count, float b, float g,
                            float K_on, float K_off) {
    // Each thread handles one simulation
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; n < sim_count;
         n += blockDim.x * gridDim.x) {
        int state = states[n];
        float event_sample = samples[n];            // PRNG value for next event
        float time_sample = samples[sim_count + n]; // PRNG value for timespan
        float propensity_sum;

        // Handle the case where the chemical-producing system is off (I would
        // have just treated these as booleans, but I have a third state of -1
        // representing finished simulations)
        if (state == 0) {
            // Calculate the sum of the propensities
            propensity_sum = K_on + concentrations[n] * g;
            // A random number between 0 and 1 has a probability p of falling in
            // the range [0, p), so we can use this to give the system the
            // correct chance of turning on
            if (event_sample < K_on / propensity_sum)
                states[n] = 1;
            // If it doesn't turn on, it just consumes some of the chemical
            else
                concentrations[n]--;
        
            // Randomly choose a timespan from an exponential distribution
            times[n] -= logf(time_sample) / propensity_sum;
        }
        // Handle the case where the chemical-producing system is on
        else if (state == 1) {
            // Same process of above, but since there are three choices we need
            // to check if our random value falls in the range [0, p_0), for
            // event 0, then in [p_0, p_0 + p_1) for event 1, etc. 
            propensity_sum = K_off + b + concentrations[n] * g;
            float first_bin = K_off / propensity_sum;
            if (event_sample < first_bin)
                states[n] = 0;
            else if (event_sample < first_bin + b / propensity_sum) {
                concentrations[n]++;
            }
            else
                concentrations[n]--;
        
            // Randomly choose a timespan from an exponential distribution
            times[n] -= logf(time_sample) / propensity_sum;
        }
    }
}

/**
 * This kernel updates the array of specific-timestamp samples for each
 * simulation. It is called immediately following the simulation-advancing
 * kernel, and updates the value for a specific simulation at a specific
 * timestep with the first known concentration of that simulation after said
 * timestep, if the simulation has reached that point in time.
 */
__global__
void timestep_update_kernel(int *concentrations, int *states, float *times,
                            int *timestep_samples, int sim_count,
                            int timestep_count, float total_time, int *done) {
    // Each thread handles one simulation
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; n < sim_count;
         n += blockDim.x * gridDim.x) {
        // If the simulation is done running, don't bother with it
        if (states[n] == -1)
            break;

        // Represents the timestep value's index in the
        // sim_count x timestep_count array
        unsigned int actual_i;
        // Scale our timestep checking so that our timesteps evenly divide up
        // the specified total_time interval
        float time_scale = timestep_count / total_time;
        // Loop through the simulation's timestep records, filling in the ones
        // that we haven't yet, until we get to a timestep past the current time
        for (unsigned int i = 0; i < timestep_count; i++) {
            actual_i = i * sim_count + n;
            // If we don't have a record for this timestep yet:
            if (timestep_samples[actual_i] == -1) {
                // Check if the simulation has progressed past it
                if (times[n] >= i * time_scale) {
                    // Then update the value
                    timestep_samples[actual_i] = concentrations[n];
                    // If it was the last one, mark it as done and break
                    if (i + 1 == timestep_count) {
                        states[n] = -1;
                        break;
                    }
                    // Otherwise, we need at least one more iteration, so update
                    // the done boolean to reflect this
                    atomicExch(done, 0);
                }
                // Otherwise, we're done updating and still need at least one
                // more iteration, so update the done boolean to reflect this
                else {
                    atomicExch(done, 0);
                    break;
                }
            }
        }
    }
}

/**
 * This kernel uses a reduction to calculate the expected concentration across
 * all simulations at each timestep, by summing them all together and then
 * dividing by the number of simulations at the very end.
 */
__global__
void expected_means_kernel(int *timestep_samples, float *means, int sim_count,
                           int timestep_count) {
    // Each block handles a timestep, so it can have its own shared memory
    extern __shared__ float values[];

    // Since reductions are based on repeatedly halving the number of values to
    // compute, we have to be careful not to assume there are always an even
    // number to halve
    int odd_sim_count = sim_count & 1;
    // There are ceil(sim_count / 2) slots after the first pass of the reduction
    unsigned int slots = (sim_count + 1) / 2;
    for (unsigned int j = blockIdx.x; j < timestep_count; j += gridDim.x) {
        unsigned int offset = j * sim_count;
        // We can copy the values we need to do the rest of the reduction to
        // shared memory and perform its first pass while we're at it
        for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x) {
            if (odd_sim_count && i == slots - 1)
                values[i] = (float) timestep_samples[offset + i];
            else
                values[i] = (float) (timestep_samples[offset + i] +
                                     timestep_samples[offset + i + slots]);
        }

        odd_sim_count = slots & 1;
        // While there are more than one values left to be reduced, iterate
        for (slots = (slots + 1) / 2; slots > 1; slots = (slots + 1) / 2) {
            // Make sure all the changes from the last iteration are done
            __syncthreads();
            // If the value has a matching one, at index i + slots, combine them
            // (i.e. do an iteration of the reduction process)
            for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x)
                if (!odd_sim_count || i != slots - 1)
                    values[i] = values[i] + values[i + slots];
            // This has to be a post-update since it represents the parity of
            // the number of values left to reduce after this pass
            odd_sim_count = slots & 1;
        }

        // Divide the total sum by the simulation count and store it in the
        // index in the output array corresponding to this timestep
        if (threadIdx.x == 0)
            means[j] = values[0] / sim_count;
    }
}

/**
 * This kernel uses a reduction to calculate the expected variance in
 * concentration across all simulations at each timestep, by summing them all
 * together and then dividing by the number of simulations at the very end.
 */
__global__
void expected_variances_kernel(int *timestep_samples, float *means,
                               float *variances, int sim_count,
                               int timestep_count) {
    // Each block handles a timestep, so it can have its own shared memory
    extern __shared__ float values[];


    for (unsigned int j = blockIdx.x; j < timestep_count; j += gridDim.x) {
        unsigned int offset = j * sim_count;
        // Compute the squared difference between the simulation's concentration
        // and the expected concentration for its timestep, and then copy it to
        // shared memory
        float difference;
        for (unsigned int i = threadIdx.x; i < sim_count; i += blockDim.x) {
            difference = (float) timestep_samples[offset + i] - means[j];
            values[i] = difference * difference;
        }

        int odd_sim_count = sim_count & 1;
        // While there are more than one values left to be reduced, iterate
        for (unsigned int slots = (sim_count + 1) / 2; slots > 1;
             slots = (slots + 1) / 2) {
            __syncthreads();
            // If the value has a matching one, at index i + slots, combine them
            // (i.e. do an iteration of the reduction process)
            for (unsigned int i = threadIdx.x; i < slots; i += blockDim.x)
                if (!odd_sim_count || i != slots - 1)
                    values[i] = values[i] + values[i + slots];
            // This has to be a post-update since it represents the parity of
            // the number of values left to reduce after this pass
            odd_sim_count = slots & 1;
        }

        // Divide the total sum by the simulation count and store it in the
        // index in the output array corresponding to this timestep
        if (threadIdx.x == 0)
            variances[j] = values[0] / sim_count;
    }
}

/**
 * This function calls the simulation step kernel with a single block and each
 * thread handling its own simulation, although the kernel itself works with
 * arbitrarily many threads in aribitrarily many blocks.
 */
void call_simulation_step_kernel(float *samples, int *concentrations, int *states,
                                 float *times, int sim_count, float b, float g,
                                 float K_on, float K_off) {
    simulation_step_kernel<<<1, sim_count>>>(samples, concentrations, states,
                                             times, sim_count, b, g, K_on,
                                             K_off);
}

/**
 * This function calls the timestep update kernel with a single block and each
 * thread handling its own simulation, although the kernel itself works with
 * arbitrarily many threads in aribitrarily many blocks.
 */
void call_timestep_update_kernel(int *concentrations, int *states, float *times,
                                 int *timestep_samples, int sim_count,
                                 int timestep_count, float total_time,
                                 int *done) {
    // Set the done boolean to 1 so that if none of the threads update it, we
    // know all the simulations have finished
    cudaMemset(done, 1, sizeof(int));
    timestep_update_kernel<<<1, sim_count>>>(concentrations, states, times,
                                             timestep_samples, sim_count,
                                             timestep_count, total_time, done);
}

/**
 * This function calls the expected concentration reduction kernel with a block
 * of ((sim_count + 1) / 2) threads handling each timestep; we only need
 * this many threads since we do the first part of the reduction while copying
 * the necessary values to shared memory. This works with arbitrarily many
 * blocks and threads (excluding cases where not enough shared memory can be
 * allocated), but the kernel is designed around the values used below.
 */
void call_expected_means_kernel(int *timestep_samples, float *means,
                                int sim_count, int timestep_count) {
    int block_size = ((sim_count + 1) / 2);
    expected_means_kernel<<<timestep_count,
                            block_size,
                            block_size * sizeof(float)>>>
                         (timestep_samples, means, sim_count, timestep_count);
}

/**
 * This function calls the expected variance reduction kernel with a block of
 * sim_count threads handling each timestep; unlike the mean reduction kernel,
 * we use sim_count threads here since it's easier to have each simulation's
 * squared deviation from the mean calculated in parallel and copied to shared
 * memory rather than having one thread calculate the value for two in order to
 * perform the first reduction iteration at the same time.
 */
void call_expected_variances_kernel(int *timestep_samples, float *means,
                                    float *variances, int sim_count,
                                    int timestep_count) {
    expected_variances_kernel<<<timestep_count,
                                sim_count,
                                sim_count * sizeof(float)>>>
                             (timestep_samples, means, variances, sim_count,
                              timestep_count);
}
