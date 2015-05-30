/**
 * Header file for the kernels and kernel-calling code used for the Gillespie
 * Algorithm simulations.
 */

void call_simulation_step_kernel(float *samples, int *concentrations, int *states,
                                 float *times, int sim_count, float b, float g,
                                 float K_on, float K_off);

void call_timestep_update_kernel(int *concentrations, int *states, float *times,
                                 int *timestep_samples, int sim_count,
                                 int timestep_count, float total_time,
                                 int *done);

void call_expected_means_kernel(int *timestep_samples, float *means,
                                int sim_count, int timestep_count);

void call_expected_variances_kernel(int *timestep_samples, float *means,
                                    float *variances, int sim_count,
                                    int timestep_count);