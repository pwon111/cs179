/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


#include <cuda_runtime.h>
#include <algorithm>

#include "Cuda1DFDWave_cuda.cuh"






int main(int argc, char* argv[]) {
    
    
  if (argc < 3){
      printf("Usage: (threads per block) (max number of blocks)\n");
      exit(-1);
  }
  

  // make sure output directory exists
  std::ifstream test("output");
  if ((bool)test == false) {
    printf("Cannot find output directory, please make it (\"mkdir output\")\n");
    exit(1);
  }
  
  /* Additional parameters for the assignment */
  
  const bool CUDATEST_WRITE_ENABLED = true;   //enable writing files
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);
  
  

  // Parameters regarding our simulation
  const size_t numberOfIntervals = 1e5;
  const size_t numberOfTimesteps = 1e5;
  const size_t numberOfOutputFiles = 3;

  //Parameters regarding our initial wave
  const float courant = 1.0;
  const float omega0 = 10;
  const float omega1 = 100;

  // derived
  const size_t numberOfNodes = numberOfIntervals + 1;
  const float courantSquared = courant * courant;
  const float dx = 1./numberOfIntervals;
  const float dt = courant * dx;




  /************************* CPU Implementation *****************************/


  // make 3 copies of the domain for old, current, and new displacements
  float ** data = new float*[3];
  for (unsigned int i = 0; i < 3; ++i) {
    // make a copy
    data[i] = new float[numberOfNodes];
    // fill it with zeros
    std::fill(&data[i][0], &data[i][numberOfNodes], 0);
  }

  for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
       ++timestepIndex) {
    if (timestepIndex % (numberOfTimesteps / 10) == 0) {
      printf("Processing timestep %8zu (%5.1f%%)\n",
             timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
    }

    // nickname displacements
    const float * oldDisplacements =     data[(timestepIndex - 1) % 3];
    const float * currentDisplacements = data[(timestepIndex + 0) % 3];
    float * newDisplacements =           data[(timestepIndex + 1) % 3];
    
    for (unsigned int a = 1; a <= numberOfNodes - 2; ++a){
        newDisplacements[a] = 
                2*currentDisplacements[a] - oldDisplacements[a]
                + courantSquared * (currentDisplacements[a+1]
                        - 2*currentDisplacements[a] 
                        + currentDisplacements[a-1]);
    }


    // apply wave boundary condition on the left side, specified above
    const float t = timestepIndex * dt;
    if (omega0 * t < 2 * M_PI) {
      newDisplacements[0] = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
    } else {
      newDisplacements[0] = 0;
    }

    // apply y(t) = 0 at the rightmost position
    newDisplacements[numberOfNodes - 1] = 0;


    // enable this is you're having troubles with instabilities
#if 0
    // check health of the new displacements
    for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
      if (std::isfinite(newDisplacements[nodeIndex]) == false ||
          std::abs(newDisplacements[nodeIndex]) > 2) {
        printf("Error: bad displacement on timestep %zu, node index %zu: "
               "%10.4lf\n", timestepIndex, nodeIndex,
               newDisplacements[nodeIndex]);
      }
    }
#endif

    // if we should write an output file
    if (numberOfOutputFiles > 0 &&
        (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles) == 0) {
      printf("writing an output file\n");
      // make a filename
      char filename[500];
      sprintf(filename, "output/CPU_data_%08zu.dat", timestepIndex);
      // write output file
      FILE* file = fopen(filename, "w");
      for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
        fprintf(file, "%e,%e\n", nodeIndex * dx,
                newDisplacements[nodeIndex]);
      }
      fclose(file);
    }
  }
  


  /************************* GPU Implementation *****************************/

  {
  
  
    const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
                numberOfNodes/float(threadsPerBlock)));
  
    //Space on the CPU to copy file data back from GPU
    float *file_output = new float[numberOfNodes];

    // Create pointers for timesteps t and t - 1, as well as one to swap them,
    float *d_old, *d_current, *d_temp;
    // Figure out how much space each timestep's values take up
    size_t nodes_size = numberOfNodes * sizeof(float);

    // Allocate space for the two timesteps needed to calculate step t + 1
    cudaMalloc((void **) &d_old, nodes_size);
    cudaMalloc((void **) &d_current, nodes_size);

    // Set them to all 0s
    cudaMemset(d_old, 0.0, nodes_size);
    cudaMemset(d_current, 0.0, nodes_size);
    
    // Looping through all times t = 0, ..., t_max
    for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
            ++timestepIndex) {
        
        if (timestepIndex % (numberOfTimesteps / 10) == 0) {
            printf("Processing timestep %8zu (%5.1f%%)\n",
                 timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
        }

        // Update nodes 1 through length - 2  (where indexing starts from 0)
        call_update_nodes_kernel(blocks, threadsPerBlock, d_old, d_current,
                                 numberOfNodes, courantSquared);

        // The kernel writes timestep t + 1 into the memory used for t - 1, so
        // swap the old and current pointers so that they're in the right order
        d_temp = d_old;
        d_old = d_current;
        d_current = d_temp;

        //Left boundary condition on the CPU - a sum of sine waves
        const float t = timestepIndex * dt;
        float left_boundary_value;
        if (omega0 * t < 2 * M_PI) {
            left_boundary_value = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
        } else {
            left_boundary_value = 0;
        }

        // Update the boundary conditions, with the one on the right always 0
        call_update_boundaries_kernel(d_current, left_boundary_value,
                                      numberOfNodes);
        
        // Check if we need to write a file
        if (CUDATEST_WRITE_ENABLED == true && numberOfOutputFiles > 0 &&
                (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles) 
                == 0) {
            
            // Copy the current output from the GPU back to the host
            cudaMemcpy(file_output, d_current, nodes_size,
                       cudaMemcpyDeviceToHost);

            printf("writing an output file\n");
            // make a filename
            char filename[500];
            sprintf(filename, "output/GPU_data_%08zu.dat", timestepIndex);
            // write output file
            FILE* file = fopen(filename, "w");
            for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
                fprintf(file, "%e,%e\n", nodeIndex * dx,
                        file_output[nodeIndex]);
            }
            fclose(file);
        }
        
    }
    
    // Free memory allocated on the GPU
    cudaFree(d_old);
    cudaFree(d_current);
}
  
  printf("You can now turn the output files into pictures by running "
         "\"python makePlots.py\". It should produce png files in the output "
         "directory. (You'll need to have gnuplot in order to produce the "
         "files - either install it, or run on the remote machines.)\n");

  return 0;
}
