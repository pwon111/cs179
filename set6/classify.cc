#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"

using namespace std;

#define N_STREAMS 3
#define STEP_SIZE 2.2

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
  }

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
  // seed generator to 2015
  std::default_random_engine generator(2015);
  std::normal_distribution<float> distribution(0.0, 0.1);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[stride * component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
  // So that I'm not using the GPU driving my display
  gpuErrChk(cudaSetDevice(1));

  // Size in bytes of the input buffers and weight vector
  size_t buffer_size = batch_size * (REVIEW_DIM + 1) * sizeof(float);
  size_t weights_size = REVIEW_DIM * sizeof(float);

  // Device and host rrays of input buffers, one for each stream
  float *h_buff[N_STREAMS];
  float *d_buff[N_STREAMS];
  
  // Array of streams
  cudaStream_t s[N_STREAMS];

  // Array of memcpy cudaEvents used to prevent a stream from overwriting the
  // data buffer while it's being copied to the device
  cudaEvent_t memcpyEvents[N_STREAMS];

  // For each stream, allocate GPU memory and pinned host memory for the
  // buffers, initialize the stream, and ready each stream to read input
  for (int i = 0; i < N_STREAMS; i++) {
    gpuErrChk(cudaMallocHost((void **) &(h_buff[i]), buffer_size));
    gpuErrChk(cudaMalloc((void **) &(d_buff[i]), buffer_size));
    
    gpuErrChk(cudaStreamCreate(&s[i]));
    
    gpuErrChk(cudaEventCreate(&memcpyEvents[i]));
    gpuErrChk(cudaEventRecord(memcpyEvents[i]));
  }

  // Initialize the weights vector with random Gaussian noise
  float *h_weights = (float *) malloc(weights_size);
  gaussianFill(h_weights, REVIEW_DIM);
  
  // Copy the weights vector to the device
  float *d_weights;
  cudaMalloc((void **) &d_weights, weights_size);
  cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);

  // Counter for the number of misclassifications per batch, the sum across all
  // batches, and the number of batches executed
  float wrong = 0.0, total_wrong = 0.0;
  int batch_count = 0;

  // Main loop to process input lines (each line corresponds to a review)
  int review_idx = 0, relative_idx, stream_i = 0, batch_i = batch_size - 1;
  div_t which_stream;
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    // Get where we are relative to the start of the first stream's buffer
    relative_idx = review_idx % (N_STREAMS * batch_size);

    // Figure out which stream's buffer we're in, and at what position
    which_stream = div(relative_idx, batch_size);
    stream_i = which_stream.quot;
    batch_i = which_stream.rem;
    
    // Make sure this stream's buffer isn't still being copied to the device
    gpuErrChk(cudaEventSynchronize(memcpyEvents[stream_i]));

    // Read in a review and transpose it so that the first batch_size elements
    // of the buffer are the first elements of all the review vectors, then the
    // second set of batch_size elements are the second elements, etc. since
    // this maximizes coalescing as all batch_size threads per kernel will make
    // spacially consecutive accesses to get each element
    readLSAReview(review_str, h_buff[stream_i] + batch_i, batch_size);

    // If we've filled up a buffer, copy it to the GPU and call the kernel, then
    // transfer the output back
    if (batch_i == batch_size - 1) {
      // Copy the input buffer to the device and signal that the process is done
      gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                                cudaMemcpyHostToDevice, s[stream_i]));
      gpuErrChk(cudaEventRecord(memcpyEvents[stream_i]));

      // Call the kernel and get the number of misclassifications
      wrong = cudaClassify(d_buff[stream_i], batch_size, batch_size, STEP_SIZE,
                           d_weights);

      // Update the total number of misclassifications and the batch count
      total_wrong += wrong;
      batch_count++;
      printf("Batch %d, error rate: %f\n", batch_count, wrong / batch_size);
    }
  }

  // Get the remaining unassigned reviews and do a batch of size however many
  // there are left
  if (batch_i != batch_size - 1) {
    // Copy the input buffer to the device
    gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                              cudaMemcpyHostToDevice, s[stream_i]));

    // Call the kernel
    wrong = cudaClassify(d_buff[stream_i], batch_i + 1, batch_size, STEP_SIZE,
                         d_weights);

    // Update cumulative information
    total_wrong += wrong;
    batch_count++;
    printf("Batch %d, error rate: %f\n", batch_count, wrong / (batch_i + 1));
  }

  printf("Overall error rate: %f\n", total_wrong / review_idx);

  // Copy the weights back to the device and print them
  cudaMemcpy(h_weights, d_weights, weights_size, cudaMemcpyDeviceToHost);
  printf("Final weights are:\n[");
  for (int i = 0; i < REVIEW_DIM - 1; i++)
    printf("%f, ", h_weights[i]);
  printf("%f]\n", h_weights[REVIEW_DIM - 1]);

  // Free the weight vectors
  free(h_weights);
  cudaFree(d_weights);

  // Free all the other dynamically-allocated memory, and destroy the streams
  // and events used
  for (int i = 0; i < N_STREAMS; i++) {
    gpuErrChk(cudaFreeHost(h_buff[i]));
    gpuErrChk(cudaFree(d_buff[i]));
    gpuErrChk(cudaStreamSynchronize(s[i]));
    gpuErrChk(cudaStreamDestroy(s[i]));
    gpuErrChk(cudaEventDestroy(memcpyEvents[i]));
  }
}

int main(int argc, char** argv) {
  int batch_size = 2048;

  if (argc == 1) {
    classify(cin, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
  }
}
