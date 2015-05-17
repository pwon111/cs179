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

  size_t buffer_size = batch_size * (REVIEW_DIM + 1) * sizeof(float);
  size_t weights_size = REVIEW_DIM * sizeof(float);

  float *h_buff[N_STREAMS];
  float *d_buff[N_STREAMS];
  
  cudaStream_t s[N_STREAMS];

  cudaEvent_t memcpyEvents[N_STREAMS];
  cudaEvent_t memsetEvents[N_STREAMS];
  // For each stream, allocate GPU memory and pinned host memoryfor the output
  // and buffers, and initialize the stream
  for (int i = 0; i < N_STREAMS; i++) {
    gpuErrChk(cudaMallocHost((void **) &(h_buff[i]), buffer_size));
    gpuErrChk(cudaMalloc((void **) &(d_buff[i]), buffer_size));
    
    gpuErrChk(cudaStreamCreate(&s[i]));
    
    gpuErrChk(cudaEventCreate(&memcpyEvents[i]));
    gpuErrChk(cudaEventRecord(memcpyEvents[i]));

    gpuErrChk(cudaEventCreate(&memsetEvents[i]));
    gpuErrChk(cudaEventRecord(memsetEvents[i]));
  }

  float *h_weights = (float *) malloc(weights_size);
  gaussianFill(h_weights, REVIEW_DIM);
  
  float *d_weights, *d_spare_weights;
  cudaMalloc((void **) &d_weights, weights_size);
  cudaMalloc((void **) &d_spare_weights, weights_size);
  
  cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
  cudaMemset(d_spare_weights, 0.0, weights_size);

  float wrong = 0.0;

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0, relative_idx, stream_i = 0, batch_i = batch_size - 1;
  div_t which_stream;
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    // Get where we are relative to the start of the first stream's buffer
    relative_idx = review_idx % (N_STREAMS * batch_size);
    // Figure out which stream's buffer we're in, and at what position
    which_stream = div(relative_idx, batch_size);
    stream_i = which_stream.quot;
    batch_i = which_stream.rem;
    // Read in a review and transpose it so that the first batch_size elements
    // of the buffer are the first elements of all the review vectors, then the
    // second set of batch_size elements are the second elements, etc.
    gpuErrChk(cudaEventSynchronize(memcpyEvents[stream_i]));
    readLSAReview(review_str, h_buff[stream_i] + batch_i, batch_size);

    // If we've filled up a buffer, copy it to the GPU and call the kernel, then
    // transfer the output back
    if (batch_i == batch_size - 1) {
      gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                                cudaMemcpyHostToDevice, s[stream_i]));
      gpuErrChk(cudaEventRecord(memcpyEvents[stream_i]));
      
      gpuErrChk(cudaEventSynchronize(memsetEvents[stream_i]))
      wrong += cudaClassify(d_buff[stream_i], batch_size, STEP_SIZE, d_weights, d_spare_weights);
      
      gpuErrChk(cudaMemsetAsync(d_spare_weights, 0.0, weights_size,
                                s[stream_i]));
      gpuErrChk(cudaEventRecord(memsetEvents[stream_i]));
    }
  }

  // Get the remaining unassigned reviews and do a batch of size however many
  // there are left
  if (batch_i != batch_size - 1) {
    gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                              cudaMemcpyHostToDevice, s[stream_i]));
    gpuErrChk(cudaEventRecord(memcpyEvents[stream_i]));
    wrong += cudaClassify(d_buff[stream_i], batch_i + 1, STEP_SIZE, d_weights,
                          d_spare_weights);
  }

  printf("Total incorrect rate: %f\n", wrong / review_idx);

  free(h_weights);
  cudaFree(d_weights);
  cudaFree(d_spare_weights);

  // TODO: print out weights

  // Free the pinned output and host buffers, and destroy the streams
  for (int i = 0; i < N_STREAMS; i++) {
    gpuErrChk(cudaFreeHost(h_buff[i]));
    gpuErrChk(cudaFree(d_buff[i]));
    gpuErrChk(cudaStreamSynchronize(s[i]));
    gpuErrChk(cudaStreamDestroy(s[i]));
    gpuErrChk(cudaEventDestroy(memcpyEvents[i]));
    gpuErrChk(cudaEventDestroy(memsetEvents[i]));
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
