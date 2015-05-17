#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

using namespace std;

#define N_STREAMS 4

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
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
//
// NOTE: I changed this function so that it takes a stride, so the input can be
// transposed as it's read in so that the accesses in the kernel are coalesced
void readLSAReview(string review_str, float *output, int stride) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[component_idx * stride] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
  int review_idx_start;
  int batch_size;
  int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
// TODO: Call with cudaStreamAddCallback (after completing D->H memcpy)
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
  printerArg *arg = static_cast<printerArg *>(userData);

  for (int i=0; i < arg->batch_size; i++) {
    printf("%d: %d\n", 
       arg->review_idx_start + i, 
       arg->cluster_assignments[i]);
  }

  delete arg;
}

void cluster(istream& in_stream, int k, int batch_size) {
  // So that I'm not using the GPU driving my display
  // gpuErrChk(cudaSetDevice(1));

  // cluster centers
  float *d_clusters;

  // how many points lie in each cluster
  int *d_cluster_counts;

  // allocate memory for cluster centers and counts
  gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

  // randomly initialize cluster centers
  float *clusters = new float[k * REVIEW_DIM];
  gaussianFill(clusters, k * REVIEW_DIM);
  gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
               cudaMemcpyHostToDevice));

  // initialize cluster counts to 0
  gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));

  // Sizes of the output and of each buffer
  size_t output_size = sizeof(int) * batch_size;
  size_t buffer_size = sizeof(float) * batch_size * REVIEW_DIM;
  // Set up device and host lists of output pointers, one for each stream
  int *h_output[N_STREAMS];
  int *d_output[N_STREAMS];
  // Set up device and host lists of buffer pointers, one for each stream
  float *h_buff[N_STREAMS];
  float *d_buff[N_STREAMS];
  // Set up list of streams
  cudaStream_t s[N_STREAMS];
  // For each stream, allocate GPU memory and pinned host memoryfor the output
  // and buffers, and initialize the stream
  for (int i = 0; i < N_STREAMS; i++) {
    gpuErrChk(cudaMallocHost((void **) &(h_output[i]), output_size));
    gpuErrChk(cudaMalloc((void **) &(d_output[i]), output_size));
    gpuErrChk(cudaMallocHost((void **) &(h_buff[i]), buffer_size));
    gpuErrChk(cudaMalloc((void **) &(d_buff[i]), buffer_size));
    gpuErrChk(cudaStreamCreate(&s[i]));
  }

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
    readLSAReview(review_str, h_buff[stream_i] + batch_i, batch_size);

    // If we've filled up a buffer, copy it to the GPU and call the kernel, then
    // transfer the output back
    if (batch_i == batch_size - 1) {
      gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                                cudaMemcpyHostToDevice, s[stream_i]));
      cudaCluster(d_clusters, d_cluster_counts, k, d_buff[stream_i],
                  d_output[stream_i], batch_size, s[stream_i]);
      gpuErrChk(cudaMemcpyAsync(h_output[stream_i], d_output[stream_i],
                                output_size,
                                cudaMemcpyDeviceToHost, s[stream_i]));
      
      // Create a printerArg struct and attach a callback to print it to the
      // stream
      // printerArg *arg = new printerArg;
      // arg->batch_size = batch_size;
      // arg->review_idx_start = review_idx - (batch_size - 1);
      // arg->cluster_assignments = h_output[stream_i];
      // cudaStreamAddCallback(s[stream_i], printerCallback,
      //                       (void *) arg, 0);
    }
  }

  // Get the remaining unassigned reviews and do a batch of size however many
  // there are left
  if (batch_i != batch_size - 1) {
    gpuErrChk(cudaMemcpyAsync(d_buff[stream_i], h_buff[stream_i], buffer_size,
                              cudaMemcpyHostToDevice, s[stream_i]));
    cudaCluster(d_clusters, d_cluster_counts, k, d_buff[stream_i],
                d_output[stream_i], batch_i + 1, s[stream_i]);
    gpuErrChk(cudaMemcpyAsync(h_output[stream_i], d_output[stream_i],
                              output_size, cudaMemcpyDeviceToHost,
                              s[stream_i]));
    
    // Create a printerArg struct and attach a callback to print it to the
    // stream
    printerArg *arg = new printerArg;
    arg->batch_size = batch_i;
    arg->review_idx_start = review_idx - (batch_i - 1);
    arg->cluster_assignments = h_output[stream_i];
    cudaStreamAddCallback(s[stream_i], printerCallback,
                          (void *) arg, 0);
  }

  // wait for everything to end on GPU before final summary
  gpuErrChk(cudaDeviceSynchronize());

  // retrieve final cluster locations and counts
  int *cluster_counts = new int[k];
  gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
               cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
               cudaMemcpyDeviceToHost));

  // print cluster summaries
  for (int i=0; i < k; i++) {
    printf("Cluster %d, population %d\n", i, cluster_counts[i]);
    printf("[");
    for (int j=0; j < REVIEW_DIM; j++) {
      printf("%.4e,", clusters[i * REVIEW_DIM + j]);
    }
    printf("]\n\n");
  }

  // free cluster data
  gpuErrChk(cudaFree(d_clusters));
  gpuErrChk(cudaFree(d_cluster_counts));
  delete[] cluster_counts;
  delete[] clusters;

  // Free the pinned output and host buffers, and destroy the streams
  for (int i = 0; i < N_STREAMS; i++) {
    cudaFreeHost(h_output[i]);
    gpuErrChk(cudaFree(d_output[i]));
    cudaFreeHost(h_buff[i]);
    gpuErrChk(cudaFree(d_buff[i]));
    cudaStreamSynchronize(s[i]);
    cudaStreamDestroy(s[i]);
  }
}

int main(int argc, char** argv) {
  int k = 50;
  int batch_size = 2048;

  if (argc == 1) {
    cluster(cin, k, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    cluster(buffer, k, batch_size);
  }
}
