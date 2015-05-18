#ifndef CUDA_CLASSIFY_CUH
#define CUDA_CLASSIFY_CUH

#define REVIEW_DIM 50

float cudaClassify(float *data, int batch_size, int data_stride,
                   float step_size, float *weights);

#endif
