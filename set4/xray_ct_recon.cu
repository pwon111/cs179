
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#include <stdlib.h>

#define PI 3.14159265358979


texture<float, 2, cudaReadModeElementType> texreference;


/**
 * Takes a list of consecutive complex Fourier transformed sinogram readings and
 * applies a ramp filter to each
 */
__global__ void cudaRampFilterKernel(cufftComplex *sinogram_cmplx,
                                     int sinogram_width, int length)
{
    int relative;
    float factor;
    // Each thread handles one individual sample (assuming there are enough)
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x; n < length;
         n += blockDim.x * gridDim.x)
    {
        // Figure out which frequency we're at
        relative = n % sinogram_width;
        // Scale each frequency relative to the maximum one
        factor = (float) relative / (float) sinogram_width;
        sinogram_cmplx[n].x *= factor;
        sinogram_cmplx[n].y *= factor;
    }
}

/**
 * Takes a list of filtered, real-valued sinogram readings and constructs a
 * backprojected image from them
 */
__global__ void cudaBackprojectionKernel(float *output_dev, int width,
                                         int height, int sinogram_width,
                                         int nAngles)
{
    float sum, xi, yi, q, m, d, theta, step = PI / nAngles;
    float d_adjust = (float) sinogram_width / 2.0;
    int x0, y0, width_adjust = width / 2, height_adjust = height / 2;
    int half_samples = nAngles / 2;
    bool pi2 = half_samples * 2 == nAngles; // Only true if nAngles is even
    // Each thread handles a single pixel (assuming there are enough threads)
    for (unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
         n < width * height; n += blockDim.x * gridDim.x)
    {
        // Calculate y_0 in pixel coordinates
        y0 = n / width;
        // Use y_0 to get x_0 in pixel coordinates, then adjust to geometric
        x0 = n - y0 * width - width_adjust;
        // Take care of the theta = 0 case
        sum = tex2D(texreference, x0 + d_adjust, 0);
        // Invert and adjust y to geometric coordinates
        y0 = height - (y0 + 1) - height_adjust;
        // Loop through each value of theta and calculate the pixel's value
        for (int sample = 1; sample < nAngles; sample++) {
            theta = sample * step;
            // Take care of the theta = pi/2 case
            if (pi2 && sample == half_samples) {
                sum += tex2D(texreference, y0 + d_adjust, sample);
                continue;
            }
            // Calculate m and q
            q = tanf(theta);
            m = -1.0 / q;

            // Calculate x_i and y_i
            xi = (y0 - m * x0) / (q - m);
            yi = q * xi;

            // Calculate distance and direction d from the sinogram's center
            d = sqrtf(xi * xi + yi * yi);
            if ((q > 0 && xi < 0) || (q < 0 && xi > 0))
                d *= -1;
            // Adjust so it's a valid array coordinate
            d += d_adjust;

            // Get the corresponding sinogram value from texture memory
            sum += tex2D(texreference, d, sample);
        }
        // Set the pixel's value
        output_dev[n] = sum;
    }
}


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}




int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    float *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    int nSamples = sinogram_width * nAngles;
    sinogram_host = (float *) malloc(sizeof(float) * nSamples);

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile, "%f", &sinogram_host[i]);
        // fscanf(dataFile,"%f",&sinogram_host[i].x);
        // sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    // Allocate GPU memory for and copy sinogram data
    cudaMalloc((void **) &dev_sinogram_float, sizeof(float) * nSamples);
    cudaMemcpy(dev_sinogram_float, sinogram_host, sizeof(float) * nSamples,
               cudaMemcpyHostToDevice);

    // Set up parameters for cuFFT batch mode
    int batch = nAngles; // One batch per sinogram angle
    int rank = 1;        // Only one dimension

    int *n = &sinogram_width; // Length of said dimension

    int idist = sinogram_width;           // Distance between real inputs
    int odist = sinogram_width / 2 + 1;   // Distance between complex outputs
    int nSamples_cmplx = nAngles * odist; // Number of samples in complex output

    int istride = 1; // Distance between real input samples
    int ostride = 1; // Distance between complex output samples

    // Allocate GPU memory for complex output data
    cudaMalloc((void **) &dev_sinogram_cmplx,
               sizeof(cufftComplex) * nSamples_cmplx);

    // Create a cuFFT plan to batch transform the reading of the sinogram at
    // each individual angle, and execute it
    cufftHandle plan;
    cufftPlanMany(&plan, rank, n, &idist, istride, idist, &odist, ostride,
                  odist, CUFFT_R2C, batch);
    cufftExecR2C(plan, dev_sinogram_float, dev_sinogram_cmplx);

    // Call the ramp filter kernel on the complex transformed sinogram data
    cudaRampFilterKernel<<<nBlocks, threadsPerBlock>>>(dev_sinogram_cmplx,
                                                       odist, nSamples_cmplx);

    // Transform the filtered complex data back to real values
    cufftPlanMany(&plan, rank, n, &odist, ostride, odist, &idist, istride,
                  idist, CUFFT_C2R, batch);
    cufftExecC2R(plan, dev_sinogram_cmplx, dev_sinogram_float);

    // Allocate memory for and copy the filtered sinogram data to a
    // two-dimensional array
    cudaArray *tex_sinogram;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
    cudaMallocArray(&tex_sinogram, &channel, sinogram_width, nAngles);
    cudaMemcpyToArray(tex_sinogram, 0, 0, dev_sinogram_float,
                      sizeof(float) * nSamples, cudaMemcpyDeviceToDevice);

    // Set up the texture reference
    texreference.addressMode[0] = cudaAddressModeClamp;
    texreference.addressMode[1] = cudaAddressModeClamp;
    texreference.filterMode = cudaFilterModePoint;
    texreference.normalized = false;

    // Bind the array of sinogram data to texture memory
    cudaBindTextureToArray(texreference, tex_sinogram);

    // Allocate GPU memory for the backprojection output
    cudaMalloc((void **) &output_dev, size_result);

    // Call the backprojection kernel
    cudaBackprojectionKernel<<<nBlocks, threadsPerBlock>>>(output_dev, width,
                                                           height,
                                                           sinogram_width,
                                                           nAngles);

    // Copy the backprojection output back to the host
    cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost);;

    // Unbind the texture memory and free all the GPU memory we used
    cudaUnbindTexture(texreference);
    cudaFree(output_dev);
    cudaFree(dev_sinogram_float);
    cudaFree(dev_sinogram_cmplx);
    cudaFreeArray(tex_sinogram);

    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



