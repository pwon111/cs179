#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <png.h>

#include "entities.h"
#include "raytrace_cuda.cuh"

#define RECURSION_DEPTH 1

#define MAX_INTENSITY 255 // Maximum color intensity
#define N_INDEX 1.52f // Refraction index of glass

#define FD 0.05
#define FX 0.035
#define XRES 1920
#define YRES 1080

#define SPHERE_COUNT 3
#define PLANE_COUNT 1
#define LIGHT_COUNT 1

/* Utility function to normalize a 3-float vector on the host. */
void h_normalize(float *v) {
    float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; i++)
        v[i] /= length;
}

/**
 * This function takes in the float array representing the red, green, and blue
 * intensities of each pixel on the screen, and writes it to a PNG image using
 * libpng.
 */
int save_png(float *screen, char *path) {
    FILE *fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_byte **row_pointers = NULL;
    
    // Open a handle for the file we're going to write to
    fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    // Create the data and info structures used by libpng
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);
    if (!png_ptr || !info_ptr)
        return -1;
    
    // Set up libpng to write a XRES x YRES image with 8 bits of RGB color depth
    png_set_IHDR(png_ptr, info_ptr, XRES, YRES, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    // Uncomment this if you want to save hard drive space; it takes a lot
    // longer for the CPU to write the PNGs, but they're significantly smaller
    // png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);

    float *pixel;
    // Allocate an array of pointers to each row of pixels in the PNG file
    row_pointers = (png_byte **) png_malloc(png_ptr, YRES * sizeof(png_byte *));
    for (int y = 0; y < YRES; y++) {
        // Allocate a row of pixels in the PNG file
        png_byte *row = (png_byte *) png_malloc(png_ptr,
                                                3 * XRES * sizeof(uint8_t));
        // The screen matrix has (x, y) = (0, 0) in the lower left so that it's
        // easy to translate between it and the "camera sensor" grid, so we have
        // to go through the rows in reverse because of convention
        row_pointers[YRES - 1 - y] = row;
        for (int x = 0; x < XRES; x++) {
            // Get the pixel's location and fill in its red, green, and blue
            // values, bounding them between 0 and MAX_INTENSITY = 255
            pixel = screen + 3 * (y * XRES + x);
            *row++ = (uint8_t) (fmin(fmax(0, pixel[0]), 1.0) * MAX_INTENSITY);
            *row++ = (uint8_t) (fmin(fmax(0, pixel[1]), 1.0) * MAX_INTENSITY);
            *row++ = (uint8_t) (fmin(fmax(0, pixel[2]), 1.0) * MAX_INTENSITY);
        }
    }

    // Write the PNG to file
    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    
    // Free each row of the PNG that we allocated
    for (int y = 0; y < YRES; y++) {
        png_free(png_ptr, row_pointers[y]);
    }
    // Free the array of row pointers
    png_free(png_ptr, row_pointers);

    // Return and indicate success
    return 0;
}

int main(int argc, char **argv) {
    // Don't use the GPU driving the display
    cudaSetDevice(1);

    // The position of the camera, and the target it's looking directly at
    float h_cam_pos[3] = {0.0, 0.0, 0.0};
    float target[3] = {0.0, 0.0, -1.0};
    
    // Sizes for the screen, sphere, plane, and light arrays
    size_t screen_size = XRES * YRES * 3 * sizeof(float);
    size_t spheres_size = SPHERE_COUNT * sizeof(Sphere);
    size_t planes_size = PLANE_COUNT * sizeof(Plane);
    size_t lights_size = LIGHT_COUNT * sizeof(Light);

    // Allocate the host arrays
    float *h_screen = (float *) malloc(screen_size);
    Sphere *h_spheres = (Sphere *) malloc(spheres_size);
    Plane *h_planes = (Plane *) malloc(planes_size);
    Light *h_lights = (Light *) malloc(lights_size);

    // Allocate the device arrays
    float *d_screen;
    Sphere *d_spheres;
    Plane *d_planes;
    Light *d_lights;
    cudaMalloc((void **) &d_screen, screen_size);
    cudaMalloc((void **) &d_spheres, spheres_size);
    cudaMalloc((void **) &d_planes, planes_size);
    cudaMalloc((void **) &d_lights, lights_size);
    
    // Set up some stuff in the environment for testing
    Sphere *sphere = h_spheres;
    sphere->density = 1.0;
    sphere->radius = 0.5;
    sphere->loc[0] = -1.5 - 5.0;
    sphere->loc[1] = 0.0;
    sphere->loc[2] = -10.0;
    sphere->vel[0] = 0.0;
    sphere->vel[1] = 0.0;
    sphere->vel[2] = 0.0;

    sphere = h_spheres + 1;
    sphere->density = 1.0;
    sphere->radius = 0.5;
    sphere->loc[0] = 0.0 - 5.0;
    sphere->loc[1] = 0.0;
    sphere->loc[2] = -10.0;
    sphere->vel[0] = 0.0;
    sphere->vel[1] = 0.0;
    sphere->vel[2] = 0.0;

    sphere = h_spheres + 2;
    sphere->density = 1.0;
    sphere->radius = 0.5;
    sphere->loc[0] = 1.5 - 5.0;
    sphere->loc[1] = 0.0;
    sphere->loc[2] = -10.0;
    sphere->vel[0] = 0.0;
    sphere->vel[1] = 0.0;
    sphere->vel[2] = 0.0;


    Plane *plane = h_planes;
    plane->origin[0] = 0.0;
    plane->origin[1] = -1.0;
    plane->origin[2] = -10.0;
    plane->u[0] = 1.0;
    plane->u[1] = 0.0;
    plane->u[2] = 0.0;
    plane->v[0] = 0.0;
    plane->v[1] = 0.5;
    plane->v[2] = -1.0;
    plane->u_min = -4.0;
    plane->u_max = 4.0;
    plane->v_min = -4.0;
    plane->v_max = 4.0;


    Light *light = h_lights;
    light->position[0] = 0.0;
    light->position[1] = 4.5;
    light->position[2] = -10.0;
    light->color[0] = 1.0;
    light->color[1] = 0.41176470588;
    light->color[2] = 0.70588235294;
    light->attenuation_k = 0.04;

    // light = h_lights + 1;
    // light->position[0] = 0.0;
    // light->position[1] = -5.0;
    // light->position[2] = -10.0;
    // light->color[0] = 1.0;
    // light->color[1] = 1.0;
    // light->color[2] = 1.0;
    // light->attenuation_k = 0.04;

    // Copy the sphere, plane, and light arrays to the device
    cudaMemcpy(d_spheres, h_spheres, spheres_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_planes, h_planes, planes_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, h_lights, lights_size, cudaMemcpyHostToDevice);

    // Get basis vectors for the camera space; we could do this on the device
    // but we only need them once and it's faster to do it with the CPU
    float h_e1[3];
    float h_e2[3];
    float h_e3[3];
    // Vector from the camera to its target - SHOULD BE MORE OR LESS HORIZONTAL
    for (int i = 0; i < 3; i++)
        h_e3[i] = target[i] - h_cam_pos[i];
    h_normalize(h_e3);
    // Vector representing the vertical component of camera space; this is
    // equivalent to e2 = up - e3 * (up dot e3), where up = {0.0, 1.0, 0.0}, the
    // vertical vector in world space
    for (int i = 0; i < 3; i++)
        h_e2[i] = -h_e3[i] * h_e3[1];
    h_e2[1] += 1.0;
    h_normalize(h_e2);
    // e1 = e3 cross e2
    h_e1[0] = h_e3[1] * h_e2[2] - h_e3[2] * h_e2[1];
    h_e1[1] = h_e3[2] * h_e2[0] - h_e3[0] * h_e2[2];
    h_e1[2] = h_e3[0] * h_e2[1] - h_e3[1] * h_e2[0];
    h_normalize(h_e1);

    // Allocate device memory for the camera-related vectors
    size_t vec_size = 3 * sizeof(float);
    float *d_cam_pos, *d_e1, *d_e2, *d_e3;
    cudaMalloc((void **) &d_cam_pos, vec_size);
    cudaMalloc((void **) &d_e1, vec_size);
    cudaMalloc((void **) &d_e2, vec_size);
    cudaMalloc((void **) &d_e3, vec_size);

    // Copy the camera-related vectors from host to device
    cudaMemcpy(d_cam_pos, h_cam_pos, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e1, h_e1, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e2, h_e2, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e3, h_e3, vec_size, cudaMemcpyHostToDevice);

    // Basic event loop, change as necessary
    for (int i = 0; i < 1000; i++) {
        // KEEP THIS THOUGH since it sets all pixels to black initially
        cudaMemset(d_screen, 0.0, screen_size);

        // Raytrace a frame
        call_raytrace_kernel(d_screen, d_spheres, d_planes, d_lights,
                             SPHERE_COUNT, PLANE_COUNT, LIGHT_COUNT, d_cam_pos,
                             d_e1, d_e2, d_e3,  FD, FX, FX * YRES / XRES, XRES,
                             YRES, 1 / N_INDEX);

        // Copy the frame back to the host so it can be output as a PNG
        cudaMemcpy(h_screen, d_screen, screen_size, cudaMemcpyDeviceToHost);

        // Save the frame as a PNG; by default, these go in an "output" folder,
        // with their filename being the frame number they represent
        char path[100];
        sprintf(path, "./output/%d.png", i);
        if (save_png(h_screen, path) == -1) {
            fprintf(stderr, "Error: couldn't write frame to file\n");
            break;
        }

        // This just moves the spheres in the test scene to the right
        for (int j = 0; j < SPHERE_COUNT; j++)
            (h_spheres + j)->loc[0] += 0.01;

        // Update the sphere array
        cudaMemcpy(d_spheres, h_spheres, spheres_size, cudaMemcpyHostToDevice);
    }

    // Free all of our allocated memory
    cudaFree(d_screen);
    cudaFree(d_spheres);
    cudaFree(d_planes);
    cudaFree(d_lights);
    cudaFree(d_cam_pos);
    cudaFree(d_e1);
    cudaFree(d_e2);
    cudaFree(d_e3);

    free(h_screen);
    free(h_spheres);
    free(h_lights);
}
