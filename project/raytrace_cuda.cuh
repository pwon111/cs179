#include <stdio.h>

#include "entities.h"


__device__ void normalize(float *v);

__device__ float dotf(float *a, float *b);

__device__ float get_attenuation(float *d, float k);

__device__ void phong_lighting(float *point, float *normal, float *eye,
                               Light *light, float *pixel);

__device__ Intersect get_sphere_intersection(float *av, float *bv,
                                             Sphere *sphere, int refracting);

__device__ Intersect get_plane_intersection(float *av, float *bv, Plane *plane);

__device__ Intersect get_nearest_intersection(float *av, float *bv,
                                              void *start, Sphere *spheres,
                                              Plane *planes, int sphere_count,
                                              int plane_count);

__device__ void get_shadows(Sphere *spheres, Plane *planes, Light *lights,
                            int sphere_count, int plane_count, int light_count,
                            Intersect intersect, float *parent, float *pixel);

__device__ void get_reflection(Sphere *spheres, Plane *planes, Light *lights,
                               int sphere_count, int plane_count,
                               int light_count, Intersect intersect,
                               float *parent, float n, float *pixel);

__device__ void get_refraction(Sphere *spheres, Plane *planes, Light *lights,
                               int sphere_count, int plane_count,
                               int light_count, Intersect intersect,
                               float *parent, float n, float *pixel);

__global__ void raytrace_kernel(float *screen, Sphere *spheres, Plane *planes,
                                Light *lights, int sphere_count,
                                int light_count, float *cam_pos, float *e1,
                                float *e2, float *e3, float Fd, float Fx,
                                float Fy, int xres, int yres, float n);

void call_raytrace_kernel(float *screen, Sphere *spheres, Plane *planes,
                          Light *lights, int sphere_count, int plane_count,
                          int light_count, float *cam_pos, float *e1, float *e2,
                          float *e3, float Fd, float Fx, float Fy, int xres,
                          int yres, float n);
