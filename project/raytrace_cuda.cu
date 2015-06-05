#include "raytrace_cuda.cuh"

#include <cstdio>

#include <Eigen/Eigen>

#include "entities.h"

using namespace std;
using namespace Eigen;

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define RECURSION_DEPTH 1

#define MAX_INTENSITY 255 // Maximum color intensity
#define N_INDEX 1.52f // Refraction index of glass

#define XRES 1920
#define YRES 1080

#define SPHERE_COUNT 2
#define LIGHT_COUNT 2

typedef struct point_light_t {
    Vector3f position;
    Vector3f color;
    float attenuation_k;
} Point_Light;

typedef struct intersection_t {
    Sphere *object;
    float t;
    Vector3f position;
    Vector3f normal;
} Intersection;

Point_Light *lights;
Sphere *spheres;

Vector3f cam_pos(0.0, 0.0, 0.0);
Vector3f target(0.0, 0.0, -1.0);
float Fd = 0.05;
float Fx = 0.035, Fy = Fx * YRES / XRES;
Vector3f *screen;

Vector3f get_lighting(Intersection intersect, Vector3f parent, int depth);

__device__
float sphere_io(Vector3f p, Sphere *sphere) {
    Vector3f rel = p - sphere->loc;
    return sqrtf(rel.dot(rel)) - sphere->radius;
}

__device__
Vector3f sphere_normal(Vector3f p, Sphere *sphere) {
    return p - sphere->loc;
}

__device__
Vector3f normalize(Vector3f v) {
    float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return v / length;
}

/**
 * Calculates the attenuation to apply in the Phong lighting model given the
 * vector difference between the light and the point to be rendered, and a
 * factor k
 */
 __device__
float get_attenuation(Vector3f d, float k) {
    // If k is zero (the default), then we don't have to do any computation
    return (k == 0.0) ? 1.0 : 1.0 / (1.0 + k * d.dot(d));
}

/**
 * Calculates the color of a point using the Phong lighting model, given the
 * surface's normal at that point
 */
 __device__
Vector3f phong_lighting(Vector3f point, Vector3f normal, Vector3f eye,
                        Point_Light *light) {
    Vector3f diffuse_sum, specular_sum, eye_rel, color;
    diffuse_sum = Vector3f(0.0, 0.0, 0.0);
    specular_sum = Vector3f(0.0, 0.0, 0.0);
    eye_rel = normalize(eye - point);

    Vector3f light_rel, light_color;
    light_rel = light->position - point;
    // Calculate and apply the attenuation
    light_color =
        light->color * get_attenuation(light_rel, light->attenuation_k);
    light_rel = normalize(light_rel);

    // Calculate and add the light's contribution to the diffuse and
    // specular reflection of the point
    diffuse_sum += light_color * fmax(0.0, normal.dot(light_rel));
    specular_sum +=
        light_color * fmax(0.0, normal.dot(normalize(eye_rel + light_rel)));

    // Calculate and return the overall point color intensities for red, green,
    // and blue
    for (int i = 0; i < 3; i++) {
        color[i] = fmin(1, diffuse_sum[i] + specular_sum[i]) * MAX_INTENSITY;
    }
    return color;
}

/**
 * Uses Newton's Method to intersect a ray a starting from the camera's position
 * with the surface of a superquadric
 */
 __device__
Intersection get_sphere_intersection(Vector3f av, Vector3f bv, Sphere *sphere,
                                     int refracting = 0) {
    Intersection intersect;
    Vector3f rel = bv - sphere->loc;
    float a = av.dot(av);
    float b = 2 * av.dot(rel);
    float c = rel.dot(rel) - sphere->radius * sphere->radius;

    float disc = b * b - 4 * a * c;
    if (disc < 0.0) {
        intersect.t = -1.0;
        return intersect;
    }
    disc = sqrtf(disc);
    float t = (-b - disc) / (2 * a);
    if (refracting)
        t += disc / a;
    if (t < 0.0)
        t = -1.0;

    // Store the intersection point and the normal vector at it
    intersect.t = t;
    if (t != -1.0) {
        intersect.object = sphere;
        intersect.position = av * t + bv;
        intersect.normal = normalize(sphere_normal(intersect.position, sphere));
    }
    return intersect;
}

/**
 * Gets the intersection of a ray 'av' with a superquadric surface nearest to
 * the ray's origin, 'bv'
 */
 __device__
Intersection get_nearest_intersection(Vector3f av, Vector3f bv, Sphere *start) {
    // Loop through every object in the scene and test it for intersection
    Intersection nearest, temp;
    nearest.t = -1.0;
    float distance, best_dist = -1.0;
    for (int i = 0; i < SPHERE_COUNT; i++) {
        // Skip over the object the ray started on, if this is a child ray
        if (spheres + i == start)
            continue;
        // If the ray intersects the object closer than any of the other objects
        // we've tested so far, make this the new intersection point
        temp = get_sphere_intersection(av, bv, spheres + i, 0);
        if (temp.t != -1.0) {
            distance = (bv - temp.position).norm();
            if (distance < best_dist || nearest.t == -1) {
                nearest = temp;
                nearest.object = spheres + i;
                best_dist = distance;
            }
        }
    }
    return nearest;
}

/**
 * Calculates the shading of a point using the Phong lighting model, taking into
 * account the fact that some light sources are blocked by objects, casting
 * shadows
 */
 __device__
Vector3f get_shadows(Intersection intersect, Vector3f parent) {
    // Loop through all the lights, send a ray at each one, and test if it hits
    // an object along the way
    Vector3f color(0.0, 0.0, 0.0);
    Intersection blocked;
    for (int i = 0; i < LIGHT_COUNT; i++) {
        for (int j = 0; j < SPHERE_COUNT; j++) {
            // Skip over the object the point we're shading is on
            if (spheres + j == (Sphere *) intersect.object) {
                blocked.t = -1.0;
                continue;
            }
            blocked =
                get_sphere_intersection((lights + i)->position - intersect.position,
                                        intersect.position, spheres + j, 0);
            if (blocked.t != -1.0)
                break;
        }
        // If the light isn't blocked, calculate its contribution to the point
        if (blocked.t == -1.0) {
            color += phong_lighting(intersect.position, intersect.normal,
                parent, lights + i);
        }
    }
    return color;
}

/**
 * Recursively calculates the contributions of reflected rays to a point's
 * color
 */
 __device__
Vector3f get_reflection(Intersection intersect, Vector3f parent, int depth) {
    if (intersect.t == -1.0) {
        Vector3f color(0.0, 0.0, 0.0);
        return color;
    }
    // Rotate the vector from the intersection to the eye 180 degrees around the
    // normal, as though it reflected off of the surface
    Vector3f av = normalize(parent - intersect.position);
    Vector3f nv = normalize(intersect.normal);
    nv = av.dot(nv) * nv;
    av = nv - (av - nv);
    // Find the next object this ray hits, and calculate its shading, and add it
    // to the shading at the ray's start location
    Intersection next = get_nearest_intersection(av, intersect.position,
                                                 intersect.object);
    return get_lighting(next, intersect.position, depth);
}

/**
 * Recursively calculates the contributions of frefracted rays to a point's 
 * color
 */
 __device__
Vector3f get_refraction(Intersection intersect, Vector3f parent, int depth) {
    if (intersect.t == -1.0) {
        Vector3f color(0.0, 0.0, 0.0);
        return color;
    }
    // Calculate the incoming vector
    Vector3f incoming = normalize(intersect.position - parent);
    Vector3f av(0.0, 0.0, 0.0);
    float n = 1 / N_INDEX;
    // Calculate the cosine of the incident angle and the squared cosine of the
    // refracted angle
    float cos1 = -incoming.dot(intersect.normal);
    float cossq2 = 1 - n * n * (1 - cos1 * cos1);
    // If the ray is totally reflected, don't trace it
    if (cossq2 < 0) {
        Vector3f color(0.0, 0.0, 0.0);
        return color;
    }
    // Calculate the refracted ray, and find its intersection with the other
    // side of the superquadric
    av = n * incoming + (n * cos1 - sqrt(cossq2)) * intersect.normal;
    Intersection next = get_sphere_intersection(av, intersect.position,
                                                intersect.object, 1);

    // Repeat the process, refracting the ray with the opposite relative index
    // of refraction
    incoming = av;
    n = N_INDEX;
    cos1 = incoming.dot(next.normal);
    cossq2 = 1 - n * n * (1 - cos1 * cos1);
    if (cossq2 < 0) {
        Vector3f color(0.0, 0.0, 0.0);
        return color;
    }
    // Follow this ray to the first object it hits, and calculate the shading
    // there, so we can add it to the initially refracted point
    av = n * incoming + (sqrt(cossq2) - n * cos1) * next.normal;
    intersect = get_nearest_intersection(av, next.position,
                                         intersect.object);
    return get_lighting(intersect, next.position, depth);
}

/**
 * Calculates the lighting at the intersection point of a ray with a
 * superquadric surface, by taking into account shadows, reflections, and (in
 * theory) refractions
 */
__device__
Vector3f get_lighting(Intersection intersect, Vector3f parent, int depth)
{
    Vector3f color(0.0, 0.0, 0.0);
    // If the intersection isn't really an intersection, return black
    if (intersect.t == -1.0)
        return color;
    // Test each light for if it's blocked, and calculate its contribution
    color += get_shadows(intersect, parent);
    // Recursively trace the rays reflected and refracted from the intersection
    // to get their contribution
    if (depth < RECURSION_DEPTH) {
        color += get_reflection(intersect, parent, depth + 1);
        color += get_refraction(intersect, parent, depth + 1);
    }

    // Convert the color to one in the range [0, 255].
    for (int i = 0; i < 3; i++)
        color[i] = fmax(fmin(color[i], MAX_INTENSITY), 0);
    return color;
}