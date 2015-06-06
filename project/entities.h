#ifndef ENTITIES_H
#define ENTITIES_H

// dynamic object
typedef struct sphere_t {
    /********** init variables **********/

    float density;  // density
    float radius;   // shape information
    float loc[3];   // location
    float vel[3];   // velocity

    /******* constructed variables ******/

    float mass;     // mass = density * radius ^ 3 * pi * 4.0 / 3.0
} Sphere;

// static object
typedef struct plane_t {
    // a corner location
    float origin[3];

    // two normalized vectors for parameterizing plane
    float u[3];
    float v[3];

    // range that the plane extends to from the origin in the u and v direction
    float u_min, u_max;
    float v_min, v_max;
} Plane;

// Structure representing a point light
typedef struct light_t {
    float position[3];      // Its position
    float color[3];         // Its color in RGB as floats between 0 and 1
    float attenuation_k;    // Its attenuation factor
} Light;

// Structure representing the intersection of a ray and an object
typedef struct intersect_t {
    int sphere;         // Is this a sphere or not (i.e. a plane)?
    void *object;       // Pointer to the object this is an intersection with
    float t;            // Distance (in ray vectors) from source to intersection
    float position[3];  // The intersection's position
    float normal[3];    // The object's surface normal at the intersection
} Intersect;

#endif