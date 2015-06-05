#ifndef ENTITIES_H
#define ENTITIES_H

#include <Eigen/Eigen>

using namespace Eigen;

// dynamic object
typedef struct sphere_t {
    /********** init variables **********/

    float density;  // density
    float radius;   // shape information
    Vector3f loc;   // location
    Vector3f vel;   // velocity

    /******* constructed variables ******/

    float mass;     // mass = density * radius ^ 3 * pi * 4.0 / 3.0
} Sphere;

// static object
typedef struct plane_t {
    // a corner location
    Vector3f origin;

    // two normalized vectors for parameterizing plane
    Vector3f u;
    Vector3f v;

    // range that the plane extends to from the origin in the u and v direction
    float u_min, u_max;
    float v_min, v_max;
} Plane;

#endif