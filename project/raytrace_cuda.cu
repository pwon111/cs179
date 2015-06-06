#include "raytrace_cuda.cuh"


/* Utility function to normalize a vector. */
__device__
void normalize(float *v) {
    float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; i++)
        v[i] /= length;
}

/* Utility function to compute the dot product of two vectors. */
__device__
float dotf(float *a, float *b) {
    float d = 0.0;
    for (int i = 0; i < 3; i++)
        d += a[i] * b[i];
    return d;
}

/**
 * This function calculates the attenuation to apply in the Phong lighting model 
 * given the vector difference between the light and the point to be rendered,
 * and a factor k.
 */
__device__
float get_attenuation(float *d, float k) {
    // If k is zero (the default), then we don't have to do any computation
    return (k == 0.0) ? 1.0 : 1.0 / (1.0 + k * dotf(d, d));
}

/**
 * This function calculates the color of a point due to a single light using the
 * Phong lighting model, given the surface's normal at that point.
 */
__device__
void phong_lighting(float *point, float *normal, float *eye, Light *light,
                    float *pixel) {
    float diffuse_sum[3] = {0.0, 0.0, 0.0};
    float specular_sum[3] = {0.0, 0.0, 0.0};
    // Get the vector from the camera to the point
    float eye_rel[3];
    for (int i = 0; i < 3; i++)
        eye_rel[i] = eye[i] - point[i];
    normalize(eye_rel);

    // Get the vector from the light to the point
    float light_rel[3];
    for (int i = 0; i < 3; i++)
        light_rel[i] = light->position[i] - point[i];
    // Calculate and apply the attenuation
    float light_color[3];
    for (int i = 0; i < 3; i++) {
        light_color[i] =
            light->color[i] * get_attenuation(light_rel, light->attenuation_k);
    }
    normalize(light_rel);

    // Calculate and add the light's contribution to the diffuse and
    // specular reflection of the point
    float p = fmax(0.0, dotf(normal, light_rel));
    for (int i = 0; i < 3; i++)
        diffuse_sum[i] += light_color[i] * p;
    for (int i = 0; i < 3; i++)
        eye_rel[i] += light_rel[i];
    normalize(eye_rel);
    p = fmax(0.0, dotf(normal, eye_rel));
    for (int i = 0; i < 3; i++)
        specular_sum[i] += light_color[i] * p;

    // Calculate and add the overall point color intensities for red, green,
    // and blue
    for (int i = 0; i < 3; i++) {
        pixel[i] += fmin(1, diffuse_sum[i] + specular_sum[i]);
    }
}

/**
 * This function calculates the intersection of a ray 'bv' shot from source 'av'
 * with a given sphere.
 */
__device__
Intersect get_sphere_intersection(float *av, float *bv, Sphere *sphere,
                                  int refracting) {
    // Given a sphere with center C and radius R, and a ray equation B + tA,
    // the intersection of the two can be found by solving |B + tA - C|^2 = R^2,
    // which enforces that the ray be on the sphere's surface. This is a
    // quadratic equation in t.
    Intersect intersect;
    float rel[3];
    for (int i = 0; i < 3; i++)
        rel[i] = bv[i] - sphere->loc[i];
    float a = dotf(av, av);
    float b = 2 * dotf(av, rel);
    float c = dotf(rel, rel) - sphere->radius * sphere->radius;

    float disc = b * b - 4 * a * c;
    // If the discriminant is less than 0, there is no solution and thus no
    // intersection
    if (disc < 0.0) {
        intersect.t = -1.0;
        return intersect;
    }
    disc = sqrtf(disc);
    // Choose the smaller solution for t
    float t = (-b - disc) / (2 * a);
    // If we're using this while we're refracting we're actually trying to go
    // across the sphere to the other side, so we want the larger solution for t
    if (refracting)
        t += disc / a;
    // If t is less than 0, then sphere is behind the camera (or the camera is
    // inside the sphere, in which case we just don't render it)
    if (t < 0.0)
        t = -1.0;

    // Store the intersection point and the normal vector at it
    intersect.t = t;
    if (t != -1.0) {
        intersect.sphere = 1;
        intersect.object = sphere;
        for (int i = 0; i < 3; i++)
            intersect.position[i] = av[i] * t + bv[i];
        float normal[3];
        for (int i = 0; i < 3; i++)
            normal[i] = intersect.position[i] - sphere->loc[i];
        normalize(normal);
        for (int i = 0; i < 3; i++)
            intersect.normal[i] = normal[i];
    }
    return intersect;
}

/**
 * This function calculates the intersection of a ray 'bv' shot from source 'av'
 * with a given plane.
 */
__device__
Intersect get_plane_intersection(float *av, float *bv, Plane *plane) {
    // Given a plane with a normal vector N and containing a point Q, and a ray
    // equation B + tA, the value of t at which the ray intersects the plane is
    // given as t = (N dot (Q - B)) / (N dot A)
    Intersect intersect;
    float rel[3];
    // Calculate Q - B
    for (int i = 0; i < 3; i++)
        rel[i] = plane->origin[i] - bv[i];
    float normal[3];
    float *u = plane->u;
    float *v = plane->v;
    // Calculate the plane's normal vector as u cross v
    normal[0] = u[1] * v[2] - u[2] * v[1];
    normal[1] = u[2] * v[0] - u[0] * v[2];
    normal[2] = u[0] * v[1] - u[1] * v[0];
    normalize(normal);
    // If the normal is perpendicular to the ray vector, they don't intersect
    float t = dotf(normal, av);
    if (t != 0.0) {
        t = dotf(normal, rel) / t;
        // If t < 0 then the plane is behind the ray's origin
        if (t < 0.0)
            t = -1.0;
    }
    else
        t = -1.0;

    intersect.t = t;
    if (t != -1.0) {
        // Calculate the point on the plane that the ray intersects
        float position[3];
        for (int i = 0; i < 3; i++)
            position[i] = av[i] * t + bv[i];
        // Get its position relative to the plane's origin
        for (int i = 0; i < 3; i++)
            rel[i] = position[i] - plane->origin[i];
        // Project this vector onto the plane's u vector and make sure that it
        // falls within the plane's u-wise bounds
        float p = dotf(rel, plane->u);
        if (p < plane->u_min || p > plane->u_max) {
            intersect.t = -1.0;
            return intersect;
        }
        // Project this vector onto the plane's v vector and make sure that it
        // falls within the plane's v-wise bounds
        p = dotf(rel, plane->v);
        if (p < plane->v_min || p > plane->v_max) {
            intersect.t = -1.0;
            return intersect;
        }

        // Mark this intersection as a plane and fill in its values
        intersect.sphere = 0;
        intersect.object = plane;
        for (int i = 0; i < 3; i++) {
            intersect.position[i] = position[i];
            intersect.normal[i] = normal[i];
        }
    }
    return intersect;
}

/**
 * This function gets the intersection of a ray 'av' with an object surface
 * nearest to the ray's origin, 'bv'.
 */
__device__
Intersect get_nearest_intersection(float *av, float *bv, void *start,
                                   Sphere *spheres, Plane *planes,
                                   int sphere_count, int plane_count) {
    // Loop through every object in the scene and test it for intersection
    Intersect nearest, temp;
    nearest.t = -1.0;
    void *object;
    // Check all of the spheres
    for (int i = 0; i < sphere_count; i++) {
        object = (void *) (spheres + i);
        // Skip over the object the ray started on, if this is a child ray
        if (object == start)
            continue;
        // If the ray intersects the sphere closer than any of the other objects
        // we've tested so far, make this the new intersection point
        temp = get_sphere_intersection(av, bv, (Sphere *) object, 0);
        if (temp.t != -1.0 && (temp.t < nearest.t || nearest.t == -1)) {
            nearest = temp;
            nearest.object = object;
        }
    }

    // Then check all of the planes
    for (int i = 0; i < plane_count; i++) {
        object = (void *) (planes + i);
        // Skip over the object the ray started on, if this is a child ray
        if (object == start)
            continue;
        // If the ray intersects the plane closer than any of the other objects
        // we've tested so far, make this the new intersection point
        temp = get_plane_intersection(av, bv, (Plane *) object);
        if (temp.t != -1.0 && (temp.t < nearest.t || nearest.t == -1)) {
            nearest = temp;
            nearest.object = object;
        }
    }

    return nearest;
}

/**
 * This function calculates the shading of a point using the Phong lighting
 * model, taking into account the fact that some light sources are blocked by
 * objects, casting shadows.
 */
__device__
void get_shadows(Sphere *spheres, Plane *planes, Light *lights,
                 int sphere_count, int plane_count, int light_count,
                 Intersect intersect, float *parent, float *pixel) {
    // Loop through all the lights, send a ray at each one, and test if it hits
    // an object along the way
    Intersect blocked;
    float outgoing[3];
    Light *light;
    void *object;
    for (int i = 0; i < light_count; i++) {
        light = lights + i;
        blocked.t = -1.0;
        // Loop through all the spheres to see if the ray hits one
        for (int j = 0; j < sphere_count; j++) {
            object = (void *) (spheres + j);
            // Skip over the object the point we're shading is on
            if (object == intersect.object)
                continue;

            for (int k = 0; k < 3; k++)
                outgoing[k] = light->position[k] - intersect.position[k];
            blocked = get_sphere_intersection(outgoing, intersect.position,
                                              (Sphere *) object, 0);
            if (blocked.t != -1.0)
                break;
        }

        // If it didn't hit any spheres, check the planes
        if (blocked.t == -1.0) {
            for (int j = 0; j < plane_count; j++) {
                object = (void *) (planes + j);
                // Skip over the object the point we're shading is on
                if (object == intersect.object)
                    continue;

                for (int k = 0; k < 3; k++)
                    outgoing[k] = light->position[k] - intersect.position[k];
                blocked = get_plane_intersection(outgoing, intersect.position,
                                                 (Plane *) object);
                if (blocked.t != -1.0)
                    break;
            }
        }

        // If the light isn't blocked, calculate its contribution to the point
        if (blocked.t == -1.0) {
            phong_lighting(intersect.position, intersect.normal, parent, light,
                           pixel);
        }
    }
}

/**
 * The function calculates the contribution of an incoming ray reflected off an
 * object to the color of the point on its surface it first struck.
 */
__device__
void get_reflection(Sphere *spheres, Plane *planes, Light *lights,
                    int sphere_count, int plane_count, int light_count,
                    Intersect intersect, float *parent, float n, float *pixel) {
    // Rotate the vector from the intersection to the eye 180 degrees around the
    // normal, as though it reflected off of the surface
    float av[3];
    for (int i = 0; i < 3; i++)
        av[i] = parent[i] - intersect.position[i];
    normalize(av);

    float p = dotf(av, intersect.normal);    
    for (int i = 0; i < 3; i++)
        av[i] = 2 * p * intersect.normal[i] - av[i];
    // Find the next object this ray hits, and calculate its shading, and add it
    // to the shading at the ray's start location
    Intersect next = get_nearest_intersection(av, intersect.position,
                                              (Sphere *) intersect.object,
                                              spheres, planes, sphere_count,
                                              plane_count);
    get_shadows(spheres, planes, lights, sphere_count, plane_count, light_count,
                next, intersect.position, pixel);
}

/**
 * This function calculates the contribution of an incoming ray refracted
 * through an object to the color of the point on its surface it first struck.
 */
__device__
void get_refraction(Sphere *spheres, Plane *planes, Light *lights,
                    int sphere_count, int light_count, int plane_count,
                    Intersect intersect, float *parent, float n, float *pixel) {
    // Calculate the incoming vector
    float incoming[3];
    for (int i = 0; i < 3; i++)
        incoming[i] = intersect.position[i] - parent[i];
    normalize(incoming);

    // Calculate the cosine of the incident angle and the squared cosine of the
    // refracted angle
    float cos1 = -dotf(incoming, intersect.normal);
    float cossq2 = 1 - n * n * (1 - cos1 * cos1);
    // If the ray is totally reflected, don't trace it
    if (cossq2 < 0) {
        return;
    }
    // Calculate the refracted ray
    for (int i = 0; i < 3; i++) {
        incoming[i] = n * incoming[i]
                      + (n * cos1 - sqrt(cossq2)) * intersect.normal[i];
    }
    // If we're refracting through a plane, we cam just get the next object the
    // ray hits and be done
    Intersect next;
    if (!intersect.sphere) {
        next = get_nearest_intersection(incoming, intersect.position,
                                        intersect.object, spheres, planes,
                                        sphere_count, plane_count);
        get_shadows(spheres, planes, lights, sphere_count, plane_count,
                    light_count, next, intersect.position, pixel);
        return;
    }

    // Otherwise, we have to find its intersection with the other side of the
    // sphere, refract it back through the surface, and then go on
    next = get_sphere_intersection(incoming, intersect.position,
                                   (Sphere *) intersect.object, 1);

    // Repeat the process, refracting the ray with the opposite relative index
    // of refraction
    float np = 1 / n;
    cos1 = dotf(incoming, next.normal);
    cossq2 = 1 - np * np * (1 - cos1 * cos1);
    if (cossq2 < 0) {
        return;
    }
    // Follow this ray to the first object it hits, and calculate the shading
    // there, adding it to the initially refracted point
    for (int i = 0; i < 3; i++)
        incoming[i] = np * incoming[i]
                      + (np * cos1 - sqrt(cossq2)) * next.normal[i];
    intersect = get_nearest_intersection(incoming, next.position,
                                         intersect.object, spheres, planes,
                                         sphere_count, plane_count);
    get_shadows(spheres, planes, lights, sphere_count, plane_count, light_count,
                intersect, next.position, pixel);
}

/**
 * This kernel raytraces the current scene by having each thread handle an
 * individual pixel.
 */
 __global__
void raytrace_kernel(float *screen, Sphere *spheres, Plane *planes,
                     Light *lights, int sphere_count, int plane_count,
                     int light_count, float *cam_pos, float *e1, float *e2,
                     float *e3, float Fd, float Fx, float Fy, int xres,
                     int yres, float n) {
    // Get the x and y pixel coordinates (with 0, 0 in the lower left for
    // convenience)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < xres && y < yres) {
        Intersect intersect;
        intersect.t = -1.0;
        // Express the ray vector in terms of our basis by shooting it from the
        // camera through a point on its imaginary sensor grid
        float av[3];
        for (int i = 0; i < 3; i++) {
            av[i] = Fd * e3[i] + (x - xres / 2) * (Fx / xres) * e1[i]
                    + (y - yres / 2) * (Fy / yres) * e2[i];
        }
        // Trace the ray to the first surface it hits
        intersect = get_nearest_intersection(av, cam_pos, NULL, spheres, planes,
                                             sphere_count, plane_count);
        // If it hits a surface, calculate its lighting, as well as a simple
        // reflection and refraction of the ray (i.e. recursion depth 1) 
        if (intersect.t != -1.0) {
            float *pixel = screen + 3 * (y * xres + x);
            get_shadows(spheres, planes, lights, sphere_count, plane_count,
                        light_count, intersect, cam_pos, pixel);

            get_reflection(spheres, planes, lights, sphere_count, plane_count,
                           light_count, intersect, cam_pos, n, pixel);
            if (intersect.sphere) {
                get_refraction(spheres, planes, lights, sphere_count,
                               plane_count, light_count, intersect, cam_pos, n,
                               pixel);
            }
        }
    }
}

/* Calls the kernel to raytrace the current scene. */
void call_raytrace_kernel(float *screen, Sphere *spheres, Plane *planes,
                          Light *lights, int sphere_count, int plane_count,
                          int light_count, float *cam_pos, float *e1, float *e2,
                          float *e3, float Fd, float Fx, float Fy, int xres,
                          int yres, float n) {
    // Have each block handle a 32 x 32 square of pixels
    dim3 blocks((xres - 1) / 32 + 1, (yres - 1) / 32 + 1);
    dim3 threads(32, 32);
    raytrace_kernel<<<blocks, threads>>>(screen, spheres, planes, lights,
                                         sphere_count, plane_count, light_count,
                                         cam_pos, e1, e2, e3, Fd, Fx, Fy, xres,
                                         yres, n);
}
