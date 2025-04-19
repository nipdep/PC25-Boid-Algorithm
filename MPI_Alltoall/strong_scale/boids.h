#ifndef BOIDS_H
#define BOIDS_H

typedef struct {
    float x, y;
} Vector2;

typedef struct {
    int id;
    Vector2 position;
    Vector2 velocity;
    float rotation;
    int timestep;
} Boid;

#endif
