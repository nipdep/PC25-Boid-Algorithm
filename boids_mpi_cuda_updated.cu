
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>

#define WIDTH 800
#define HEIGHT 450
#define FLOCKSIZE 100
#define FRAMES 60

#define MAX_VELOCITY 20
#define MIN_VELOCITY -20
#define TURNBACK_DISTANCE 100
#define TURN_FACTOR 0.2
#define CENTERING_FACTOR 0.0005
#define AVOID_FACTOR 0.005
#define MATCHING_FACTOR 0.005
#define SEPARATION_DISTANCE 8
#define COHESION_DISTANCE 20

struct Vector2 {
    float x, y;
};

struct Boid {
    int id;
    Vector2 position;
    Vector2 velocity;
    float rotation;
    int timestep;
};

__device__ float cap_velocity(float velocity) {
    if (velocity > MAX_VELOCITY) return MAX_VELOCITY;
    if (velocity < MIN_VELOCITY) return MIN_VELOCITY;
    return velocity;
}

__device__ float distance(Vector2 a, Vector2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void initBoids(Boid* flock, curandState* states, int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= flocksize) return;

    curand_init(seed, i, 0, &states[i]);

    float x = curand_uniform(&states[i]) * WIDTH;
    float y = curand_uniform(&states[i]) * HEIGHT;
    float vx = curand_uniform(&states[i]) * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY;
    float vy = curand_uniform(&states[i]) * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY;

    flock[i].id = i;
    flock[i].position = {x, y};
    flock[i].velocity = {vx, vy};
    flock[i].rotation = 0.0f;
    flock[i].timestep = 0;
}

__global__ void updateBoids(Boid* local_flock, Boid* full_flock, int local_size, int timestep) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= local_size) return;

    Boid* boid = &local_flock[i];

    Vector2 cohesion = {0, 0}, alignment = {0, 0}, separation = {0, 0};
    int cohesion_count = 0, separation_count = 0;

    for (int j = 0; j < flocksize; j++) {
        if (i == j) continue;
        float dist = distance(boid->position, flock[j].position);
        if (dist < COHESION_DISTANCE) {
            cohesion.x += flock[j].position.x;
            cohesion.y += flock[j].position.y;
            alignment.x += flock[j].velocity.x;
            alignment.y += flock[j].velocity.y;
            cohesion_count++;
        }
        if (dist < SEPARATION_DISTANCE) {
            separation.x += boid->position.x - flock[j].position.x;
            separation.y += boid->position.y - flock[j].position.y;
            separation_count++;
        }
    }

    if (cohesion_count > 0) {
        cohesion.x /= cohesion_count;
        cohesion.y /= cohesion_count;
        alignment.x /= cohesion_count;
        alignment.y /= cohesion_count;
    }

    if (separation_count > 0) {
        separation.x /= separation_count;
        separation.y /= separation_count;
    }

    if (boid->position.x < TURNBACK_DISTANCE)
        boid->velocity.x += TURN_FACTOR * fabsf(boid->velocity.x) + 3;
    else if (boid->position.x > WIDTH - TURNBACK_DISTANCE)
        boid->velocity.x -= TURN_FACTOR * fabsf(boid->velocity.x) + 3;
    else
        boid->velocity.x += (cohesion.x - boid->position.x) * CENTERING_FACTOR +
                            (alignment.x - boid->velocity.x) * MATCHING_FACTOR +
                            separation.x * AVOID_FACTOR;

    if (boid->position.y < TURNBACK_DISTANCE)
        boid->velocity.y += TURN_FACTOR * fabsf(boid->velocity.y) + 3;
    else if (boid->position.y > HEIGHT - TURNBACK_DISTANCE)
        boid->velocity.y -= TURN_FACTOR * fabsf(boid->velocity.y) + 3;
    else
        boid->velocity.y += (cohesion.y - boid->position.y) * CENTERING_FACTOR +
                            (alignment.y - boid->velocity.y) * MATCHING_FACTOR +
                            separation.y * AVOID_FACTOR;

    boid->velocity.x = cap_velocity(boid->velocity.x);
    boid->velocity.y = cap_velocity(boid->velocity.y);

    boid->position.x += boid->velocity.x;
    boid->position.y += boid->velocity.y;
    boid->timestep = timestep;

    printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
        boid->id, boid->timestep,
        boid->position.x, boid->position.y,
        boid->velocity.x, boid->velocity.y);
}


// create local flock of boids
Boid* createBoids(int size) {
    Boid* flock;
    cudaMallocManaged((void**)&flock, size * sizeof(Boid));

    void* states;
    cudaMallocManaged((void**)&states, size * sizeof(curandState));
    initBoids<<<(size + 255)/256, 256>>>(flock, (curandState*)states, time(NULL));
    cudaDeviceSynchronize();
    // Initialize the boids with random positions and velocities
    return flock;
}

// Called from host code

Boid* updateBoids(Boid* local_flock, Boid* full_flock, int local_size, int timestep) {
    // size of d_flock is flocksize
    Boid* local_flock;
    cudaMallocManaged((void**)&local_flock, flocksize * sizeof(Boid));

    Boid* full_flock;
    cudaMallocManaged((void**)&full_flock, flocksize * sizeof(Boid));

    updateBoids<<<(local_size + 255)/256, 256>>>(local_flock, full_flock, local_size, timestep);
    cudaDeviceSynchronize();
    // Update the boids with the new positions and velocities
    return local_flock;
}
