#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>

#include "boids.h"

#define WIDTH 800
#define HEIGHT 450

#define MAX_VELOCITY 20
#define MIN_VELOCITY -20
#define TURNBACK_DISTANCE 100
#define TURN_FACTOR 0.2
#define CENTERING_FACTOR 0.0005
#define AVOID_FACTOR 0.005
#define MATCHING_FACTOR 0.005
#define SEPARATION_DISTANCE 8
#define COHESION_DISTANCE 20

__device__ float cap_velocity(float velocity)
{
    return fmaxf(MIN_VELOCITY, fminf(MAX_VELOCITY, velocity));
}

__device__ float distance(Vector2 a, Vector2 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrtf(dx * dx + dy * dy);
}

__global__ void initBoidsKernel(Boid *flock, curandState *states, int seed, int size, int rank)
{
    // printf("running here\n");
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_count = blockDim.x * gridDim.x;
    // printf("initBoidsKernel %d\n", i);
    if (i >= size)
        return;

    curand_init(seed, i, 0, &states[i]);

    float x = curand_uniform(&states[i]) * WIDTH;
    float y = curand_uniform(&states[i]) * HEIGHT;
    float vx = curand_uniform(&states[i]) * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY;
    float vy = curand_uniform(&states[i]) * (MAX_VELOCITY - MIN_VELOCITY) + MIN_VELOCITY;
    // printf("Boid %d: (%f, %f) velocity (%f, %f)\n", i, x, y, vx, vy);
    flock[i].id = i + rank * size;
    flock[i].position = {x, y};
    flock[i].velocity = {vx, vy};
    flock[i].rotation = 0.0f;
    flock[i].timestep = 0;
}

__global__ void updateBoidsKernel(Boid *local_flock, Boid *full_flock, int local_size, int timestep, int FLOCKSIZE)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("updateBoidsKernel %d\n", i);
    if (i >= local_size)
        return;

    Boid *boid = &local_flock[i];

    Vector2 cohesion = {0, 0}, alignment = {0, 0}, separation = {0, 0};
    int cohesion_count = 0, separation_count = 0;

    for (int j = 0; j < FLOCKSIZE; j++)
    {
        if (boid->id == full_flock[j].id)
            continue;
        float dist = distance(boid->position, full_flock[j].position);
        if (dist < COHESION_DISTANCE)
        {
            cohesion.x += full_flock[j].position.x;
            cohesion.y += full_flock[j].position.y;
            alignment.x += full_flock[j].velocity.x;
            alignment.y += full_flock[j].velocity.y;
            cohesion_count++;
        }
        if (dist < SEPARATION_DISTANCE)
        {
            separation.x += boid->position.x - full_flock[j].position.x;
            separation.y += boid->position.y - full_flock[j].position.y;
            separation_count++;
        }
    }

    if (cohesion_count > 0)
    {
        cohesion.x /= cohesion_count;
        cohesion.y /= cohesion_count;
        alignment.x /= cohesion_count;
        alignment.y /= cohesion_count;
    }

    if (separation_count > 0)
    {
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
}

// Called from host to initialize boids
extern "C" Boid *createBoids(int size, int rank, void **d_states, int thread_count)
{
    Boid *flock;
    cudaMallocManaged(&flock, size * sizeof(Boid));

    curandState *states;
    cudaMallocManaged((void **)&states, size * sizeof(curandState));
    *d_states = states;

    initBoidsKernel<<<(size + thread_count - 1) / thread_count, thread_count>>>(flock, states, time(NULL), size, rank);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return flock;
}

extern "C" void updateBoids(Boid *local_flock, Boid *full_flock, int local_size, int timestep, int thread_count, int FLOCKSIZE)
{
    // printf("updateBoids %d\n", local_size);
    updateBoidsKernel<<<(local_size + thread_count - 1) / thread_count, thread_count>>>(local_flock, full_flock, local_size, timestep, FLOCKSIZE);
    cudaDeviceSynchronize();
}

extern "C" Boid *allocateFullFlock(int size)
{
    Boid *full;
    cudaMallocManaged(&full, size * sizeof(Boid), cudaMemAttachGlobal);
    return full;
}