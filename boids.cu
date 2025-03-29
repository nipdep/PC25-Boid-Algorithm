
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define WIDTH 800
#define HEIGHT 450
#define flocksize 100
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

__global__ void updateBoids(Boid* flock, int timestep) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= flocksize) return;

    Boid* boid = &flock[i];

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

int main() {
    Boid* d_flock;
    curandState* d_states;

    cudaMalloc(&d_flock, sizeof(Boid) * flocksize);
    cudaMalloc(&d_states, sizeof(curandState) * flocksize);

    initBoids<<<(flocksize + 255)/256, 256>>>(d_flock, d_states, time(NULL));
    cudaDeviceSynchronize();

    for (int t = 0; t < FRAMES; t++) {
        updateBoids<<<(flocksize + 255)/256, 256>>>(d_flock, t);
        cudaDeviceSynchronize();
    }

    Boid h_flock[flocksize];
    cudaMemcpy(h_flock, d_flock, sizeof(Boid) * flocksize, cudaMemcpyDeviceToHost);

    cudaFree(d_flock);
    cudaFree(d_states);
    return 0;
}
