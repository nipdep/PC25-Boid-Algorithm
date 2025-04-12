
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define flocksize 100
#define FRAMES 60

struct Vector2 {
    float x, y;
} typedef Vector2;

struct Boid {
    int id;
    Vector2 position;
    Vector2 velocity;
    float rotation;
    int timestep;
} typedef Boid;

// Declaration of the CUDA kernel launcher
extern void updateBoidsKernel(struct Boid* d_flock, void* d_states, int timestep);

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_count = flocksize / size;
    struct Boid* full_flock = (struct Boid*)malloc(sizeof(struct Boid) * flocksize);
    struct Boid* local_flock;

    local_flock = createBoids(local_count);

    MPI_Gather(&full_flock[rank * local_count], local_count * sizeof(struct Boid), MPI_BYTE,
                   full_flock, local_count * sizeof(struct Boid), MPI_BYTE, 0, MPI_COMM_WORLD);
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // print boids in flock 
        for (int i = 0; i < flocksize; i++) {
        printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
            full_flock[i].id, full_flock[i].timestep,
            full_flock[i].position.x, full_flock[i].position.y,
            full_flock[i].velocity.x, full_flock[i].velocity.y);
        }
    }

    for (int t = 0; t < FRAMES; t++) {
        // Bcast whole flock to all processes
        MPI_Bcast(full_flock, sizeof(struct Boid) * flocksize, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Run update kernel on GPU
        local_flock = updateBoids(local_flock, full_flock, t);

        // Optionally gather back results to root
        MPI_Gather(local_flock, local_count * sizeof(struct Boid), MPI_BYTE,
                   full_flock, local_count * sizeof(struct Boid), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // print boids in flock 
            for (int i = 0; i < flocksize; i++) {
            printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                full_flock[i].id, full_flock[i].timestep,
                full_flock[i].position.x, full_flock[i].position.y,
                full_flock[i].velocity.x, full_flock[i].velocity.y);
            }
        }
    }

    cudaFree(d_flock);
    cudaFree(d_states);
    free(full_flock);

    MPI_Finalize();
    return 0;
}
