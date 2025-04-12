
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "boids.h"

#define FLOCKSIZE 400
#define FRAMES 60

// Declare external GPU functions
extern Boid *createBoids(int size, int rank, void **d_states);
extern Boid *allocateFullFlock(int size); // now allocated in .cu
extern void updateBoids(Boid *local_flock, Boid *full_flock, int local_size, int timestep);

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_count = FLOCKSIZE / size;

    Boid *full_flock = allocateFullFlock(FLOCKSIZE);
    Boid *local_flock = NULL;
    void *d_states;

    local_flock = createBoids(local_count, rank, &d_states);

    MPI_Gather(&full_flock[rank * local_count], local_count * sizeof(Boid), MPI_BYTE,
               full_flock, local_count * sizeof(Boid), MPI_BYTE, 0, MPI_COMM_WORLD);
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        // print boids in flock
        for (int i = 0; i < FLOCKSIZE; i++)
        {
            printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                   full_flock[i].id, full_flock[i].timestep,
                   full_flock[i].position.x, full_flock[i].position.y,
                   full_flock[i].velocity.x, full_flock[i].velocity.y);
        }
    }

    for (int t = 0; t < FRAMES; t++)
    {
        // Bcast whole flock to all processes
        MPI_Bcast(full_flock, sizeof(Boid) * FLOCKSIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Run update kernel on GPU
        updateBoids(local_flock, full_flock, local_count, t);

        // Optionally gather back results to root
        MPI_Gather(local_flock, local_count * sizeof(Boid), MPI_BYTE,
                   full_flock, local_count * sizeof(Boid), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            // print boids in flock
            for (int i = 0; i < FLOCKSIZE; i++)
            {
                printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                       full_flock[i].id, full_flock[i].timestep,
                       full_flock[i].position.x, full_flock[i].position.y,
                       full_flock[i].velocity.x, full_flock[i].velocity.y);
            }
        }
    }

    cudaFree(local_flock);
    cudaFree(d_states);
    free(full_flock);

    MPI_Finalize();
    return 0;
}
