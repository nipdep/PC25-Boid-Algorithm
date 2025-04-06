// Updated boids_mpi_cuda.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "boids.h"

#define FLOCKSIZE 400
#define FRAMES 60

int flocksize = 20;
int frames = 30;

MPI_Offset global_offset = 0;

// Declare external GPU functions
extern Boid* createBoids(int size, int rank, void** d_states);
extern Boid* allocateFullFlock(int size);  // now allocated in .cu
extern void updateBoids(Boid* local_flock, Boid* full_flock, int local_size, int timestep);

void read_config(int rank, char *buffer)
{
    MPI_File fh;
    MPI_Status status;
    MPI_Offset file_size;
    int rc;

    // Open the config file collectively in read-only mode.
    rc = MPI_File_open(MPI_COMM_WORLD, "config.ini", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS)
    {
        if (rank == 0)
            fprintf(stderr, "Error opening file 'config.ini'\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(EXIT_FAILURE);
    }

    // Get the size of the file.
    MPI_File_get_size(fh, &file_size);

    // Allocate a buffer to hold the file contents plus a null terminator.
    buffer = (char *)malloc((file_size + 1) * sizeof(char));
    if (!buffer)
    {
        fprintf(stderr, "Memory allocation error\n");
        MPI_File_close(&fh);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Read the file content collectively into the buffer.
    MPI_File_read_all(fh, buffer, file_size, MPI_CHAR, &status);
    buffer[file_size] = '\0'; // Null-terminate the string

    // Split by line
    char *line = strtok(buffer, "\n");

    while (line != NULL)
    {
        // Split line into key and value
        char *equals = strchr(line, '=');
        if (equals)
        {
            *equals = '\0'; // Temporarily split the string
            char *key = line;
            char *value = equals + 1;

            // Compare and assign
            if (strcmp(key, "FLOCKSIZE") == 0)
            {
                flocksize = atoi(value);
            }
            else if (strcmp(key, "FRAMES") == 0)
            {
                frames = atoi(value);
            }
        }

        line = strtok(NULL, "\n");
    }

    printf("FLOCKSIZE: %d\n", flocksize);
    printf("FRAMES: %d\n", frames);

    MPI_File_close(&fh);
    free(buffer);
}

void write_output(Boid *full_flock, int create, int delete)
{
    MPI_File fh;

    if (delete)
    {
        // Create the file if it doesn't exist
        MPI_File_delete("boids_output.csv", MPI_INFO_NULL);
    }

    if (create)
    {
        // Create the file if it doesn't exist
        MPI_File_open(MPI_COMM_WORLD, "boids_output.csv",
                      MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh);
    }else
    {
        // Open the file in append mode
        MPI_File_open(MPI_COMM_WORLD, "boids_output.csv",
                      MPI_MODE_APPEND | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh);
    }
    

    char buffer[100];
    int len;

    for (int i = 0; i < FLOCKSIZE; i++)
    {
        len = snprintf(buffer, sizeof(buffer), "%d,%d,%.2f,%.2f,%.2f,%.2f\n",
                       full_flock[i].id, full_flock[i].timestep,
                       full_flock[i].position.x, full_flock[i].position.y,
                       full_flock[i].velocity.x, full_flock[i].velocity.y);

        MPI_File_write_at(fh, global_offset, buffer, len, MPI_CHAR, MPI_STATUS_IGNORE);
        global_offset += len;
    }

    MPI_File_close(&fh);
}

int main(int argc, char** argv) {
    char *config;
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open the config file collectively in read-only mode.
    read_config(rank, config);



    int local_count = flocksize / size; 

    Boid* full_flock = allocateFullFlock(FLOCKSIZE);
    Boid* local_flock = NULL;
    void* d_states;

    local_flock = createBoids(local_count, rank, &d_states);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // All-to-all exchange so each rank gets the full flock
    MPI_Alltoall(local_flock, local_count * sizeof(Boid), MPI_BYTE,
        full_flock, local_count * sizeof(Boid), MPI_BYTE, MPI_COMM_WORLD);

    if (rank == 0) {
        // print boids in flock 
        // printf("Initial Flock State:\n");
        write_output(full_flock,1,1);
        //for (int i = 0; i < FLOCKSIZE; i++) {
        //    printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
        //        full_flock[i].id, full_flock[i].timestep,
        //        full_flock[i].position.x, full_flock[i].position.y,
        //        full_flock[i].velocity.x, full_flock[i].velocity.y);
        //}
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 1; t < frames; t++) {
        // printf("Rank %d: Updating boids at timestep %d\n", rank, t);
        // printf("Rank %d: All-to-all exchange completed\n", rank);
        updateBoids(local_flock, full_flock, local_count, t);
        // printf("Updated Flock State:\n");

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("Rank %d: Boids updated\n", rank);

        // All-to-all exchange so each rank gets the full flock
        MPI_Alltoall(local_flock, local_count * sizeof(Boid), MPI_BYTE,
                     full_flock, local_count * sizeof(Boid), MPI_BYTE, MPI_COMM_WORLD);
        // printf("Rank %d: All-to-all exchange completed\n", rank);
        // Optionally collect final state at root for printing
        // MPI_Gather(local_flock, local_count * sizeof(Boid), MPI_BYTE, full_flock, local_count * sizeof(Boid), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // printf("Updated Flock State:\n");
            write_output(full_flock,0,0);
            //for (int i = 0; i < FLOCKSIZE; i++) {
            //    printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n",
            //           full_flock[i].id, full_flock[i].timestep,
            //           full_flock[i].position.x, full_flock[i].position.y,
            //           full_flock[i].velocity.x, full_flock[i].velocity.y);
            //}
        }
    }

    cudaFree(local_flock);
    cudaFree(d_states);
    cudaFree(full_flock);
    // free(full_flock);

    MPI_Finalize();
    return 0;
}
