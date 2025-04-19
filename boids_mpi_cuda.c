// Updated boids_mpi_cuda.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include "boids.h"

int flocksize = 20;
int frames = 30;
int char_buffer_size = 50;
int* length = NULL;

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

MPI_File create_file(const char *filename)
{
    MPI_File fh;
    MPI_File_delete(filename, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, filename,
                  MPI_MODE_CREATE | MPI_MODE_RDWR,
                  MPI_INFO_NULL, &fh);
    return fh;
}

void close_file(MPI_File *fh)
{
    MPI_File_close(fh);
}

void write_output(Boid *full_flock, int flock_count, MPI_File fh, int frame_no, int size, int rank)
{ 

    char buffer[char_buffer_size];
    MPI_Offset offset = char_buffer_size * frame_no*size*flock_count+ rank*char_buffer_size*flock_count;
    int len;

    for (int i = 0; i < flock_count; i++)
    {
        len = snprintf(buffer, sizeof(buffer), "%d,%d,%.2f,%.2f,%.2f,%.2f",
                       full_flock[i].id, full_flock[i].timestep,
                       full_flock[i].position.x, full_flock[i].position.y,
                       full_flock[i].velocity.x, full_flock[i].velocity.y);

        for (int j = len; j < char_buffer_size-1; j++)
        {
            buffer[j] = ' ';
        }
        buffer[char_buffer_size-1] = '\n'; // Ensure null termination

        MPI_File_write_at(fh, offset, buffer, char_buffer_size, MPI_CHAR, MPI_STATUS_IGNORE);
        offset += char_buffer_size;
    }
}

void assign_boids_values(Boid *full_flock, int flock_count, char *buffer)
{   
    char *line = strtok(buffer, "\n");

    for (int i=0; i < flock_count; i++)
    {
        if (line == NULL)
        {
            printf("Error in assignment @ %d: %s\n", i, line);
        }

        sscanf(line, "%d,%d,%f,%f,%f,%f",
               &full_flock[i].id, &full_flock[i].timestep,
               &full_flock[i].position.x, &full_flock[i].position.y,
               &full_flock[i].velocity.x, &full_flock[i].velocity.y);

	line = strtok(NULL, "\n");
    }
}

void read_output(Boid *full_flock, int flock_count,  MPI_File fh, int frame_no)
{
    MPI_Status status;
    MPI_Offset file_size;
    MPI_Offset offset;

    file_size = flock_count*char_buffer_size* sizeof(char);
    offset = char_buffer_size * frame_no * flock_count;
    // Allocate a buffer to hold the file contents plus a null terminator.
    char *buffer = (char *)malloc((file_size + 1) * sizeof(char));
    if (!buffer)
    {
        fprintf(stderr, "Memory allocation error\n");
        MPI_File_close(&fh);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Read the file content collectively into the buffer.
    MPI_File_read_at(fh, offset, buffer, file_size, MPI_CHAR, &status);
    assign_boids_values(full_flock, flock_count, buffer);
}

int main(int argc, char** argv) {
    char *config;
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File fh = create_file("boids_output.csv");

    // Open the config file collectively in read-only mode.
    read_config(rank, config);

    int local_count = flocksize / size; 

    Boid* full_flock = allocateFullFlock(flocksize);
    Boid* local_flock = NULL;
    void* d_states;

    local_flock = createBoids(local_count, rank, &d_states);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    //output the initial state of the boids
    write_output(local_flock, local_count, fh, 0, size, rank);

    // All-to-all exchange, Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    read_output(full_flock, flocksize, fh, 0);

    for (int t = 1; t < frames; t++) {
        // time step update.
        updateBoids(local_flock, full_flock, local_count, t);
        // printf("Updated Flock State:\n");
	
	    write_output(local_flock, local_count, fh, t, size, rank);

        // All-to-all exchange, Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
	    read_output(full_flock, flocksize, fh, t);
    }

    close_file(&fh);
    cudaFree(d_states);
    cudaFree(full_flock);

    MPI_Finalize();
    return 0;
}
