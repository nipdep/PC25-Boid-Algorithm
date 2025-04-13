// Updated boids_mpi_cuda.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include "boids.h"
#include "mclock.h"

int flocksize;
int frames;
int threadCount;
int char_buffer_size = 50;
char *output_file = "boids_output.csv";

// Declare external GPU functions
extern Boid *createBoids(int size, int rank, void **d_states, int thread_count);
extern Boid *allocateFullFlock(int size); // now allocated in .cu
extern void updateBoids(Boid *local_flock, Boid *full_flock, int local_size, int timestep, int thread_count);

void read_config(int argc, char *argv[], int rank, int size)
{
    // Check if the number of arguments is less than expected
    if (argc < 4)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: mpirun -np <n> ./your_mpi_executable <parameter>\n");
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    flocksize = atoi(argv[1]);
    threadCount = atoi(argv[2]);
    frames = atoi(argv[3]);

    int numChars = snprintf(NULL, 0, "%d_%d_", size, threadCount);
    size_t originalLength = strlen(output_file);
    char *file_type;

        if (atoi(argv[4]) == 0)
    {
        file_type = "strong";
        numChars += 6;
    }else
    {
        file_type = "weak";
        numChars += 4;
    }

    size_t totalLength = numChars + 1 + originalLength + 1;
    char *newStr = malloc(totalLength);
    sprintf(newStr, "%d_%d_%s_%s", size, threadCount,file_type, output_file);
    output_file = newStr;
    if (rank == 0)
    {
        printf("Output file: %s\n", output_file);
    }
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
    MPI_Offset offset = char_buffer_size * frame_no * size * flock_count + rank * char_buffer_size * flock_count;
    int len;

    for (int i = 0; i < flock_count; i++)
    {
        len = snprintf(buffer, sizeof(buffer), "%d,%d,%.2f,%.2f,%.2f,%.2f",
                       full_flock[i].id, full_flock[i].timestep,
                       full_flock[i].position.x, full_flock[i].position.y,
                       full_flock[i].velocity.x, full_flock[i].velocity.y);

        for (int j = len; j < char_buffer_size - 1; j++)
        {
            buffer[j] = ' ';
        }
        buffer[char_buffer_size - 1] = '\n'; // Ensure null termination

        MPI_File_write_at(fh, offset, buffer, char_buffer_size, MPI_CHAR, MPI_STATUS_IGNORE);
        offset += char_buffer_size;
    }
}

void assign_boids_values(Boid *full_flock, int flock_count, char *buffer)
{
    char *line = strtok(buffer, "\n");

    for (int i = 0; i < flock_count; i++)
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

void read_output(Boid *full_flock, int flock_count, MPI_File fh, int frame_no)
{
    MPI_Status status;
    MPI_Offset file_size;
    MPI_Offset offset;

    file_size = flock_count * char_buffer_size * sizeof(char);
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

int main(int argc, char *argv[])
{
    int rank, size;

    uint64_t start_p, end_p, start_cuda, end_cuda, start_io, end_io;
    start_p = clock_now();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open the config file collectively in read-only mode.
    read_config(argc, argv, rank, size);

    MPI_File fh = create_file(output_file);
    int local_count = flocksize;

    Boid *full_flock = allocateFullFlock(flocksize);
    Boid *local_flock = NULL;
    void *d_states;

    local_flock = createBoids(local_count, rank, &d_states, threadCount);

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // output the initial state of the boids
    start_io = clock_now();
    write_output(local_flock, local_count, fh, 0, size, rank);
    end_io = clock_now();
    if (rank == 0)
    {
        printf("|FILEOUT|IO_WRITE|TS %d|SIZE %d| time taken: %f seconds, clock cycles: %lu\n", 0, size, (double)(end_io - start_io) / 512000000.0, end_io - start_io);
    }

    // All-to-all exchange, Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    start_io = clock_now();
    read_output(full_flock, flocksize, fh, 0);
    end_io = clock_now();
    if (rank == 0)
    {
        printf("|FILEOUT|IO_READ|TS %d|SIZE %d| time taken: %f seconds, clock cycles: %lu\n",0, size, (double)(end_io - start_io) / 512000000.0, end_io - start_io);
    }

    for (int t = 1; t < frames; t++)
    {
        // time step update.
        start_cuda = clock_now();
        updateBoids(local_flock, full_flock, local_count, t, threadCount);
        end_cuda = clock_now();
        if (rank == 0)
        {
            printf("|FILEOUT|CUDA|TS %d|SIZE %d| time taken: %f seconds, clock cycles: %lu\n", t, size, (double)(end_cuda - start_cuda) / 512000000.0, end_cuda - start_cuda);
        }
        // printf("Updated Flock State:\n");

        start_io = clock_now();
        write_output(local_flock, local_count, fh, t, size, rank);
        end_io = clock_now();
        if (rank == 0)
        {
            printf("|FILEOUT|IO_WRITE|TS %d|SIZE %d| time taken: %f seconds, clock cycles: %lu\n", t, size, (double)(end_io - start_io) / 512000000.0, end_io - start_io);
        }

        // All-to-all exchange, Synchronize all processes

        MPI_Barrier(MPI_COMM_WORLD);

        start_io = clock_now();
        read_output(full_flock, flocksize, fh, t);
        end_io = clock_now();
        if (rank == 0)
        {
            printf("|FILEOUT|IO_READ|TS %d|SIZE %d| time taken: %f seconds, clock cycles: %lu\n", t, size, (double)(end_io - start_io) / 512000000.0, end_io - start_io);
        }
    }

    end_p = clock_now();
    if (rank == 0)
    {
        printf("|FILEOUT|MPI|SIZE %d| time taken: %f seconds, clock cycles: %lu\n", size, (double)(end_p - start_p) / 512000000.0, end_p - start_p);
    }

    close_file(&fh);
    cudaFree(d_states);
    cudaFree(full_flock);
    free(output_file);

    MPI_Finalize();
    return 0;
}
