# PC25-Boid-Algorithm
Parallel Computing project Boid Algorithm implementation in hyper-paralallized setting

## Sequetial implementaion 
Done in the `boids.c` file \\
To compile
```
gcc boids.c -o boids.exe -lm
```
To run with output to `.csv`
```
./boids.exe > sample_sim.csv
```

## CUDA implementation 
Done in the `boids.cu` file \\
To compile 
```
nvcc boids.cu -o boids_cuda
```

To run with output to `.csv`
```
./boids_cuda > sample_cuda_sim.csv
```

## MPI CUDA implementation
Done in two files `boids_mpi_cuda.c` and `boids_mpi_cuda.cu` \\
To compile
```
make mpi-cuda
```

To run with outpout to `.csv`
```
mpirun -np 2 ./boids_mpi_cuda-exe > sample_mpi_cuda_sim.csv
```
__Note: please be mind full in `Makefile` under the cuda related library versions. 