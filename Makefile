mpi-cuda: boids_mpi_cuda.c boids_mpi_cuda.cu boids.h
	mpicc  -O3 -I. boids_mpi_cuda.c -c -o boids_mpi_cuda-xlc.o
	nvcc   -O3 -I. -arch=sm_61 boids_mpi_cuda.cu -c -o boids_mpi_cuda-nvcc.o 
	mpicc  -O3 boids_mpi_cuda-xlc.o boids_mpi_cuda-nvcc.o -o boids_mpi_cuda-exe \
	       -L/usr/local/cuda-12.4/lib64/ -lcudadevrt -lcudart -lstdc++


mpi-cuda-update: boids_mpi_cuda_updated.c boids_mpi_cuda_updated.cu
	mpicc  -O3 boids_mpi_cuda_updated.c -c -o boids_mpi_cuda_updated-xlc.o
	nvcc -O3 -arch=sm_61 boids_mpi_cuda_updated.cu -c -o boids_mpi_cuda_updated-nvcc.o 
	mpicc  -O3 boids_mpi_cuda_updated-xlc.o boids_mpi_cuda_updated-nvcc.o -o boids_mpi_cuda_updated-exe -L/usr/local/cuda-12.4/lib64/ -lcudadevrt -lcudart -lstdc++