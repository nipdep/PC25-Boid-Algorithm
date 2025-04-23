mpi-cuda-local: boids_mpi_cuda.c boids_mpi_cuda.cu boids.h
	mpicc  -O3 -I. boids_mpi_cuda.c -c -o boids_mpi_cuda-xlc.o
	nvcc   -O3 -I. -arch=sm_61 boids_mpi_cuda.cu -c -o boids_mpi_cuda-nvcc.o 
	mpicc  -O3 boids_mpi_cuda-xlc.o boids_mpi_cuda-nvcc.o -o boids_mpi_cuda-exe \
	       -L/usr/local/cuda-12.4/lib64/ -lcudadevrt -lcudart -lstdc++

mpi-cuda: boids_mpi_cuda.c boids_mpi_cuda.cu boids.h
	mpixlc -O3 boids_mpi_cuda.c -c -o boids-mpi-cuda-xlc.o
	nvcc -O3 -arch=sm_70 boids_mpi_cuda.cu -c -o boids-mpi-cuda-nvcc.o    
	mpixlc -O3 boids-mpi-cuda-xlc.o boids-mpi-cuda-nvcc.o -o mpi-cuda-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lcurand -lstdc++

