#!/bin/bash
module load xl_r spectrum-mpi cuda/11.2

FLOCK_SIZE=524288
THREAD_COUNT=(64 128 256 512 1024)
TIME_STEP=60
NP_COUNT=(1 2 4)

for j in "${THREAD_COUNT[@]}"; do
    for i in "${NP_COUNT[@]}"; do
        echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT $j|NP_COUNT $i|"
        mpirun --bind-to core --report-bindings -np "$i" ./strong_scale/mpi-cuda-exe $FLOCK_SIZE $j $TIME_STEP 0
    done

    for i in "${NP_COUNT[@]}"; do
        echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT $j|NP_COUNT $i|"
        mpirun --bind-to core --report-bindings -np "$i" ./weak_scale/mpi-cuda-exe $FLOCK_SIZE $j $TIME_STEP 1
    done

    echo "Elapsed time: ${SECONDS} seconds"
done



#echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT 64|";
#mpirun --bind-to core --report-bindings -np 1 ./strong_scale/mpi-cuda-exe 1048576 64 60 0
#mpirun --bind-to core --report-bindings -np 2 ./strong_scale/mpi-cuda-exe 1048576 64 60 0
#mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 1048576 64 60 0

#echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT 64|";
#mpirun --bind-to core --report-bindings -np 1 ./weak_scale/mpi-cuda-exe 1048576 64 60 1
#mpirun --bind-to core --report-bindings -np 2 ./weak_scale/mpi-cuda-exe 1048576 64 60 1
#mpirun --bind-to core --report-bindings -np 4 ./weak_scale/mpi-cuda-exe 1048576 64 60 1

#echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT 128|";
#mpirun --bind-to core --report-bindings -np 1 ./strong_scale/mpi-cuda-exe 1048576 128 60 0
#mpirun --bind-to core --report-bindings -np 2 ./strong_scale/mpi-cuda-exe 1048576 128 60 0
#mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 1048576 128 60 0

#echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT 128|";
#mpirun --bind-to core --report-bindings -np 1 ./weak_scale/mpi-cuda-exe 1048576 128 60 1
#mpirun --bind-to core --report-bindings -np 2 ./weak_scale/mpi-cuda-exe 1048576 128 60 1
#mpirun --bind-to core --report-bindings -np 4 ./weak_scale/mpi-cuda-exe 1048576 128 60 1

#echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT 256|";
#mpirun --bind-to core --report-bindings -np 1 ./strong_scale/mpi-cuda-exe 1048576 256 60 0
#mpirun --bind-to core --report-bindings -np 2 ./strong_scale/mpi-cuda-exe 1048576 256 60 0
#mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 1048576 256 60 0

:<<'EOF'
echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT 256|";
mpirun --bind-to core --report-bindings -np 1 ./weak_scale/mpi-cuda-exe 1048576 256 60 1
mpirun --bind-to core --report-bindings -np 2 ./weak_scale/mpi-cuda-exe 1048576 256 60 1
mpirun --bind-to core --report-bindings -np 4 ./weak_scale/mpi-cuda-exe 1048576 256 60 1

echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT 512|";
mpirun --bind-to core --report-bindings -np 1 ./strong_scale/mpi-cuda-exe 1048576 512 60 0
mpirun --bind-to core --report-bindings -np 2 ./strong_scale/mpi-cuda-exe 1048576 512 60 0
mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 1048576 512 60 0

echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT 512|";
mpirun --bind-to core --report-bindings -np 1 ./weak_scale/mpi-cuda-exe 1048576 512 60 1
mpirun --bind-to core --report-bindings -np 2 ./weak_scale/mpi-cuda-exe 1048576 512 60 1
mpirun --bind-to core --report-bindings -np 4 ./weak_scale/mpi-cuda-exe 1048576 512 60 1

echo "|FILEOUT|MPI_IO|STRONG_SCALE|THREAD_COUNT 1024|";
mpirun --bind-to core --report-bindings -np 1 ./strong_scale/mpi-cuda-exe 1048576 1024 60 0
mpirun --bind-to core --report-bindings -np 2 ./strong_scale/mpi-cuda-exe 1048576 1024 60 0
mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 1048576 1024 60 0

echo "|FILEOUT|MPI_IO|WEAK_SCALE|THREAD_COUNT 1024|";
mpirun --bind-to core --report-bindings -np 1 ./weak_scale/mpi-cuda-exe 1048576 1024 60 1
mpirun --bind-to core --report-bindings -np 2 ./weak_scale/mpi-cuda-exe 1048576 1024 60 1
mpirun --bind-to core --report-bindings -np 4 ./weak_scale/mpi-cuda-exe 1048576 1024 60 1