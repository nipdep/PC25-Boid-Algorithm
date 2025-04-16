#!/bin/bash
module load xl_r spectrum-mpi cuda/11.2

#FLOCK_SIZE=( 1024 8192 65536 524288 )
STRONG_FLOCK_SIZE=( 100 1000 10000 100000)
WEAK_FLOCK_SIZE=( 100 1000 10000 100000)
THREAD_COUNT=(64 128 256 512 1024)
TIME_STEP=10
NP_COUNT=( 1 2 4 )


for k in "${STRONG_FLOCK_SIZE[@]}"; do
    for j in "${THREAD_COUNT[@]}"; do
        for i in "${NP_COUNT[@]}"; do
            echo "|FILEOUT|ALGO MPI_IO|STRONG_SCALE|THREAD_COUNT $j|NP_COUNT $i|BOIDS_COUNT $k|"
            mpirun --bind-to core --report-bindings -np "$i" ./strong_scale/mpi-cuda-exe $k $j $TIME_STEP 0
        done
        echo "Elapsed time: ${SECONDS} seconds"
    done
done

for k in "${WEAK_FLOCK_SIZE[@]}"; do
    for j in "${THREAD_COUNT[@]}"; do
        for i in "${NP_COUNT[@]}"; do
            echo "|FILEOUT|ALGO MPI_IO|WEAK_SCALE|THREAD_COUNT $j|NP_COUNT $i|BOIDS_COUNT $k|"
            mpirun --bind-to core --report-bindings -np "$i" ./weak_scale/mpi-cuda-exe $k $j $TIME_STEP 1
        done

        echo "Elapsed time: ${SECONDS} seconds"
    done
done