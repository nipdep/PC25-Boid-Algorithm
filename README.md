# PC25-Boid: Comparative Evaluation of MPI Communication Models in GPU-Accelerated Boids Simulation.
Parallel Computing project Boid Algorithm implementation in hyper-parallelized setting. As stated in the paper we implement three algorithmic variants for different inter process communication

## Structure of the Code

```plaintext
boids-mpi-cuda/
├── 📂 MPI_Allgather/             # MPI_Allgather algorithm variant
│   ├── 📂 strong_scale/          # Files for strong scaling experiments
│   │   ├── 📄 boids_mpi_cuda.c   # main c file with MPI implementation
│   │   ├── 📄 boids_mpi_cuda.cu  # main cu file with CUDA implementation
│   │   ├── 📄 boids.h            # contains the boids structs that are used.
│   │   ├── 📄 mclock.h           # contains the clock implementation
│   │   └── 📄 Makefile           # Make file to build the folder implementation
│   │
│   ├── 📂 weak_scale/            # Files for weak scaling experiments
│   │   └── ***                   # same structure as above 
│   ├── 📄 run_n1.sh              # Bash file to run on 1 node (np 1,2,4)
│   └── 📄 run_n2.sh              # Bash file to run on 2 node (np 8)
|   └── 📄 run_n4.sh              # Bash file to run on 4 node (np 16)
│
├── 📂 MPI_Alltoall/              # MPI_Alltoall algorithm variant
│   └── ***      
│                 
├── 📂 MPI_IO/                    # MPI_IO algorithm variant
│   └── ***   
│       # Config parameters
├── 📂 results/                   # Contains the slurm job outputs
│   ├── 📂 MPI_Allgather/        
│   ├── 📂 MPI_Alltoall/
│   └── 📂 MPI_IO/
│
├── 📄 BoidsAnim2_Updated.html    # Visualizer
└── 📄 README.md                  # Project documentation

```

### Please Note: All the experimentation was done on AiMOS thus the mclock.h contains the timing function for the super computer, which will give error if run on local computer.

## Run Instructions.

### Run on AiMOS.

1. copy the code folder to the AiMOS super computer
2. log into the super computer and then the `dcsfen01` node
3. load MPI and CUDA
```terminal
module load xl_r spectrum-mpi cuda/11.2
```
4. Go to the folder location of the required variant
```terminal
cd MPI_Allgather/strong_scale
```
5. Use the Makefile to build the code.
```terminal
make mpi-cuda
```
6. For a single run, $NP is the number of processors, $NUM_BOIDS is the number of boids (eg: 4096), $THREAD_COUNT is the number of threads per block, $TIME_STEP number of time steps the algo is run, $SCALE_TYPE is either 0 (for strong scale) or 1 (for weak scale) this is only used for printing purposes:
```terminal
mpirun --bind-to core --report-bindings -np $NP ./strong_scale/mpi-cuda-exe $NUM_BOIDS $THREAD_COUNT $TIME_STEP $SCALE_TYPE
        
```
Example (for a strong scaling simulation running on 4 processes with 4096 boids using 512 threads per block for 30 frames):
```terminal
mpirun --bind-to core --report-bindings -np 4 ./strong_scale/mpi-cuda-exe 4096 512 30 0
        
```

7. For a batch run, utilize the given bash scripts (when doing this be careful of the directory you are in this should be called in the directory with the bash file):
```dotnetcli
sbatch -N 4 --partition=el8 --gres=gpu:4 -t 30 run_n4.sh
```
