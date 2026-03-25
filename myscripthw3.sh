#!/bin/bash
module purge
module load intel-mpi
mpiexec -n 1 ./life_mpi 5120 5000 1 out

