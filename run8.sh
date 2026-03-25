#!/bin/bash
#PBS -N life_P8
#PBS -q medium
#PBS -j oe
#PBS -l select=1:ncpus=8:mpiprocs=8:mem=8000mb
#PBS -l walltime=150:00:00

set -euo pipefail
cd "$PBS_O_WORKDIR"

module purge || true
module load openmpi/4.1.7-gcc12

mpicc -O3 -Wall -Wextra -std=c11 -o life_mpi mpi_life1.c

mkdir -p out
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

N=5120
ITERS=5000
P=8

for t in 1 2 3; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  mpiexec --mca btl self,tcp --mca pml ob1 -n "$P" ./life_mpi "$N" "$ITERS" "$P" out |& tee "out/p${P}_trial${t}_${stamp}.log"
done
