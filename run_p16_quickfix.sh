#!/bin/bash
#PBS -N life_P16_fix
#PBS -q medium
#PBS -j oe
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=12000mb
#PBS -l walltime=150:00:00
set -euo pipefail
cd "$PBS_O_WORKDIR"
module purge || true
module load openmpi >/dev/null 2>&1 || true
module list 2>&1 || true
SRC="mpi_life1.c"
EXE="life_mpi"
mpicc -O3 -march=native -Wall -Wextra -std=gnu11 -o "$EXE" "$SRC"
mkdir -p out
N=5120
MAX_ITERS=5000
P=16
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
for trial in 1 2 3; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="out/run_N${N}_I${MAX_ITERS}_P${P}_trial${trial}_${stamp}.log"
  mpiexec --mca btl self,tcp --mca pml ob1 -n "$P" "./$EXE" "$N" "$MAX_ITERS" "$P" out |& tee "$log"
done
