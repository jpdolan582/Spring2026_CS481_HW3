#!/bin/bash
#PBS -N life_P1
#PBS -q small
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=2000mb
#PBS -l walltime=60:00:00

set -euo pipefail
cd "$PBS_O_WORKDIR"

module purge || true
try_module() {
  local m="$1"
  module load "$m" >/dev/null 2>&1 || return 1
  command -v mpicc >/dev/null 2>&1 || { module unload "$m" >/dev/null 2>&1 || true; return 1; }
  (command -v mpiexec >/dev/null 2>&1 || command -v mpirun >/dev/null 2>&1) || { module unload "$m" >/dev/null 2>&1 || true; return 1; }
  echo "Loaded MPI module: $m"
  return 0
}
for m in openmpi OpenMPI openmpi/4.1.5 openmpi/4.1.6 mpich MPICH mpich/4.2.1 intelmpi intel-mpi intel_mpi impi intel/mpi intel/2025.3.1 intel mpi; do
  try_module "$m" && break
done

SRC="mpi_life1.c"
EXE="life_mpi"
mpicc -O3 -march=native -Wall -Wextra -std=gnu11 -o "$EXE" "$SRC"

mkdir -p out
N=5120
MAX_ITERS=5000
P=1

RUNNER="mpiexec"
command -v mpiexec >/dev/null 2>&1 || RUNNER="mpirun"

for trial in 1 2 3; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="out/run_N${N}_I${MAX_ITERS}_P${P}_trial${trial}_${stamp}.log"
  echo "=== P=$P trial=$trial ==="
  if [ "$RUNNER" = "mpiexec" ]; then
    mpiexec -n "$P" "./$EXE" "$N" "$MAX_ITERS" "$P" out |& tee "$log"
  else
    mpirun -np "$P" "./$EXE" "$N" "$MAX_ITERS" "$P" out |& tee "$log"
  fi
done
