#!/bin/bash
#PBS -N life_mpi
#PBS -q small
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=2000mb
#PBS -l walltime=60:00:00

set -euo pipefail

cd "$PBS_O_WORKDIR"

echo "=== Host: $(hostname) ==="
echo "=== Workdir: $PBS_O_WORKDIR ==="
echo "=== PBS_NP: ${PBS_NP:-1} ==="

# --- Load an MPI environment (cluster uses Lmod/modules) ---
module purge || true

try_module() {
  local m="$1"
  module load "$m" >/dev/null 2>&1 || return 1
  command -v mpicc >/dev/null 2>&1 || { module unload "$m" >/dev/null 2>&1 || true; return 1; }
  (command -v mpiexec >/dev/null 2>&1 || command -v mpirun >/dev/null 2>&1) || { module unload "$m" >/dev/null 2>&1 || true; return 1; }
  echo "Loaded MPI module: $m"
  return 0
}

# Try common MPI module names. Add/remove entries if your module list differs.
MPI_MODULE=""
for m in \
  "openmpi" "OpenMPI" "openmpi/4.1.5" "openmpi/4.1.6" \
  "mpich" "MPICH" "mpich/4.2.1" \
  "intelmpi" "intel-mpi" "intel_mpi" "impi" \
  "intel/mpi" "intel/2025.3.1" "intel" \
  "mpi" \
; do
  if try_module "$m"; then
    MPI_MODULE="$m"
    break
  fi
done

echo "=== module list ==="
module list 2>&1 || true

echo "=== toolchain ==="
which mpicc || true
which mpiexec || true
which mpirun || true
mpicc --version || true

# If no MPI module loaded, fail with a useful message.
if ! command -v mpicc >/dev/null 2>&1; then
  echo "ERROR: mpicc not found. Run 'module avail mpi' and add the correct module name to the script."
  exit 2
fi
if ! (command -v mpiexec >/dev/null 2>&1 || command -v mpirun >/dev/null 2>&1); then
  echo "ERROR: mpiexec/mpirun not found after loading modules."
  exit 3
fi

# --- Build (rebuild inside job so runtime libs match the loaded MPI) ---
SRC="mpi_life1.c"        # <-- change if your file name differs
EXE="life_mpi"

echo "=== building $EXE from $SRC ==="
mpicc -O3 -march=native -Wall -Wextra -std=gnu11 -o "$EXE" "$SRC"

echo "=== ldd (mpi libs) ==="
ldd "./$EXE" | egrep 'mpi|mpifort|open-rte|open-pal|mpich|intel' || true

mkdir -p out

# --- Run ---
P="${PBS_NP:-1}"   # number of ranks allocated by PBS
N=5120
MAX_ITERS=5000

RUNNER=""
if command -v mpiexec >/dev/null 2>&1; then
  RUNNER="mpiexec"
elif command -v mpirun >/dev/null 2>&1; then
  RUNNER="mpirun"
fi

echo "=== running with $RUNNER, ranks=$P ==="
if [ "$RUNNER" = "mpiexec" ]; then
  mpiexec -n "$P" "./$EXE" "$N" "$MAX_ITERS" "$P" out
else
  mpirun -np "$P" "./$EXE" "$N" "$MAX_ITERS" "$P" out
fi

echo "=== done ==="
ls -l out || true
