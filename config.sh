# config.sh -- every machine-specific knob lives here.
# Sourced by all other scripts. Override any variable from the environment, e.g.
#   CUDA_ARCH=90 NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7 ./02_build_neuron.sh

# Root of this setup (directory containing this file).
TOP="${TOP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---- Source versions (pinned for reproducibility) ---------------------------
NRN_REPO="${NRN_REPO:-https://github.com/neuronsimulator/nrn.git}"
NRN_COMMIT="${NRN_COMMIT:-53de154d61300f3ca74d84ffa61ad4093f002bfd}"   # master 2026-09-15 (9.0.1-115)
OB_REPO="${OB_REPO:-https://github.com/HumanBrainProject/olfactory-bulb-3d.git}"
OB_COMMIT="${OB_COMMIT:-1517ecb97c610ad26409570e2fbfd0b135eb2b86}"      # master; same code as NEURON's CI pin b07b76dc

# ---- Toolchain ---------------------------------------------------------------
# NVIDIA HPC SDK (provides nvc/nvc++/nvcc; needed for the OpenACC GPU backend).
NVHPC_ROOT="${NVHPC_ROOT:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.7}"
# Directory containing mpicc/mpicxx/mpirun. Empty = whatever is first on PATH
# (here: Ubuntu's OpenMPI 4.1.6). Alternative: "$NVHPC_ROOT/comm_libs/mpi/bin" (HPC-X).
MPI_BIN="${MPI_BIN:-}"
# Python used to create the venv (>= 3.10).
PYTHON_BASE="${PYTHON_BASE:-python3}"

# ---- GPU ---------------------------------------------------------------------
# Compute capability without the dot: 89 = RTX 4090, 80 = A100, 90 = H100/GH200, 100 = B200.
# Default: auto-detect GPU 0. Semicolon-separate to build for several, e.g. "80;90".
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')}"

# ---- Layout ------------------------------------------------------------------
SRC_DIR="${SRC_DIR:-$TOP/src}"
BUILD_DIR="${BUILD_DIR:-$TOP/build/nrn}"
PREFIX="${PREFIX:-$TOP/install}"
VENV="${VENV:-$TOP/venv}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

# ---- Launch ------------------------------------------------------------------
# MPI launcher prefix; the rank count is appended. OpenMPI default below. Examples:
#   Slurm:        MPI_LAUNCH="srun --mpi=pmix -n"
#   MPICH/HPC-X:  MPI_LAUNCH="mpiexec -n"
MPI_LAUNCH="${MPI_LAUNCH:-mpirun --oversubscribe -x OMP_NUM_THREADS=1 -x PYTHONPATH -x HWLOC_COMPONENTS -x OB_NEIGHBOUR_CACHE -np}"
