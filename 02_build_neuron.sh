#!/usr/bin/env bash
# Configure, build and install NEURON with CoreNEURON GPU support (OpenACC via nvc++).
# Logs go to logs/nrn-{configure,build,install}.log.
# Pass --clean to wipe the build directory first (needed after changing compilers/CUDA_ARCH).
set -euo pipefail
source "$(dirname "$0")/config.sh"

[ "${1:-}" = "--clean" ] && rm -rf "$BUILD_DIR"
[ -n "$CUDA_ARCH" ] || { echo "CUDA_ARCH not set and no GPU detected" >&2; exit 1; }

export PATH="$NVHPC_ROOT/compilers/bin:$PATH"
[ -n "$MPI_BIN" ] && export PATH="$MPI_BIN:$PATH"
# Don't let a CUDA toolkit on PATH (e.g. /usr/local/cuda) be mixed with NVHPC's.
unset CUDAHOSTCXX CUDA_HOME CUDA_PATH

# NEURON patches (patches/nrn/*.patch) are applied to the src/nrn checkout, idempotently, so
# the tree is always "pinned commit + these files" and never hand-edited. NO_NRN_PATCHES=1
# reverse-applies any that are present. Pristine tree: git -C src/nrn checkout -- .
for p in "$TOP"/patches/nrn/*.patch; do
  [ -e "$p" ] || break
  if git -C "$SRC_DIR/nrn" apply --reverse --check "$p" 2>/dev/null; then
    if [ "${NO_NRN_PATCHES:-0}" = "1" ]; then
      echo "reverting NEURON patch $(basename "$p")"
      git -C "$SRC_DIR/nrn" apply --reverse "$p"
    else
      echo "NEURON patch already applied: $(basename "$p")"
    fi
  elif [ "${NO_NRN_PATCHES:-0}" != "1" ]; then
    echo "applying NEURON patch $(basename "$p")"
    git -C "$SRC_DIR/nrn" apply "$p"
  fi
done

mkdir -p "$BUILD_DIR" "$TOP/logs"
echo "NVHPC: $NVHPC_ROOT | CUDA_ARCH: $CUDA_ARCH | MPI: $(command -v mpicc) | prefix: $PREFIX"

cmake -S "$SRC_DIR/nrn" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_C_COMPILER=nvc \
  -DCMAKE_CXX_COMPILER=nvc++ \
  -DCMAKE_CUDA_COMPILER="$NVHPC_ROOT/compilers/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" \
  -DNRN_ENABLE_CORENEURON=ON \
  -DCORENRN_ENABLE_GPU=ON \
  -DCORENRN_ENABLE_OPENMP=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_TESTS=OFF \
  ${EXTRA_CMAKE_ARGS:-} \
  2>&1 | tee "$TOP/logs/nrn-configure.log"

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" 2>&1 | tee "$TOP/logs/nrn-build.log"
cmake --install "$BUILD_DIR" 2>&1 | tee "$TOP/logs/nrn-install.log" | tail -3
echo "Installed to $PREFIX"
