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

# NEURON patches (patches/nrn/*.patch) are applied, in order, to the src/nrn checkout, so the
# tree is always "pinned commit + a prefix of this series" and never hand-edited. Which prefix
# is already applied is found by comparing git trees (computed in a scratch index, without
# touching files), and the rest are applied. NRN_PATCHES_UPTO=NN stops after patch NN (and
# reverts later ones if present), e.g. 01 = NVTX only; NO_NRN_PATCHES=1 reverts all of them.
# (Checking patches one by one does not work: later patches change earlier patches' context.)
# Pristine tree: git -C src/nrn checkout -- .
apply_nrn_patches() {
  local nrn="$SRC_DIR/nrn" idx current applied=-1 i
  local patches=("$TOP"/patches/nrn/*.patch)
  [ -e "${patches[0]}" ] || patches=()
  idx="$(mktemp)"
  # tree of the checkout as it is now
  GIT_INDEX_FILE="$idx" git -C "$nrn" read-tree HEAD
  git -C "$nrn" diff --ignore-submodules --binary HEAD | GIT_INDEX_FILE="$idx" git -C "$nrn" apply --cached --allow-empty
  current="$(GIT_INDEX_FILE="$idx" git -C "$nrn" write-tree)"
  # trees of the pinned commit plus the first k patches
  local trees=()
  GIT_INDEX_FILE="$idx" git -C "$nrn" read-tree "$NRN_COMMIT"
  trees+=("$(GIT_INDEX_FILE="$idx" git -C "$nrn" write-tree)")
  for p in "${patches[@]}"; do
    GIT_INDEX_FILE="$idx" git -C "$nrn" apply --cached "$p"
    trees+=("$(GIT_INDEX_FILE="$idx" git -C "$nrn" write-tree)")
  done
  rm -f "$idx"
  for i in "${!trees[@]}"; do [ "${trees[$i]}" = "$current" ] && applied=$i; done
  if [ "$applied" -lt 0 ]; then
    echo "src/nrn is not $NRN_COMMIT plus a prefix of patches/nrn/; refusing to patch it." >&2
    echo "Reset with: git -C $nrn checkout --detach $NRN_COMMIT && git -C $nrn checkout -- ." >&2
    exit 1
  fi
  # how many patches the checkout should end up with
  local target=${#patches[@]} n
  if [ "${NO_NRN_PATCHES:-0}" = "1" ]; then
    target=0
  elif [ -n "${NRN_PATCHES_UPTO:-}" ]; then
    target=0
    for p in "${patches[@]}"; do
      n="$(basename "$p")"; n="${n%%-*}"
      [ "$((10#$n))" -le "$((10#$NRN_PATCHES_UPTO))" ] && target=$((target + 1))
    done
  fi
  for ((i = applied; i > target; i--)); do
    echo "reverting NEURON patch $(basename "${patches[i - 1]}")"
    git -C "$nrn" apply --reverse "${patches[i - 1]}"
  done
  for ((i = 1; i <= target; i++)); do
    if [ "$i" -le "$applied" ]; then
      echo "NEURON patch already applied: $(basename "${patches[i - 1]}")"
    else
      echo "applying NEURON patch $(basename "${patches[i - 1]}")"
      git -C "$nrn" apply "${patches[i - 1]}"
    fi
  done
}
apply_nrn_patches

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
