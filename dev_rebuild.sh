#!/usr/bin/env bash
# Incremental rebuild after editing NEURON or CoreNEURON sources in src/nrn.
#
# Why this exists instead of re-running 02+03: CoreNEURON is a *static* library
# (install/lib/libcoreneuron-core.a), so it gets linked into the model's mechanism library.
# Editing CoreNEURON therefore requires re-running nrnivmodl -- rebuilding NEURON alone is
# not enough, and a stale model/x86_64/special will silently keep running the old code.
#
# But nrnivmodl is make-based (`make -j 4 -f nrnmech_makefile ... special`, no rm -rf), so
# re-running it *in place* only relinks: the ~30 .mod -> .cpp -> .o compiles are reused.
# 03_build_model.sh starts with `rm -rf "$MODEL_DIR"`, which throws that away and costs
# several minutes every time. This script keeps the tree and lets make do its job.
#
# Usage: ./dev_rebuild.sh [--mods] [--verify]
#   (default)  build+install NEURON/CoreNEURON, then relink the mechanism library
#   --mods     also force .mod files to be regenerated (use when you edited a .mod, or
#              changed NMODL/mod2c code generation)
#   --verify   finish by running ./bench.sh verify, which must print IDENTICAL
#
# After changing compilers or CUDA_ARCH, use ./02_build_neuron.sh --clean instead.
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
source "$TOP/env.sh"

MODS=0 VERIFY=0
for a in "$@"; do
  case $a in
    --mods) MODS=1 ;;
    --verify) VERIFY=1 ;;
    *) sed -n '2,22p' "$0"; exit 1 ;;
  esac
done

[ -d "$BUILD_DIR" ] || { echo "No build dir at $BUILD_DIR -- run ./02_build_neuron.sh first" >&2; exit 1; }
[ -d "$TOP/model" ] || { echo "No model dir -- run ./03_build_model.sh first" >&2; exit 1; }

echo "=== 1/2 NEURON + CoreNEURON (incremental) ==="
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
cmake --install "$BUILD_DIR" > "$TOP/logs/dev-install.log"

echo "=== 2/2 relink model mechanisms (incremental) ==="
cd "$TOP/model"
# The model dir is a symlink farm plus the generated x86_64 tree; keep both.
if [ "$MODS" = 1 ]; then
  echo "    (--mods: dropping generated .cpp/.o so mod files are regenerated)"
  rm -rf x86_64
fi
nrnivmodl -coreneuron . 2>&1 | tee "$TOP/logs/dev-nrnivmodl.log" | tail -5
test -x "$TOP/model/x86_64/special"

echo
echo "Rebuilt $TOP/model/x86_64/special"
ls -l --time-style=+%H:%M:%S "$TOP/model/x86_64/special" "$PREFIX/lib/libcoreneuron-core.a"

if [ "$VERIFY" = 1 ]; then
  echo
  echo "=== verify (must print IDENTICAL) ==="
  cd "$TOP" && ./bench.sh verify
fi
