#!/usr/bin/env bash
# Copy the model's sim/ directory to model/, apply patches/*.patch to the copy, and
# compile its mechanisms for NEURON + CoreNEURON (GPU). Produces model/x86_64/special.
# The upstream checkout in src/ is never modified. Skip patches with NO_PATCHES=1.
# Opt-in patches from patches/optional/ are applied only when named in EXTRA_PATCHES,
# e.g. EXTRA_PATCHES=02-sample-without-materializing ./03_build_model.sh (or =all).
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
source "$TOP/env.sh"

MODEL_DIR="$TOP/model"
rm -rf "$MODEL_DIR"
cp -r "$SRC_DIR/olfactory-bulb-3d/sim" "$MODEL_DIR"
cp "$TOP/bulb_bench.py" "$TOP/profile_setup.py" "$MODEL_DIR/"

if [ "${NO_PATCHES:-0}" != "1" ]; then
  for p in "$TOP"/patches/*.patch; do
    [ -e "$p" ] || break
    echo "applying $(basename "$p")"
    patch -p1 -d "$MODEL_DIR" < "$p"
  done
  if [ "${EXTRA_PATCHES:-}" = "all" ]; then
    set -- "$TOP"/patches/optional/*.patch
  else
    set -- ${EXTRA_PATCHES:+$(for n in $EXTRA_PATCHES; do echo "$TOP/patches/optional/$n.patch"; done)}
  fi
  for p in "$@"; do
    [ -e "$p" ] || { echo "no such optional patch: $p" >&2; exit 1; }
    echo "applying optional $(basename "$p")"
    patch -p1 -d "$MODEL_DIR" < "$p"
  done
fi

cd "$MODEL_DIR"
# NVHPC gives each object a CUDA module ID made of a header path plus a small random number.
# With ~20 mechanism objects two occasionally collide, and the device link then fails with
# "redefinition of __cudaRegisterLinkedBinary_...". Recompiling draws new IDs, so retry.
for attempt in 1 2 3; do
  if nrnivmodl -coreneuron . 2>&1 | tee "$TOP/logs/nrnivmodl.log"; then
    break
  fi
  if [ "$attempt" -lt 3 ] && grep -q "redefinition of .__cudaRegisterLinkedBinary" "$TOP/logs/nrnivmodl.log"; then
    echo "CUDA module-ID collision in the device link (NVHPC); rebuilding mechanisms, attempt $((attempt + 1))"
    rm -rf x86_64
  else
    exit 1
  fi
done
test -x "$MODEL_DIR/x86_64/special"
echo "Built $MODEL_DIR/x86_64/special"
