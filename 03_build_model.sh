#!/usr/bin/env bash
# Copy the model's sim/ directory to model/, apply patches/*.patch to the copy, and
# compile its mechanisms for NEURON + CoreNEURON (GPU). Produces model/x86_64/special.
# The upstream checkout in src/ is never modified. Skip patches with NO_PATCHES=1.
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
fi

cd "$MODEL_DIR"
nrnivmodl -coreneuron . 2>&1 | tee "$TOP/logs/nrnivmodl.log"
test -x "$MODEL_DIR/x86_64/special"
echo "Built $MODEL_DIR/x86_64/special"
