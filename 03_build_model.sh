#!/usr/bin/env bash
# Copy the model's sim/ directory to model/ and compile its mechanisms for
# NEURON + CoreNEURON (GPU). Produces model/x86_64/special.
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
source "$TOP/env.sh"

MODEL_DIR="$TOP/model"
rm -rf "$MODEL_DIR"
cp -r "$SRC_DIR/olfactory-bulb-3d/sim" "$MODEL_DIR"
cp "$TOP/bulb_bench.py" "$MODEL_DIR/"

cd "$MODEL_DIR"
nrnivmodl -coreneuron . 2>&1 | tee "$TOP/logs/nrnivmodl.log"
test -x "$MODEL_DIR/x86_64/special"
echo "Built $MODEL_DIR/x86_64/special"
