#!/usr/bin/env bash
# Benchmark sweep. Each configuration is a separate ./run_bulb.sh call; results go to
# runs/ and are tabulated by summarize_runs.py into runs/summary.csv.
#
# Usage: ./bench.sh [verify|quick|full]
#   verify: NEURON CI test case (1 glomerulus, 50 ms, 4 ranks) on neuron/cpu/gpu,
#           then checks all three spike outputs are identical.
#   quick : bulb3dtest.py's 5 glomeruli, 50 ms, GPU at 1/2/4 ranks + CPU at all cores.
#   full  : whole bulb (127 glomeruli), model default 1050 ms, GPU + CPU.
# Override lists via env: NPS_GPU="1 2 4", NPS_CPU="16", GLOMS=..., TSTOP=...
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
cd "$TOP"
mkdir -p runs

SUITE="${1:-quick}"
NCORES="$(nproc)"

new_runs() { find runs -mindepth 1 -maxdepth 1 -type d | sort; }

case $SUITE in
  verify)
    before="$(new_runs)"
    for m in neuron cpu gpu; do ./run_bulb.sh -m $m -n 4 -t 50 -g 5 > /dev/null; done
    ./compare_spikes.sh $(comm -13 <(echo "$before") <(new_runs))
    exit
    ;;
  quick) GLOMS="${GLOMS:-5,37,32,78,7}" TSTOP="${TSTOP:-50}" ;;
  full)  GLOMS="${GLOMS:-all}"          TSTOP="${TSTOP:-1050}" ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac

before="$(new_runs)"
for np in ${NPS_GPU:-1 2 4}; do
  echo ">> gpu np=$np gloms=$GLOMS tstop=$TSTOP"
  ./run_bulb.sh -m gpu -n "$np" -t "$TSTOP" -g "$GLOMS" > /dev/null || echo "   FAILED"
done
for np in ${NPS_CPU:-$NCORES}; do
  echo ">> cpu np=$np gloms=$GLOMS tstop=$TSTOP"
  ./run_bulb.sh -m cpu -n "$np" -t "$TSTOP" -g "$GLOMS" > /dev/null || echo "   FAILED"
done
./summarize_runs.py $(comm -13 <(echo "$before") <(new_runs))
