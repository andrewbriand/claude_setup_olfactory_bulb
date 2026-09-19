#!/usr/bin/env bash
# Run the olfactory bulb model. Each run gets its own directory under runs/
# containing the full log and spike output.
#
# Usage: ./run_bulb.sh [-m neuron|cpu|gpu] [-n RANKS] [-t TSTOP_MS] [-g GLOMS] [-- extra model args]
#   -m  neuron = plain NEURON on CPU
#       cpu    = CoreNEURON on CPU
#       gpu    = CoreNEURON on GPU            (default)
#   -n  MPI ranks (default 4). With -m gpu all ranks share the visible GPU(s);
#       ranks are assigned to GPUs round-robin by CoreNEURON.
#   -t  simulated time in ms (default 1050, the model's default)
#   -g  glomeruli: "5", "5,37,32,78,7" (default), "first:N", or "all" (127)
#
# Examples:
#   ./run_bulb.sh -m gpu -n 4 -t 50 -g 5          # quick smoke test (NEURON CI size)
#   ./run_bulb.sh -m gpu -n 8 -g all              # full-size benchmark
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
source "$TOP/env.sh"

MODE=gpu NP=4 TSTOP=1050 GLOMS=5,37,32,78,7
while getopts "m:n:t:g:h" o; do
  case $o in
    m) MODE=$OPTARG ;; n) NP=$OPTARG ;; t) TSTOP=$OPTARG ;; g) GLOMS=$OPTARG ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

case $MODE in
  neuron) MODE_ARGS="" ;;
  cpu)    MODE_ARGS="--coreneuron" ;;
  gpu)    MODE_ARGS="--coreneuron --gpu" ;;
  *) echo "bad mode: $MODE" >&2; exit 1 ;;
esac

MODEL_DIR="$TOP/model"
[ -x "$MODEL_DIR/x86_64/special" ] || { echo "Run ./03_build_model.sh first" >&2; exit 1; }

EXTRA="$*"; EXTRA="${EXTRA//--/}"; EXTRA="${EXTRA// /_}"
RUN_DIR="$TOP/runs/$(date +%Y%m%d-%H%M%S)_${MODE}_np${NP}_t${TSTOP}_g${GLOMS//[,:]/-}${EXTRA:+_$EXTRA}"
mkdir -p "$RUN_DIR"
# Model reads its input files from the cwd; link them in so outputs stay per-run.
ln -s "$MODEL_DIR"/* "$RUN_DIR"/
cd "$RUN_DIR"

CMD=($MPI_LAUNCH "$NP"
     "$MODEL_DIR/x86_64/special" -mpi -python bulb_bench.py
     --tstop "$TSTOP" --gloms "$GLOMS" $MODE_ARGS "$@")
{
  echo "# $(date -Is) host=$(hostname)"
  echo "# gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "# nrn: $(git -C "$SRC_DIR/nrn" rev-parse --short HEAD)  model: $(git -C "$SRC_DIR/olfactory-bulb-3d" rev-parse --short HEAD)"
  echo "# cmd: ${CMD[*]}"
} > run.log
export OMP_NUM_THREADS=1
/usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a run.log
echo "Run directory: $RUN_DIR"
