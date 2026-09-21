#!/usr/bin/env bash
# Profile the model's CoreNEURON solve with Nsight Systems.
#
# CoreNEURON wraps psolve in cudaProfilerStart()/cudaProfilerStop()
# (src/coreneuron/apps/main1.cpp, enabled by -DCORENEURON_CUDA_PROFILING, which the
# GPU build sets automatically), so --capture-range=cudaProfilerApi traces *only* the
# solver and skips the serial Python network construction, which otherwise dominates.
#
# Usage: ./profile_bulb.sh [-n RANKS] [-t TSTOP_MS] [-g GLOMS] [-d SECS] [-r LIST] [-- extra model args]
#   -n  MPI ranks (default 4)
#   -t  simulated ms (default 50)
#   -g  glomeruli, as in run_bulb.sh (default 5,37,32,78,7)
#   -d  cap collection at SECS of solver wall time (default 0 = whole solve).
#       NOTE: nsys SIGTERMs the job at the cap, which kills it mid-psolve, so the run
#       prints no "Solver Time" and you lose the nsys-vs-bare overhead comparison.
#       Prefer bounding the trace with a small -t instead, and keep -d as a safety net.
#   -r  comma-separated ranks to profile, or "all" (default "all"). Unprofiled ranks
#       run bare, so the job stays a normal N-rank MPI job either way.
#
# Output: runs/<timestamp>_prof_.../rank<N>.nsys-rep plus kernel/API summaries.
set -euo pipefail
TOP="$(cd "$(dirname "$0")" && pwd)"
source "$TOP/env.sh"

NP=4 TSTOP=50 GLOMS=5,37,32,78,7 DUR=0 WHICH=all
while getopts "n:t:g:d:r:h" o; do
  case $o in
    n) NP=$OPTARG ;; t) TSTOP=$OPTARG ;; g) GLOMS=$OPTARG ;; d) DUR=$OPTARG ;; r) WHICH=$OPTARG ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

MODEL_DIR="$TOP/model"
[ -x "$MODEL_DIR/x86_64/special" ] || { echo "Run ./03_build_model.sh first" >&2; exit 1; }
command -v nsys >/dev/null || { echo "nsys not on PATH" >&2; exit 1; }

DUR_TAG="" DUR_ARG=""
[ "$DUR" != 0 ] && { DUR_TAG="_d$DUR"; DUR_ARG="--duration=$DUR"; }
RUN_DIR="$TOP/runs/$(date +%Y%m%d-%H%M%S)_prof_np${NP}_t${TSTOP}_g${GLOMS//[,:]/-}${DUR_TAG}"
mkdir -p "$RUN_DIR"
ln -s "$MODEL_DIR"/* "$RUN_DIR"/
cd "$RUN_DIR"

# Per-rank wrapper: decides whether this rank runs under nsys, and gives each its own
# report file. OMPI_COMM_WORLD_RANK is only set inside the launched processes.
cat > wrap.sh <<WRAP
#!/usr/bin/env bash
r="\${OMPI_COMM_WORLD_RANK:-0}"
want="$WHICH"
prof=0
if [ "\$want" = all ]; then prof=1
else for w in \${want//,/ }; do [ "\$w" = "\$r" ] && prof=1; done
fi
if [ "\$prof" = 1 ]; then
  exec nsys profile \\
    --capture-range=cudaProfilerApi --capture-range-end=stop \\
    $DUR_ARG \\
    --trace=cuda,nvtx,osrt,mpi --mpi-impl=openmpi \\
    --cuda-memory-usage=false --force-overwrite=true \\
    -o "$RUN_DIR/rank\$r" \\
    "\$@"
else
  exec "\$@"
fi
WRAP
chmod +x wrap.sh

CMD=($MPI_LAUNCH "$NP" ./wrap.sh
     "$MODEL_DIR/x86_64/special" -mpi -python bulb_bench.py
     --tstop "$TSTOP" --gloms "$GLOMS" --coreneuron --gpu "$@")
{
  echo "# $(date -Is) host=$(hostname)"
  echo "# gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "# nsys: $(nsys --version | tail -1)"
  echo "# profiled ranks: $WHICH   duration cap: ${DUR}s (0 = whole solve)"
  echo "# cmd: ${CMD[*]}"
} > run.log
export OMP_NUM_THREADS=1
# nsys SIGTERMs the job at the duration cap, so a non-zero exit here is expected.
/usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a run.log || echo "(launcher exit $? -- expected when -d terminates the run)" | tee -a run.log

echo "=== reports ==="
ls -lh "$RUN_DIR"/*.nsys-rep 2>/dev/null || { echo "no .nsys-rep produced" >&2; exit 1; }
for rep in "$RUN_DIR"/*.nsys-rep; do
  b="${rep%.nsys-rep}"
  nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report cuda_gpu_mem_time_sum \
       --format csv --output "$b" "$rep" > "$b.stats.log" 2>&1 || true
done
echo "Run directory: $RUN_DIR"
