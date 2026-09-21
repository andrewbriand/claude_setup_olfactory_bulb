# NEURON + CoreNEURON (GPU) for the olfactory-bulb-3d model

Reproducible setup to build [NEURON](https://github.com/neuronsimulator/nrn) with
CoreNEURON GPU support (OpenACC via NVIDIA HPC SDK) and run the
[olfactory-bulb-3d](https://github.com/HumanBrainProject/olfactory-bulb-3d) model on the GPU.
Developed and validated on an RTX 4090 (WSL2); intended to be re-run on an FP64-capable GPU
for benchmarking.

```bash
./setup_all.sh               # fetch -> venv -> build NEURON -> build model -> verify  (~15 min)
./run_bulb.sh -m gpu -n 4 -t 50 -g 5                # a single run
./bench.sh quick                                    # a small benchmark sweep
./summarize_runs.py                                 # table + runs/summary.csv
```

## Prerequisites

| What | Tested with | Notes |
|---|---|---|
| NVIDIA driver | 595.97 (CUDA 13.2) | must support the CUDA version bundled with NVHPC |
| [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk-downloads) | 25.7 (bundled CUDA 12.9) | provides `nvc`, `nvc++`, `nvcc`; required for the GPU build |
| MPI | Ubuntu OpenMPI 4.1.6 | any MPI; the model *requires* MPI |
| CMake | 3.28 | >= 3.20 recommended with NVHPC |
| Python | 3.12 | >= 3.10, **with shared libpython** (`python3-dev`) and `venv` |
| Other | gcc 13, bison, flex, git, readline, ncurses | |

Ubuntu 24.04 packages:

```bash
sudo apt install build-essential cmake git bison flex libreadline-dev libncurses-dev \
                 python3-dev python3-venv libopenmpi-dev openmpi-bin
```

NVHPC (if not already installed; pick the current version from the download page):

```bash
wget https://developer.download.nvidia.com/hpc-sdk/25.7/nvhpc_2025_257_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2025_257_Linux_x86_64_cuda_multi.tar.gz
sudo NVHPC_SILENT=true NVHPC_INSTALL_DIR=/opt/nvidia/hpc_sdk NVHPC_INSTALL_TYPE=single \
     nvhpc_2025_257_Linux_x86_64_cuda_multi/install
```

## Layout

```
config.sh             ALL machine-specific settings (paths, versions, GPU arch, MPI launcher)
env.sh                source to get compilers/MPI/NEURON/venv on PATH (used by run scripts)
requirements.txt      Python deps (build + runtime), versions capped as in NEURON's repo
00_fetch_sources.sh   clone nrn + olfactory-bulb-3d at pinned commits (with submodules)
01_setup_python.sh    create venv/ from requirements.txt
02_build_neuron.sh    cmake configure/build/install NEURON -> install/   (--clean to wipe)
03_build_model.sh     copy model sim/ -> model/, nrnivmodl -coreneuron -> model/x86_64/special
setup_all.sh          runs 00..03 then `bench.sh verify`
bulb_bench.py         model driver (= sim/bulb3dtest.py + selectable size, see below)
run_bulb.sh           one run -> runs/<timestamp>_<mode>_np<N>_t<ms>_g<gloms>/
bench.sh              sweeps: verify | quick | full
compare_spikes.sh     check spike outputs of runs are identical
summarize_runs.py     parse runs/*/run.log -> table + runs/summary.csv
profile_bulb.sh       one nsys-profiled run -> runs/<ts>_prof_.../rank<N>.nsys-rep
dev_rebuild.sh        incremental rebuild after editing NEURON/CoreNEURON sources
Dockerfile            targets: `bench` (all baked in) and `dev` (toolchain only)

src/  build/  install/  venv/  venv-nsys/  model/  runs/  logs/    generated
```

## Configuration (`config.sh`)

Every value can be overridden from the environment, e.g. `CUDA_ARCH=90 ./02_build_neuron.sh --clean`.

| Variable | Default | Meaning |
|---|---|---|
| `NRN_COMMIT` | `53de154d6` (master, 2026-09-15, 9.0.1-115) | NEURON version |
| `OB_COMMIT` | `1517ecb` (master) | model version; code identical to NEURON CI's pin `b07b76dc` |
| `NVHPC_ROOT` | `/opt/nvidia/hpc_sdk/Linux_x86_64/25.7` | HPC SDK location |
| `CUDA_ARCH` | auto-detected from GPU 0 | `89` = RTX 4090, `80` = A100, `90` = H100/GH200, `100` = B200; `"80;90"` for several |
| `MPI_BIN` | empty (MPI on PATH) | e.g. `$NVHPC_ROOT/comm_libs/mpi/bin` for NVHPC's HPC-X |
| `MPI_LAUNCH` | `mpirun --oversubscribe -x ... -np` | launcher prefix; e.g. `srun --mpi=pmix -n` on Slurm |
| `PREFIX`, `VENV`, `BUILD_DIR`, `SRC_DIR` | under this directory | |
| `EXTRA_CMAKE_ARGS` | empty | appended to the NEURON cmake line |

The NEURON build uses: `nvc`/`nvc++`/NVHPC `nvcc`, `NRN_ENABLE_CORENEURON=ON`,
`CORENRN_ENABLE_GPU=ON` with **OpenACC** offload (`CORENRN_ENABLE_OPENMP=OFF`), MPI on,
InterViews/RxD/tests off, `Release`. CoreNEURON compile flags come out as
`-O2 -cuda -gpu=cuda12.9,lineinfo,ccXX -acc`. `nvc++` targets the build host's CPU
(`-tp native` by default), so **always build on (the same CPU type as) the machine you run on.**

## Running

```
./run_bulb.sh [-m neuron|cpu|gpu] [-n RANKS] [-t TSTOP_MS] [-g GLOMS] [-- extra model args]
```

* `-m neuron` plain NEURON (CPU); `cpu` CoreNEURON on CPU; `gpu` CoreNEURON on GPU (default).
* `-n` MPI ranks. With `gpu`, ranks are spread round-robin over visible GPUs.
* `-t` simulated ms (model default 1050).
* `-g` model size, as glomeruli (the bulb has 127):

| `-g` | Glomeruli | Cells | Compartments (NEURON) | NetCons | Notes |
|---|---|---|---|---|---|
| `5` | 1 | 17,057 | 94,473 | 34,084 | NEURON CI test size |
| `5,37,32,78,7` | 5 | ~46,800 | ~324,500 | 144,364 | `bulb3dtest.py` default |
| `first:32` | 32 | 124,496 | 1,539,196 | 914,694 | ~14 GB peak host RAM at 4 ranks |
| `all` | 127 | ~198,000 | 5,145,388 | 3,580,886 | needs >31 GB host RAM, see below |

Every run directory has `run.log` (header with host/GPU/commits/command, full output,
`/usr/bin/time -v`), `olfactory_bulb.spikes.dat.000` (time gid), and weight files.
Key metric: **`Solver Time`** (CoreNEURON's timed `psolve`). `setup` (Python network
construction) is large and CPU-bound, and it's reported separately as `setup_s`.

## Verification

`./bench.sh verify` runs the NEURON CI configuration (1 glomerulus, 50 ms, 4 ranks) with NEURON,
CoreNEURON-CPU and CoreNEURON-GPU and requires identical spikes. Verified on the 4090:

* 1 glomerulus, 4 ranks: NEURON = CoreNEURON-CPU = CoreNEURON-GPU (51,146 spikes, identical)
* 1 glomerulus, 1 rank: CoreNEURON-CPU = CoreNEURON-GPU (51,144 spikes, identical after sort)
* 5 glomeruli, 4 ranks: CoreNEURON-CPU = CoreNEURON-GPU (250,166 spikes, identical)

## Results on this machine (RTX 4090, Ryzen Zen 5, 16 cores, WSL2) — 5 glomeruli, 50 ms

| Mode | Ranks | Solver time (s) | Wall (s) |
|---|---|---|---|
| CoreNEURON GPU | 1 | 20.9 – 22.0 | 103 – 105 |
| CoreNEURON GPU | 2 | 15.3 | 66 |
| CoreNEURON GPU | 4 | 12.7 – 13.5 | 42 – 43 |
| CoreNEURON CPU | 4 | 17.2 | 47 |
| CoreNEURON CPU | 16 | 8.5 – 9.3 | 30 – 31 |

1 glomerulus: 1 rank: CoreNEURON-CPU 16.9 s vs CoreNEURON-GPU 5.3 s (3.2×).
4 ranks: NEURON 9.1 s, CoreNEURON-CPU 5.2 s, CoreNEURON-GPU 10.0 s.
Quarter bulb (`first:32`), GPU, 4 ranks: 37.9 s solver (setup 133 s).
Ranges are repeat runs; expect ~5% run-to-run variance, so repeat benchmark points.

The 4090 has 1/64-rate FP64 and GPU utilization during the solve was only ~28%. At these sizes the
GPU run is limited by host-side work and kernel-launch latency, not FP64 throughput. Expect a different
picture on A100/H100-class GPUs and at larger model sizes. For profiling, use `profile_bulb.sh`.

## Results on an H100 (H100 80GB HBM3, Xeon Platinum 8468 / Sapphire Rapids, 16 cores, 196 GB, native Linux + MPS)

Driver 580.173.02 (CUDA 13.0), NVHPC 25.7 (bundled CUDA 12.9), `CUDA_ARCH=90`, MPS daemon running.
`./bench.sh verify` printed `IDENTICAL` — 51,146 spikes across NEURON, CoreNEURON-CPU and
CoreNEURON-GPU, matching the spike count from the 4090 reference above.

**Full bulb (`-g all`, 127 glomeruli, 193.5k cells, 5,145,388 compartments, 1050 ms, ~26.7M spikes):**

| Ranks | Solver (s) | Setup (s) | Wall (s) | Output+teardown (s) | Peak RSS/rank | Solve % of wall |
|---|---|---|---|---|---|---|
| 8 | 119.3 | 263 | **635** | 253 | 5.5 GB | 18.8% |
| 6 | 117.4 | 342 | 845 | 386 | 7.1 GB | 13.9% |
| 4 | **115.5** | 504 | 1368 | 748 | 10.5 GB | 8.4% |
| 2 | 123.3 | 939 | — | — | — | (killed in teardown) |

The full bulb runs comfortably in 196 GB; GPU memory is never the constraint (~10 GB at 4 ranks).

**Read this table twice.** Solver time is flat within noise from 4 to 8 ranks (3.3% spread, vs ~5%
run-to-run variance), with a real minimum at 4 and a clear degradation at 2 (+6.8%). But *wall clock*
moves the other way, because setup and teardown both scale as ~1/n: the solver-optimal configuration
(4 ranks) takes **2.2x longer to get an answer** than 8 ranks. Pick 4 ranks to benchmark the solver,
8+ for real science runs.

Smaller sizes, for comparison with the 4090 table above (50 ms):

| Size | Ranks | Solver (s) | 4090 equivalent |
|---|---|---|---|
| 5 gloms | 1 | 1.18 | 20.9 – 22.0 |
| 5 gloms | 2 / 4 / 8 | 1.20 / 1.26 / 1.57 | 15.3 / 12.7 – 13.5 / — |
| 5 gloms, CoreNEURON-CPU | 16 | 10.49 | 8.5 – 9.3 |
| quarter bulb (`first:32`) | 1 / 4 | 3.01 / 2.51 | 37.9 (at 4 ranks) |

At 5 glomeruli more ranks make the solver *slower* — the model is far too small to occupy an H100, and
MPI plus MPS overhead dominates. The crossover where extra ranks start paying for themselves appears
at the quarter bulb (4 ranks beats 1) and inverts again at full size.

### Setup cost scales as 1/n almost perfectly

Three full-bulb points fit `setup(n) = 1932/n + 21 s` to within 1.5 s each — a **98.9% parallel
fraction**, i.e. only ~21 s of irreducible serial work. Predicted 987 s at 2 ranks, measured 939 s.
So setup responds to rank count all the way down, with no diminishing-returns wall; there is simply a
large amount of per-rank Python work. Predicted 1 rank: ~33 min, which is why the 1-rank full-bulb
point is not worth collecting.

### Where the setup time actually goes (it is the model, not NEURON)

`run.log` carries per-phase `elapsedtime` lines. At 8 ranks, of 263 s of setup:

```
 214.06 s  Mitral 1790443 and mTufted 0 cells connection infos. generated (it=6,err=...)   <- 81%
  33.70 s  3580886 ThreshDetect for reciprocalsynapses constructed
   5.42 s  1905 mitrals created and connections to mitrals determined
   (everything else < 5 s)
```

One phase is ~81% of setup: the rejection-sampling retry loop in `mk_mconnection_info`
(`determine_connections.py`), which assigns mitral lateral dendrites to granule cells and retries
rejected ones, converging in 6 MPI-synchronised iterations. It costs **~120 us per connection**, which
is very slow for an RNG draw plus a geometric lookup. Reading `connect_to_granule`
(`lateral_connections.py`), the per-call work includes:

* **four `misc.Ellipsoid` constructions per call** (two directly, two more inside
  `get_granules_below`), all from module-level constants — ~7M identical objects built and thrown away;
* **`del gvoxels[index]` inside the rejection loop**, which is O(n) in a Python list, making the inner
  loop O(n^2) in the candidate count;
* **`get_granules_below` rebuilding the whole candidate voxel set on every retry**, though it is a pure
  function of `(dendrite point, glomid)` and therefore cacheable;
* four closures redefined per call.

All of this is in **olfactory-bulb-3d**, not NEURON. Note which fixes preserve results: hoisting the
constant objects, caching `get_granules_below` and lifting the closures do not touch the RNG stream, so
spikes stay bit-identical and `compare_spikes.sh` validates them exactly. Changing the O(n) delete to
swap-and-pop, or batching the `rng.discunif()` draws, **reorders the random stream and generates a
different (equally valid) network** — those need separate validation. `rng` is NEURON's
`h.Random().Random123()` (`params.py:146`), so each rejection-loop iteration is a Python->HOC call;
the fix is still model-side (draw less often / in bulk), not a NEURON change.
The 33.7 s of `ThreshDetect` construction is ~9.4 us per point process — genuine NEURON object-creation
cost, near its floor, and only addressable by changing the model's design.

**Caveat: the four bullets above come from reading the code, not from a profiler.** Only the 214 s phase
total and the ~120 us/connection unit cost are measured. Before optimising, confirm the split with
`cProfile` on a small model — it is pure Python and needs no GPU, e.g.
`./run_bulb.sh -m gpu -n 1 -t 1 -g first:8` with `python -m cProfile` around the construction, or simply
time the phases at two sizes. Best guess is that the candidate-set rebuild and the O(n) delete dominate
and the `Ellipsoid` churn is secondary, but that is a guess.

Per this repo's agent instructions the upstream repos are not patched here; any fix belongs on a **fork
of olfactory-bulb-3d** (last upstream commit 2022-11-07, so a fork carries almost no rebase burden —
unlike NEURON, which is pinned to an actively-developed master and is not where the win is anyway).

### The solve is host-bound, not GPU-bound

Trace: full bulb, 4 ranks, `tstop=20` (426 timesteps), all 4 ranks profiled. `nsys` overhead was only
**+6.1%** (bare 3.672 s vs 3.896 s under nsys), so the trace is representative.

* **GPU in-use per rank: 17.6 / 18.0 / 18.2 / 17.9 %** (`nsys recipe gpu_time_util`) — identical across
  ranks, so there is no load imbalance; all four are equally starved. Kernel time alone is 0.637 s of a
  3.896 s solve (16.4%).
* **Per timestep, each rank issues 39 kernel launches, ~42 host<->device transfers and 64 stream
  synchronisations.** CUDA API time: `cuStreamSynchronize` 1.026 s (62.5%), `cuMemcpyDtoHAsync` 0.456 s
  (27.8%), `cuLaunchKernel` 0.082 s (5.0%).
* **MPI is only ~211 ms, ~5% of the solve** — so this is not network-bound. But `MPI_Alltoallv` runs
  **exactly 426 times, once per timestep**: the model's gap-junction voltage transfer
  (`pc.setup_transfer()`) forces a device->host->MPI->host->device round trip every `dt`, each leg
  ending in a synchronisation.
* Hottest kernel is the Hines solver `solve_interleaved2` at 30.5% of GPU time (456 us avg); the rest is
  a long tail of `nrn_cur_*`/`nrn_state_*` mechanism kernels at 14–103 us — far too short to amortise
  that much launch and sync overhead.

This explains the whole rank sweep: 2 ranks cannot cover the host-side gaps, 4 covers them best via MPS
overlap, and 8 adds more per-rank synchronisation than it recovers. Under MPS the *device* is busier
than 18% (up to ~70% if overlap were perfect); the per-rank figure is the actionable one.

Consequence for optimisation priorities: across the whole workload the GPU is idle most of the time —
setup is ~40% of wall, output and teardown ~55%, and the solve itself is 8–19% and only ~18% GPU-busy
within that. Speeding up the model's Python construction is worth far more than any solver tuning.

## Profiling

```bash
./profile_bulb.sh [-n RANKS] [-t TSTOP_MS] [-g GLOMS] [-d SECS] [-r RANKS_TO_PROFILE]
```

CoreNEURON wraps `psolve` in `cudaProfilerStart()`/`cudaProfilerStop()`
(`src/coreneuron/apps/main1.cpp`, via `-DCORENEURON_CUDA_PROFILING`, which the GPU build defines
automatically), so `--capture-range=cudaProfilerApi` traces **only the solver** and skips the many
minutes of serial Python construction. Bound the trace with a small `-t` (dt is 0.046875 ms, so
`-t 20` is 426 timesteps and ~3 MB per rank) rather than with `-d`: the duration cap SIGTERMs the job
mid-`psolve`, so you lose the `Solver Time` line and the nsys-vs-bare overhead comparison with it.

Always take a bare run at the **same** `-n/-t/-g` to compare against — see the non-linearity gotcha
below. Aggregate across ranks with `nsys recipe` (`gpu_time_util`, `mpi_sum`, `cuda_api_sync`); its
dependencies are not bundled, install them into a venv with
`python3 <nsys>/target-linux-x64/python/packages/nsys_recipe/install.py --venv venv-nsys`
(pip may be missing from the system python, in which case create `venv-nsys` by hand and
`pip install numpy pandas psutil pyarrow`). There is no way to merge `.nsys-rep` files; either open them
together in the GUI, or capture the whole process tree in one session (`nsys profile mpirun ...`) to get
a single report.

## Container

`Dockerfile` has two targets, built from `nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu24.04` so the
8.9 GB SDK tarball never has to be downloaded again (19.3 GB image pulled at >100 MB/s, vs 2.8 MB/s
single-stream for the tarball):

* **`--target bench`** — sources, NEURON and mechanisms all baked in at the pinned commits. Self-contained
  and reproducible; the point of it is that a fresh VM needs only `docker pull` + `docker run`.
* **`--target dev`** — toolchain only, plus ccache/gdb. Bind-mount the project **at its own absolute
  path** (`-v "$PWD:$PWD" -w "$PWD"`), because RPATHs, `nrnivmodl` and `NMODLHOME` all contain absolute
  paths; mount it anywhere else and an existing build is unusable. Defaults to `Release` so numbers
  from it are comparable to the tables above.

Validated: `./bench.sh verify` inside `bulb:dev` produced 51,146 spikes with the same md5 as bare metal.
See the container gotchas below — a dev-image run needs the host MPI bind-mounted.

## Porting to the benchmark machine

1. Install prerequisites (above). Copy this directory without the generated
   dirs (`src build install venv model runs logs`), or clone it if you've put it in git.
2. Edit `config.sh` (or export overrides): `NVHPC_ROOT`, maybe `MPI_BIN`/`MPI_LAUNCH`.
   `CUDA_ARCH` is auto-detected; set it explicitly when building on a node without a GPU.
3. `./setup_all.sh`. It must end with `IDENTICAL`.
4. Several ranks per GPU help (see table). On native Linux enable MPS so ranks share the GPU
   concurrently instead of time-slicing: `nvidia-cuda-mps-control -d` (stop:
   `echo quit | nvidia-cuda-mps-control`). MPS is not available under WSL2.
5. Benchmark: `./bench.sh quick`, then `NPS_GPU="4 8 16" NPS_CPU="<cores>" ./bench.sh full`.
   Compare machines at the **same rank count** (see "network depends on rank count" below).

## Gotchas found while setting this up

* **`mpirun` hangs silently on WSL2** (even `mpirun -np 1 hostname`). hwloc's GL plugin
  `connect()`s to the X display (WSLg) and never returns. Fix: `HWLOC_COMPONENTS=-gl` (set in
  `env.sh`, forwarded by `MPI_LAUNCH`). Harmless elsewhere.
* **`nrnivmodl -coreneuron` aborts with `NMODL_PYLIB not set` / `NMODLHOME not set`.** Source
  builds (unlike pip wheels) need these for NMODL's embedded sympy. `env.sh` sets
  `NMODLHOME=$PREFIX` and `NMODL_PYLIB=$(find_libpython)`.
* **1-rank CoreNEURON runs exit 1** with `ZeroDivisionError` in the model's `parrun.printperf`
  (after the simulation, before weights are written): NEURON's step time is 0 when CoreNEURON
  solves. `bulb_bench.py` guards it; the upstream model is not modified. Its "Load Balance"
  line is meaningless under CoreNEURON.
* **The network depends on the MPI rank count** (5 glomeruli: 46,776 / 46,842 / 46,900 cells at
  1 / 2 / 4 ranks), because connectivity is generated per rank. It's deterministic for a given rank count.
  Only compare spikes, and preferably timings, between runs with the same `-n`.
* **With 1 rank, simultaneous spikes are written in arbitrary order.** `compare_spikes.sh`
  compares sorted `(time, gid)`.
* **Full bulb OOM at 31 GB.** Construction peaks around 30 GB at 8 ranks, and the NEURON→CoreNEURON
  handoff then holds two copies of the model. WSL2 gets half the host RAM by default (host here: 62 GB). To run
  `-g all` here, raise it in `C:\Users\<you>\.wslconfig` (`[wsl2]` / `memory=54GB`) and run
  `wsl --shutdown`. Otherwise use `first:N` sizes. GPU memory is not the limit (quarter bulb: 1.5 GB).
* **Post-run teardown is CPU-bound and can take longer than the whole simulation.** After the model
  prints `total elapsed time`, every rank sits at 100% CPU for minutes destroying the
  NEURON/Python object graph. At full bulb this was ~12 min at 4 ranks and still running after 14 min
  at 2 ranks (vs a 123 s solve), and it scales with cells-and-synapses *per rank*, so it is worst at low
  rank counts. It is invisible in `solver_s` but is the single largest line in wall clock (748 s of a
  1368 s run at 4 ranks). It is also tstop-independent — a 20 ms run pays the same teardown as a 1050 ms
  one. If you only need `Solver Time`, it is already in `run.log` before teardown starts and the run can
  be killed. **Kill by verified PID**, not `pkill -f special`: `run_bulb.sh` starts the next run within
  seconds, and a stale pattern match will take out the run you just launched (this cost one profiling run
  here).
* **Don't scale `Solver Time` linearly with `-t`.** There is ~1.5 s of fixed cost inside `psolve` (GPU
  warmup, first-touch, initial event-queue setup). At full bulb, `-t 20` measured 3.67 s where the
  1050 ms rate predicts 2.20 s — 8.6 ms/timestep vs 5.16 ms/timestep. Comparing a short profiled run
  against a rate extrapolated from a long run would have manufactured ~65% of phantom profiling overhead.
  Always measure the bare baseline at the same `-t`.
* **`dt` is 0.046875 ms, not NEURON's 0.025 default**, so 1050 ms is 22,400 timesteps. Size traces in
  timesteps, not milliseconds.
* **The full bulb at 1050 ms writes a ~550 MiB spike file** (~26.7M spikes) per run, plus ~40 MB of
  weight files per rank. A sweep fills a disk quickly.
* **With `Exclusive_Process` compute mode, MPS is required, not merely helpful.** Check with
  `nvidia-smi --query-gpu=compute_mode --format=csv`. Without the MPS daemon only one process can hold a
  context, so *every* multi-rank GPU run fails rather than just running slower — and the corollary is
  that you cannot do a with/without-MPS comparison without also changing the compute mode. MPS's daemon
  here ran as root with a world-writable pipe dir, so unprivileged ranks (and containers) can attach.
* **`nsys` resolution depends on `env.sh`.** `env.sh` prepends NVHPC's `compilers/bin` to `PATH`, and
  that directory contains its own `nsys`. So scripts that source `env.sh` get NVHPC's build
  (here 2025.3.1.90), *not* the one in `/usr/local/cuda-*/bin` (2025.3.2.474) that a bare shell resolves.
  Check `command -v nsys` after sourcing if the version matters; the GUI must be at least as new as the
  CLI that wrote the report.
* **Container: `/usr/bin/time` is not in the NVHPC base image.** `run_bulb.sh` calls `/usr/bin/time -v`,
  so without the `time` package every run dies *after* creating its run directory, which reads like a
  model crash rather than a missing package. The `Dockerfile` installs it.
* **Container: UID 1000 already exists in the NVHPC image** (as `ubuntu`, home `/home/ubuntu`, matching a
  stock Ubuntu host), so `useradd --uid 1000` fails with `UID 1000 is not unique`. Reuse it instead.
* **Container: host-built binaries need the host MPI bind-mounted.** `special`, `libnrniv.so` and
  `libcorenrnmech` carry the host MPI's path in their RPATH, so a host build run inside the container
  reports `libmpi.so.40: cannot open shared object file` (the container's MPI is NVHPC's, elsewhere).
  Either bind-mount the host MPI tree read-only at its own path, or rebuild inside the container against
  the bundled MPI. Only the `bench` target is free of this.
* **Don't pipe a validation command into `tail`**: `$?` then reports `tail`'s status, not the command's.
  An in-container `bench.sh verify` failure was masked this way here. Redirect to a log and check the
  exit code separately.
* **Standalone `special-core` on a `--dump-model` dataset is *not* equivalent** for this model:
  spikes differ on CPU (state and events set up by NEURON's custom `init()` aren't in the dump), and on
  GPU it segfaults in `OdorStimHelper`'s legacy Random123 `VERBATIM` code. Use the in-process
  modes (`run_bulb.sh`, optionally `-- --filemode`), which are what NEURON's CI validates.
