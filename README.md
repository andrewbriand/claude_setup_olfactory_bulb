# NEURON + CoreNEURON (GPU) for the olfactory-bulb-3d model

Reproducible setup to build [NEURON](https://github.com/neuronsimulator/nrn) with
CoreNEURON GPU support (OpenACC via NVIDIA HPC SDK) and run the
[olfactory-bulb-3d](https://github.com/HumanBrainProject/olfactory-bulb-3d) model on the GPU.
Developed and validated on an RTX 4090 (WSL2); intended to be re-run on an FP64-capable GPU
for benchmarking.

Network construction and post-run teardown used to dwarf the simulation itself. By default they are now
~2.3x and ~12x faster respectively, with bit-identical results; see
[Setup and teardown performance](#setup-and-teardown-performance). The GPU solver's spike-event
delivery makes ~40% fewer GPU round trips per timestep (solver ~2x faster on the 4090, bit-identical);
see [Faster spike-event delivery](#faster-spike-event-delivery-neuron-patches-0205-default-bit-identical).

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
analyze_nvtx.py       attribute CUDA syncs/copies/launches/MPI to CoreNEURON's NVTX phases
gpu_roundtrip_check.cu  platform check: GPU round-trip cost vs managed memory (see "NVTX ranges")
profile_setup.py      cProfile the model's network construction (CPU-only, no GPU needed)
repro_presyn_disconnect.py  standalone NEURON reproducer for the O(N^2) teardown
patches/              applied to model/ by 03_build_model.sh; upstream src/ stays pristine
patches/optional/     opt-in, enabled with EXTRA_PATCHES=<name> (changes results; see below)
patches/nrn/          NEURON patches (NVTX; faster spike-event delivery), applied in order to the
                      src/nrn checkout by 02_build_neuron.sh
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
| `OB_NEIGHBOUR_CACHE` | `65536` | entries in the setup speed-up cache (~19 KB each, per rank); `131072` if RAM allows, `0` disables. Read by the model at run time, not build time; exported by `config.sh` |
| `OB_FAST_EXIT` | `1` | skip the O(P^2) object-graph teardown at exit (see "Setup and teardown performance"); `0` = upstream behaviour |
| `NO_PATCHES` | `0` | `1` builds the model unpatched (baseline) |
| `NRN_PATCHES_UPTO` | empty (all) | `02_build_neuron.sh` applies `patches/nrn/` only up to this number, reverting later ones: `01` = NVTX only, the baseline for measuring 02–05 |
| `NO_NRN_PATCHES` | `0` | `1` makes `02_build_neuron.sh` revert every NEURON patch (pristine NEURON) |
| `EXTRA_PATCHES` | empty | opt-in patches from `patches/optional/` by name, or `all` |

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
| `first:32` | 32 | 124,496 | 1,539,196 | 914,694 | ~14 GB peak host RAM at 4 ranks (~19 GB with the default `OB_NEIGHBOUR_CACHE`) |
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
Quarter bulb (`first:32`), GPU, 4 ranks: 37.9 s solver (setup 133 s, unpatched).
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

These are **unpatched** numbers (before the setup and teardown fixes; see "Setup and teardown
performance"). The full bulb runs comfortably in 196 GB; GPU memory is never the constraint (~10 GB at
4 ranks).

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

Consequence for optimisation priorities, as measured *before* the fixes below: across the whole
workload the GPU was idle most of the time — setup ~40% of wall, output and teardown ~55%, the solve
8–19% and only ~18% GPU-busy within that. Setup and teardown are now largely fixed (see "Setup and
teardown performance"), which leaves the solve's host-boundedness as the main remaining target.

## Setup and teardown performance

On the unpatched model, the H100's full-bulb run at 4 ranks spent **504 s building the network** and
**~748 s after the simulation had finished**, around a **115 s solve**. Both overheads are fixed here
without modifying either upstream checkout: model changes are patches applied to the `model/` copy,
and the teardown fix lives in our driver, `bulb_bench.py`.

### At a glance

| Fix | Where | Default? | Effect (quarter bulb, 4 ranks, this box) | Results |
|---|---|---|---|---|
| Fast candidate search | `patches/01-fast-granule-candidate-search.patch` | **on** | candidate search 112.5 s -> 37.3 s (28.3 s with a bigger cache) | bit-identical |
| Cheaper NEURON object creation | `patches/03-cheaper-object-creation.patch` | **on** | synapse construction 12.9 s -> 9.4 s | bit-identical |
| Sample candidates without building the set | `patches/optional/02-sample-without-materializing.patch` | opt-in | candidate search -> 9.8 s, *less* memory | same distribution, different network realization |
| Fast exit (skip teardown) | `bulb_bench.py` | **on** | teardown 18.9 s -> 1.5 s | outputs byte-identical |

**Setup, measured** (quarter bulb `first:32`, 4 ranks):

| Build | `OB_NEIGHBOUR_CACHE` | Setup | vs unpatched | Peak RSS/rank |
|---|---|---|---|---|
| unpatched (`NO_PATCHES=1`) | — | 136.2 s | 1.00x | 3.44 GB |
| **default: 01 + 03** | 65536 (default) | **58.5 s** | **2.3x** | 4.46 GB |
| default: 01 + 03 | 131072 | 47.9 s | 2.8x | 5.62 GB |
| 01 + 03 + optional 02 | (unused) | **29.3 s** | **4.6x** | **3.29 GB** |

**Teardown, measured** (same model, `-t 1`): 18.9 s -> 1.5 s; wall clock 91.4 s -> 73.9 s.

**Full bulb on the H100, estimated.** Per-phase speedups measured here (candidate search 3.98x with 01
and 11.33x with 01 + 02, synapse construction 1.38x with 03) applied to the H100's measured phase split
(8 ranks: candidate search 214 s, synapses 34 s, total 263 s; 4 ranks: total 504 s):

| Configuration | Candidate search | Synapses | Other | **Setup, 4 ranks** | vs baseline | Setup, 8 ranks |
|---|---|---|---|---|---|---|
| baseline (unpatched) | 428 s | 67 s | 8 s | **504 s** (measured) | 1.00x | 263 s (measured) |
| 01 | 108 s | 67 s | 8 s | 184 s | 2.75x | 103 s |
| 01 + 03 (**default**) | 107 s | 49 s | 8 s | **165 s** | 3.06x | 93 s |
| 01 + optional 02 | 38 s | 67 s | 8 s | 114 s | 4.43x | 68 s |
| 01 + 03 + optional 02 | 38 s | 49 s | 8 s | **95 s** | **5.28x** | 59 s |

With fast exit also removing ~12 min of teardown, **full-bulb wall clock at 4 ranks should drop from
1368 s to roughly 330–380 s** with the defaults (setup ~165 s + solve 115.5 s + 50–100 s of remaining
post-setup work), or ~260–310 s with the optional sampler. These are estimates: the factors come from a
quarter bulb on Zen 5, patch 01's cache may hit less often at full scale (the sampler does not depend on
it), the teardown figure is the H100 run's approximate "~12 min", and runs carry ~5% noise. **Confirm
with one full-bulb run on the H100** before quoting them.

### Using it

* **Nothing to do for the defaults.** `03_build_model.sh` applies `patches/*.patch`; `bulb_bench.py`
  skips teardown.
* **`OB_NEIGHBOUR_CACHE`** (default 65536, exported by `config.sh`): entries in patch 01's per-voxel
  cache, ~19 KB each *per rank*. Setup improves up to ~131072 (the working set) and then flattens. Use
  131072 on big-memory nodes; lower it, or use more ranks, when RAM is tight. Ignored by optional 02.
* **`EXTRA_PATCHES=02-sample-without-materializing ./03_build_model.sh`** for the fastest setup and the
  lowest memory. It changes the network *realization* (not its distribution), so keep it off when you
  need spike checksums comparable with earlier runs.
* **`NO_PATCHES=1 ./03_build_model.sh`** rebuilds the unpatched baseline. **`OB_FAST_EXIT=0`** restores
  upstream's exit path.
* **More ranks** shorten setup and teardown about 1/n each (see "Rank scaling" below); 4 ranks remains
  the best point for benchmarking the *solver* on the H100.
* After changing patches, run `./bench.sh verify`, and compare against an unpatched run at the same `-n`
  with `compare_spikes.sh`.

### Why setup was slow

`run.log` has per-phase `elapsedtime` lines. On the H100 at 8 ranks, **214 s of 263 s (81%)** went to
one phase, `Mitral ... cells connection infos. generated`: `connect_to_granule()` in
`lateral_connections.py`, which for each mitral dendrite segment builds the set of candidate granule
voxels under it (a 5x5x5 cube of offsets around each voxel on the line to the bulb surface) and draws
uniformly from it with rejection. Setup scales 1/n with a 98.9% parallel fraction
(`setup(n) = 1932/n + 21 s` fits three measured points), so it is per-rank Python work, not
communication.

The H100 report listed four suspects from reading the code (per-call `Ellipsoid` construction, an O(n)
`del` in the rejection loop, rebuilding the candidate set, closures), with the caveat that they were
unprofiled. **`cProfile` (`profile_setup.py`) refuted two of them:**

| Suspect | Verdict |
|---|---|
| `Ellipsoid` constructions per call | real but **1.4%** of the build |
| O(n) `del gvoxels[index]` | never surfaces: the list is ~980 long and `del` is C code |
| cache the candidate set per call | only **3.6%** of calls repeat a voxel path — worth ~3% |
| **what the profiler actually showed** | `get_neighbors`: 1.36M calls, 170M tuples generated, **79% of the build** |

A counted, unprofiled run (`first:4`, 1 rank; `connect_to_granule` = 16.6 s of a 28.4 s build):

| Quantity | Count | Unit cost |
|---|---|---|
| candidate points generated / distinct | 169,444,750 / 51,149,753 | **3.31x redundancy** |
| set insertions | 169M | ~15 ns hot, ~80 ns amortized |
| tuple + int allocations (cache misses) | 18.6M | **162 ns** each |
| voxel lookups that repeat an earlier voxel | 89.3% of 1,355,558 | — |
| `rng.discunif` draws | 154,925 (2.97 per connection) | 773 ns = **0.12 s total, 0.7%** |

So setup was **allocator- and hash-bound in CPython's object model**: every candidate is a 3-tuple of
heap-allocated ints, generated 3.3 times over. It was *not* RNG-bound — the Python-to-HOC RNG call is
expensive per call, but there are only ~3 per connection — and not instruction-bound.

### Patch 01: fast candidate search (default, bit-identical)

Four changes to `lateral_connections.py`:

1. a list comprehension over a hoisted offset tuple instead of 170M `list.append` calls;
2. a bounded `lru_cache` of each voxel's 125 neighbours, since 89% of voxel lookups repeat;
3. the two boundary ellipsoids built once (they are immutable) instead of per call;
4. **delta insertion**: walking the line, each voxel's cube mostly overlaps its predecessor's, and the
   overlap was inserted one step earlier, so a per-step mask (only 25 distinct masks exist) inserts just
   the new points. This removes the 3.31x redundancy.

Why the results stay **bit-identical**: the model draws `index = discunif(0, len-1)` into `list(set)`,
whose order is fixed by the set's *first*-insertion order. None of the changes alters that order, so the
RNG maps to the same voxels and builds the same network. Verified: spikes match unpatched runs exactly
(51,146 at 1 glomerulus, 250,166 at 5, same checksums).

Measured, quarter bulb, 4 ranks (steps 1–3 first, then with step 4):

| Version | Cache entries | Setup | Candidate search | Peak RSS/rank |
|---|---|---|---|---|
| unpatched | — | 136.2 s | 112.5 s | 3.44 GB |
| steps 1 + 3 | 0 | 113.7 s | 90.8 s | 3.47 GB |
| steps 1–3 | 65536 | 75.7 s | 50.4 s | 4.68 GB |
| steps 1–3 | 131072 | 65.2 s | 41.0 s | 5.89 GB |
| steps 1–3 | 262144 | 66.5 s | 41.7 s | 6.51 GB |
| **steps 1–4** | 65536 | 62.6 s | 37.6 s | 4.48 GB |
| **steps 1–4** | 131072 | **52.3 s** | 28.3 s | 5.62 GB |

### Patch 03: cheaper NEURON object creation (default, bit-identical)

The synapse constructor (`MGRS.__init__`, ~66 µs per call under the profiler) creates two
`ThreshDetect`s, a `FastInhib`, an `AmpaNmda`, a two-section `GranuleSpine`, several `NetCon`s and gid
registrations — a floor of ~40 µs of NEURON primitives. On top of that it paid two pure-Python costs:

* **`gc_is_superficial(ggid)` was 32% of the constructor**: 96,247 calls at 17.7 µs, each building two
  `Ellipsoid`s, for a pure function of the granule id. Now memoized per granule.
* **Every `h.<Name>` is a HOC symbol lookup costing 1.55 µs** (vs 0.05 µs for a local), ~7 per synapse.
  The templates are now bound once, lazily, since `GranuleSpine` only exists after the `.hoc` files load.

Quarter bulb, 4 ranks: synapse construction **12.9 s -> 9.4 s (1.38x)**, granule building 2.08 ->
1.88 s. Spikes bit-identical to unpatched runs.

**Would a vectorized / batch creation path do better?** NEURON has no bulk constructor for point
processes or NetCons in its Python API; the closest is building objects in a HOC loop. Measured, 20,000
objects each:

| Object | Python (lookups hoisted) | Pure HOC loop |
|---|---|---|
| `ThreshDetect` | 1.18 µs | 0.44 µs |
| `NetCon` | 1.18 µs | 0.67 µs |
| `GranuleSpine` (2 sections) | 14.04 µs | **9.94 µs** |

A HOC batch path would save ~8 µs of ~40 µs per synapse — ~20% of that phase, ~6% of setup — because the
floor is genuine C++ work, above all creating each spine's two sections. Not worth the rewrite.

### Optional patch 02: sample candidates without building the set

Even after patch 01, each connection materializes a ~981-point set to draw ~3 samples from it. Patch 02
instead draws `(voxel, offset)` uniformly over the line × cube grid and accepts a pair only at the
point's *first* occurrence along the line (a range test against earlier voxels). Each distinct point has
exactly one accepted pair, so accepted draws are uniform over the same candidate set (~30% acceptance).
Rejected points go into a `tried` set, reproducing the original sampling without replacement, and after
256 consecutive useless draws it hands over to the exact set-building loop so that "cannot connect"
stays exact near exhaustion.

Quarter bulb, 4 ranks: candidate search **112.5 s -> 9.8 s (11.5x)**, setup 32.0 s on top of 01 alone
and 29.3 s on top of 01 + 03, at **3.29 GB/rank — below the unpatched baseline**, since nothing is
materialized and the cache goes unused.

It consumes the RNG stream in a different order, so the network is a different realization of the same
random ensemble. Checksums can't validate that, so it was compared statistically at 4 ranks:

| Config | Cells | NetCons | Spikes |
|---|---|---|---|
| 1 glomerulus | 17,057 = 17,057 | 34,084 = 34,084 | 51,143 vs 51,146 (0.006%) |
| 5 glomeruli | 46,797 vs 46,900 (0.22%) | 144,364 = 144,364 | 249,899 vs 250,166 (0.11%) |
| quarter bulb | — | 914,676 vs 914,694 (0.002%) | connections 457,338 vs 457,347 |

Every delta is smaller than the model's own dependence on rank count (cells move 0.27% between 1 and 4
ranks), and convergence is unchanged (`it=4`, err 0.0061% vs 0.0042%). The patch is generated against
01 + 03 and applies on top of them.

### Rank scaling and the serial floor

Quarter bulb, 01 + 03 + optional 02:

| Phase | 4 ranks | 8 ranks | Scales? |
|---|---|---|---|
| import-time bulb geometry (every rank builds all granule positions) | 3.3 s | 3.6 s | **no** |
| mitrals | 2.6 s | 1.3 s | yes |
| candidate search | 10.0 s | 6.0 s | yes |
| granules | 1.9 s | 1.0 s | yes |
| blanes -> granule | 1.9 s | 1.0 s | yes |
| synapse construction | 9.2 s | 5.1 s | yes |
| **total setup** | **29.3 s** | **18.3 s** | 1.6x |

Rank count is the remaining 2x lever for setup and needs no code; the ~3.5 s of import-time geometry is
the serial floor it approaches. With optional 02, memory is ~1.9 GB/rank at quarter scale.

### Teardown: an O(P²) loop inside NEURON, skipped at exit (default)

The H100 run spent **748 s of a 1368 s wall clock** (4 ranks) after the simulation, with every rank at
100% CPU, and it got *superlinearly* worse with fewer ranks: 253 s at 8 ranks, still running after
14 min at 2. Sampling a rank's stack with `gdb` during teardown (quarter bulb, 4 ranks), 7 of 9 samples:

```
Py_FinalizeEx -> hoc_free_object -> PreSyn::~PreSyn -> NetCvode::presyn_disconnect
                                                        -> std::find(vector<PreSyn*>) / vector::erase
```

`NetCvode::presyn_disconnect()` (`src/nrncvode/netcvode.cpp`) does a linear `std::find` + `erase` on a
vector of *every* `PreSyn` on the rank, and for threshold sources a second linear search over the
per-thread threshold lists. Each deletion is O(P), and shutdown deletes all P, so teardown is **O(P²)
per rank** — a NEURON performance bug that any large network hits. `repro_presyn_disconnect.py` shows it
with stock NEURON and built-in mechanisms: doubling N multiplies deletion time by 3.6–4.0, and the cost
per `PreSyn` grows linearly (1.0 µs at 10k `IntFire1`s, 8.6 µs at 160k; 13.9 µs at 80k voltage-threshold
sources).

**The fix.** Every output is written and closed before `util.finish()` prints `total elapsed time`, so
freeing the object graph just before exit is pure waste. `bulb_bench.py` pins an extra reference on
every model-module global before finishing; interpreter shutdown then never frees the graph, and the OS
reclaims the memory at exit. NEURON's normal exit path, `MPI_Finalize` included, still runs.

A/B, quarter bulb, 4 ranks, `-t 1`, each output line timestamped, no debugger attached:

| | Wall | Setup | Handoff + solve | Output | **Teardown** | `mpirun` rc |
|---|---|---|---|---|---|---|
| `OB_FAST_EXIT=0` (upstream) | 91.4 s | 58.4 s | 11.5 s | 2.6 s | **18.9 s** | 0 |
| `OB_FAST_EXIT=1` (default) | 73.9 s | 58.6 s | 11.2 s | 2.6 s | **1.5 s** | 0 |

All output files are byte-identical between the two; with fast exit on, `bench.sh verify` passes and
spikes match the unpatched references at 1 and 5 glomeruli. Because the cost is quadratic in per-rank
model size, the saving is far larger at full scale — essentially the H100's whole ~12 minutes.

The proper fix belongs in NEURON: amortized O(1) removal (e.g. tombstone the slot and compact
occasionally, which keeps `psl_` order unchanged) and lazily rebuilt threshold lists. That would also
help sessions that delete and rebuild networks without exiting, which a fast exit cannot.

### What is left after setup

Quarter bulb, 4 ranks, all independent of `tstop` (~13.9 s in total):

| Step | Time |
|---|---|
| `h.stdinit()`: NEURON-side initialization, incl. the model's custom `init()` HOC loop over every segment | **6.95 s** |
| handoff to CoreNEURON begins | 1.09 s |
| CoreNEURON `nrn_setup` (build + GPU upload) | 1.77 s |
| CoreNEURON mechanism setup + finitialize | 0.88 s |
| weight files (a Python string format per synapse) | **2.15 s** |
| spike sort/write | 0.42 s |

`stdinit` is the largest remaining non-solver cost after setup.

### Reproducing these measurements

* **Setup profile** (CPU only, no GPU needed):
  `source env.sh && cd model && mpirun -np 1 -x PYTHONPATH ./x86_64/special -mpi -python profile_setup.py --gloms first:4`
  prints the top functions and writes `setup.prof.<rank>` for `pstats`.
* **Per-phase setup times**: the `elapsedtime` lines in any `runs/*/run.log`.
* **Before/after comparisons**: build the baseline with `NO_PATCHES=1 ./03_build_model.sh`, run, rebuild
  with the patches, run again at the same `-n`/`-g`/`-t`, then `./compare_spikes.sh <before> <after>`.
* **Teardown**: run with `PYTHONUNBUFFERED=1` (forwarded with `-x`) and timestamp each output line, e.g.
  `... | perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%.3f %s", time, $_'`; teardown is the gap from
  the `total elapsed time` line to process exit. Compare `OB_FAST_EXIT=0` and `1`.
* **Where a live rank is spending time** (no `perf` on WSL): `gdb -p <pid> -batch -ex "bt 25"`, repeated
  every couple of seconds.
* **The NEURON O(P²) on its own**: `python repro_presyn_disconnect.py [threshold]` after `source env.sh`.

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

### NVTX ranges for the solver phases

CoreNEURON already annotates its timestep with `Instrumentor::phase` regions (`timestep`,
`deliver-events`, `check-threshold`, `net-buf-receive-<mechanism>`, `update-net-receive-buf`,
`state-update`, `setup-tree-matrix`, `matrix-solver`, `gap-v-transfer`, `spike-exchange`, ...), and the
GPU build compiles its `CudaProfiling` backend (`-DCORENEURON_CUDA_PROFILING`), but that backend's
`phase_begin`/`phase_end` were empty. `patches/nrn/01-nvtx-ranges-for-coreneuron-phases.patch` makes
them `nvtxRangePushA`/`nvtxRangePop` (NVTX3 is header-only in the CUDA toolkit; with no tool attached
the calls are near-free), so every phase shows up in Nsight Systems with no hand-placed ranges.
`NRN_PROFILE_REGIONS=a,b,c` limits which phases are emitted. The patch doesn't change numerics
(`bench.sh verify` passes, spikes identical to the unpatched references).

`profile_bulb.sh` now also writes `rank<N>.phases.txt`, produced by `analyze_nvtx.py`. That joins the
trace's NVTX ranges with CUDA API calls, GPU kernels and MPI calls (nsys's own reports summarize them
separately), charging each API call to the innermost open phase and each kernel to the phase that
launched it. Per phase and per timestep it reports inclusive/exclusive time, pure host (CPU) time, time
blocked in syncs, device<->host copies, kernel launches and MPI, with counts, plus the GPU kernel time
the phase launched:

```bash
./profile_bulb.sh -n 1 -t 20 -g first:32 -r 0     # 1 rank: no GPU time-slicing between ranks
./analyze_nvtx.py runs/<dir>/rank0.nsys-rep       # re-run on any trace
```

Profile **one rank** unless MPS is running: with several ranks time-slicing one GPU, waits in one
rank include other ranks' kernels, and the attribution becomes misleading.

**What the solver's timestep is made of** before patches 02–05 (this box, 1 rank; counts are per
timestep and do not change with model size, 5 glomeruli vs quarter bulb). The next section shows
what 02–05 changed:

| Phase | Syncs | Device->host copies | Host->device copies | Launches |
|---|---|---|---|---|
| `deliver-events` (own work) | 14 | 2 | — | 4 |
| `check-threshold` (spike detection) | 8.4 | 7.7 | — | 2 |
| `net-buf-receive-ThreshDetect` | 5.8 | 5.8 | — | 1 |
| `net-buf-receive-AmpaNmda` | 5.6 | 4.5 | — | 1 |
| `update-net-receive-buf` (+ `net-receive-buf-cpu2gpu`) | 4.5 | — | ~13 | — |
| `gap-v-transfer` | 3 | 1 | 2 | 2 |
| `net-buf-receive-FastInhib`, `-orn` | 2 each | — | — | 1 each |
| `state-orn` | 2 | 1 (240 B) | 1 | 1 |
| `matrix-solver` | 1 | — | — | 1 |
| state/current kernels, 14 mechanisms | ~1 each | — | — | 1 each |

About 64 syncs, 39 launches and 40 small copies per timestep, matching the H100 run's counts. Most of
them come from spike-event delivery rather than from the numerics: `deliver-events` and its children
are 54–65% of the timestep. The GPU does its real work in a handful of kernels (`nrn_state`/`nrn_cur`
for `nax`, `kamt`, `kdrmt`, the Hines solver `solve_interleaved2`, `nrn_rhs`/`nrn_lhs`), 35 ms per
step at quarter-bulb scale on the 4090, which leaves the GPU idle ~70% of each timestep.

**Caveat: on WSL2 the *timings* are dominated by a platform artifact.** CoreNEURON allocates part of
its GPU data with `cudaMallocManaged` (`allocate_unified()` in `src/coreneuron/utils/memory.cpp`:
NMODL instance structs, Random123 streams, per-thread data). This GPU under WSL2 reports
`concurrentManagedAccess=0`, so every launch and sync pays for *all* the managed memory the process
holds, whether or not the host touched it:

| Managed memory held | launch + sync (`gpu_roundtrip_check.cu`) |
|---|---|
| none | 32 µs |
| 100 MB | 294 µs |
| 1000 MB | 2,740 µs |

That matches what the profiles show. In a quarter-bulb timestep (118 ms), the GPU is idle for 80 ms,
and for 79 of those 80 ms the host is blocked *inside* a CUDA call: `cuLaunchKernel` averages 835 µs,
all of it with the GPU idle, against 11 µs for a bare launch. Launch cost also scales with model size
(164 µs at 5 glomeruli, 5.1x more at quarter bulb, for a 4.7x larger model), implying ~60 MB and
~300 MB of managed memory respectively. On native Linux, `concurrentManagedAccess=1` and this cost
should mostly disappear. **Measure the solver's CPU-boundedness on native Linux** (the H100 box); on
WSL2, trust the per-step *counts* above but not the times. Check any machine with:

```bash
source env.sh && nvcc -O2 -arch=sm_${CUDA_ARCH} gpu_roundtrip_check.cu -o runs/gpu_roundtrip_check && runs/gpu_roundtrip_check
```

### Faster spike-event delivery (NEURON patches 02–05, default, bit-identical)

`deliver-events` was 54–65% of every timestep. It runs twice per step (at t and t+dt/2), and each time
the host fills per-mechanism receive buffers, uploads them, and runs each synaptic mechanism's
NET_RECEIVE pass: launch a kernel, wait, copy the mechanism's send-buffer count back, reset it. Four
patches in `patches/nrn/` cut the round trips. Patches 02, 03 and 05 change the NMODL code generator
(`src/nmodl/codegen/codegen_coreneuron_cpp_visitor.cpp`), so the mechanism code is regenerated at
model build time; 04 and 05 change CoreNEURON.

| Patch | Change | Why it is safe |
|---|---|---|
| `02-skip-empty-net-receive-passes` | Return immediately from a mechanism's pass when nothing was delivered to it. Measured before: only ~2.5 of the 8 passes per step have events (`orn`: 1–10%). | With no events the kernel runs zero iterations, and the send buffer is provably empty on entry: every other writer (INITIAL, WATCH) drains it before returning. |
| `03-fewer-net-receive-syncs` | Drop a second, back-to-back stream wait; reset the device-side send count only when it was non-zero. | The duplicate wait had nothing to wait for; a zero count is already zero on the device. |
| `04-single-copy-net-receive-buffer` | A receive buffer's six arrays become rows of one block (pinned in GPU runs) with the same layout on the device, so each upload is one `cudaMemcpy2DAsync` of the used prefix of every row instead of eight copies. | Same bytes arrive. The per-pass `_cnt`/`_displ_cnt` uploads are dropped: the kernel takes its loop bound from the host, and nothing on the device reads them. |
| `05-batch-net-receive-passes` | Launch every mechanism's kernel back to back, queue each send count into pinned host memory, wait **once**, then move sent events to the host in the original mechanism order (`net_buf_receive_all()`, driven by `NrnThread::_net_buf_receive_phase`). CPU runs keep the old whole-pass path. | Kernels run in the same order on the same stream; no kernel depends on another mechanism's host-side processing; host processing order is unchanged. |

**Validation.** After every patch: `bench.sh verify` (NEURON = CoreNEURON-CPU = CoreNEURON-GPU),
the 4-rank GPU run against the unpatched reference, and **every** timing run below against the
matching unpatched 1-rank run — all bit-identical (5 glomeruli: 250,317 spikes; quarter bulb: 906,756).
Patch 04 rewrote the buffer-growth path, which never triggers at the default capacity, so it was
stress-tested separately: buffers started at capacity 8 (growth confirmed with a `gdb` breakpoint,
hit from `ThreshDetect` event delivery during the solve) and all checks were still identical. The
OpenMP-offload code paths compile in the OpenACC build but were not run.

**Speed on the 4090** (1 rank, WSL2; solver time, median of 5 runs at 5 glomeruli / 3 at quarter bulb):

| Build | 5 glomeruli, 50 ms | Quarter bulb, 20 ms |
|---|---|---|
| baseline (NVTX only, `NRN_PATCHES_UPTO=01`) | 18.2 s (16.8–19.1) | 46.4 s (45.7–57.2) |
| **all patches (default)** | **9.35 s** (8.0–10.0) | **21.1 s** (19.0–22.8) |
| speedup | **1.95x** | **2.20x** |

Per-patch progression during development (3 runs at 5 glomeruli, 1–3 at quarter bulb; indicative only):

| Applied | 5 glomeruli (median) | Quarter bulb |
|---|---|---|
| 02 | 11.4 s | 20.6 s |
| 02–03 | 12.0 s | 22.0 s |
| 02–04 | 12.3 s | 22.9 s (22.1, 22.9, 35.2) |
| 02–05 | 8.2 s | 17.7 s (17.6, 17.7, 22.3) |

Patch 02 carries most of the gain and 05 adds a further step. 03 and 04 are within this box's noise,
which is large: the 02–05 row and the final row are the *same code* measured at different times, and
single runs can be off by 50% (the 35.2 s and 57.2 s values). 04 was expected to be neutral here —
host-to-device copies cost ~11 µs on this platform — and to matter only where API calls, not the
WSL2 managed-memory tax, dominate.

**Per timestep** (quarter bulb, 1 rank, `analyze_nvtx.py` on nsys traces; counts do not depend on the
platform):

| | Baseline | All patches |
|---|---|---|
| stream syncs | 67.3 | **41.6** |
| kernel launches | 39.0 | 33.6 |
| device -> host copies | 22.0 | 18.0 |
| host -> device copies | 20.4 | **5.0** |
| other CUDA calls | 15.7 | 11.6 |
| `deliver-events`, ms (under nsys) | 66.7 | **24.9** |
| whole timestep, ms (under nsys) | 124.6 | 71.2 |

Traces: `runs/20260920-232127_prof_np1_t20_gfirst-32/` (baseline) and
`runs/20260921-014519_prof_np1_t20_gfirst-32/` (all patches), each with `rank0.phases.txt`.
NVTX note: with patch 05, launches appear under `net-buf-receive-launch`, the single wait under
`net-buf-receive-wait`, and `net-buf-receive-<mechanism>` holds only moving sent events to the host;
both delivery passes per step now have these ranges (before, only the first did).

**Expect less on the H100.** On WSL2 every removed round trip also removes a managed-memory charge
(0.3–1 ms at quarter scale, see "NVTX ranges" above), which native Linux does not pay. The removed
*counts* — ~26 syncs, ~5 launches and ~19 copies per step — carry over; their value there is
native round-trip cost x count. To measure it on the H100:

```bash
NRN_PATCHES_UPTO=01 ./02_build_neuron.sh && ./03_build_model.sh   # baseline (NVTX only)
./02_build_neuron.sh && ./03_build_model.sh                        # all patches
```

**What is left in `deliver-events`:** `check-threshold` is now its largest part (12.5 of 24.9 ms,
8.2 syncs and 7.7 device->host copies per step): spike detection and `ThreshDetect`'s WATCH check,
each copying counts and entries back separately. It is the natural next target, together with
replacing `ThreshDetect`'s WATCH -> self-event -> `net_event` chain in the model.

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
* **Post-run teardown used to take longer than the whole simulation — fixed by default.** After
  `total elapsed time`, interpreter shutdown freed the model object by object, and NEURON's
  `NetCvode::presyn_disconnect()` makes each `PreSyn` deletion O(P), so teardown was O(P^2) per rank
  (~12 min at full bulb / 4 ranks on the H100; 2 ranks never finished). `bulb_bench.py` now skips it
  (`OB_FAST_EXIT=1`, the default); see "Setup and teardown performance". With `OB_FAST_EXIT=0`, `Solver Time` is still
  in `run.log` before teardown starts and the run can be killed — **by verified PID**, not
  `pkill -f special`: `run_bulb.sh` starts the next run within seconds, and a stale pattern match will
  take out the run you just launched (this cost one profiling run here).
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
