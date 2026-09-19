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

src/  build/  install/  venv/  model/  runs/  logs/     generated
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
picture on A100/H100-class GPUs and at larger model sizes. For profiling, run the
`special` command from a `run.log` under `nsys profile`.

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
* **Standalone `special-core` on a `--dump-model` dataset is *not* equivalent** for this model:
  spikes differ on CPU (state and events set up by NEURON's custom `init()` aren't in the dump), and on
  GPU it segfaults in `OdorStimHelper`'s legacy Random123 `VERBATIM` code. Use the in-process
  modes (`run_bulb.sh`, optionally `-- --filemode`), which are what NEURON's CI validates.
