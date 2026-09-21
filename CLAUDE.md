# Agent instructions: install and validate on a new system

Goal: build NEURON + CoreNEURON with GPU support and run the olfactory-bulb-3d model on this
machine's GPU, then benchmark it. README.md has the full reference; this file is the procedure.
Do not modify the upstream checkouts in `src/`. Model changes go in `patches/*.patch`, which
`03_build_model.sh` applies to the `model/` copy (`NO_PATCHES=1` rebuilds the baseline); every
patch must keep spikes bit-identical, verified with `compare_spikes.sh` against an unpatched run,
or else document explicitly why the network legitimately changed.

## Procedure

1. **Survey the machine** before changing anything:
   `nvidia-smi` (GPU model, driver CUDA version), `nvidia-smi --query-gpu=compute_cap --format=csv`,
   `ls /opt/nvidia/hpc_sdk/Linux_x86_64/` (NVHPC versions), `which mpirun mpicc cmake python3`,
   `nproc`, `free -g`, whether Slurm/modules are in use (`which srun module`).
2. **Prerequisites** (README "Prerequisites"): NVIDIA HPC SDK, MPI, CMake >= 3.20,
   Python >= 3.10 with shared libpython (`python3-dev`) and `venv`, bison, flex, readline, ncurses.
   Ask the user before installing system packages or NVHPC (needs sudo / large download).
   The NVHPC-bundled CUDA version must be supported by the driver (`nvidia-smi` top-right).
3. **Configure** by exporting overrides or editing `config.sh`, never the other scripts:
   - `NVHPC_ROOT` if not `/opt/nvidia/hpc_sdk/Linux_x86_64/25.7`
   - `CUDA_ARCH` only if building on a node without the target GPU (80 A100, 90 H100/GH200, 100 B200)
   - `MPI_BIN` / `MPI_LAUNCH` if not using OpenMPI's `mpirun` (e.g. `MPI_LAUNCH="srun --mpi=pmix -n"`)
   On HPC clusters, build and run on a compute node with the target GPU/CPU (nvc++ targets the build host CPU).
4. **Run `./setup_all.sh`** (~15 min on 16 cores). It must print `IDENTICAL` at the end
   (NEURON, CoreNEURON-CPU and CoreNEURON-GPU spikes match). Logs are in `logs/`.
   If a step fails, fix it and re-run that step's script (`00_`..`03_`); all are idempotent.
   `./02_build_neuron.sh --clean` after changing compilers or `CUDA_ARCH`.
5. **Benchmark**: `./bench.sh quick`, then scale up, e.g.
   `NPS_GPU="1 4 8" NPS_CPU="$(nproc)" ./bench.sh full`, and `./summarize_runs.py`.
   Check host RAM first: the full bulb (`-g all`) needs > 31 GB (about 45 GB at 4 ranks).
   Use `-g first:N` to scale if RAM is short. On native Linux with several ranks per GPU,
   try MPS (`nvidia-cuda-mps-control -d`) and report with/without.
6. **Report**: `runs/summary.csv`, the GPU/CPU model, and any deviations from this procedure.
   Add new pitfalls to the "Gotchas" section of README.md.
7. **Setup time** is ~2/3 of wall clock for big runs and is model-side Python. On a big-memory node
   set `OB_NEIGHBOUR_CACHE=131072` (see README "Setup time"). Default patches 01 + 03 give ~3x on
   setup, bit-identical. For ~5x at the cost of a
   different (statistically equivalent) network realization, build with
   `EXTRA_PATCHES=02-sample-without-materializing`; it also needs *less* memory. If you optimise
   further, **profile before believing any lead** — `profile_setup.py` needs no GPU, and cProfile
   already refuted two plausible-looking leads that were read from the code.
8. **Next target is output/teardown**, ~40% of full-bulb wall clock on the H100 and never profiled.

## Known pitfalls (details in README "Gotchas")

- `mpirun` hangs silently: hwloc GL plugin probing X11. `env.sh` sets `HWLOC_COMPONENTS=-gl`.
- `NMODL_PYLIB not set` / `NMODLHOME not set` from nrnivmodl: source `env.sh` (sets both).
- Compare spikes/timings only between runs with the same rank count (`-n`); network construction
  depends on it.
- Don't use `--dump-model` + standalone `special-core`: not equivalent for this model and it
  crashes on GPU. Use `run_bulb.sh` (in-process CoreNEURON).
- `Solver Time` is the benchmark metric; network setup (`setup_s`) is serial Python and slow.
