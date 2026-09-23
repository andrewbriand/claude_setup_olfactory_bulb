# NEURON + CoreNEURON (GPU, OpenACC) + olfactory-bulb-3d for H100 (cc90).
#
# Two targets, because benchmarking and development want opposite things:
#
#   --target bench   Everything baked in at pinned commits. Immutable, reproducible,
#                    for benchmark runs on a fresh node. ~25 min to build.
#   --target dev     Toolchain ONLY -- no NEURON, no model, no sources. You bind-mount the
#                    project at its own absolute path (see DEV below), so sources, build/,
#                    install/ and model/ live on the host and survive container restarts.
#                    ~1 min to build, and it never needs rebuilding when you change code.
#
# Uses NVIDIA's NVHPC image rather than downloading the 8.9 GB SDK tarball, which is the
# slowest and most failure-prone step of a bare-metal setup (single-stream throughput from
# developer.download.nvidia.com measured at 2.8 MB/s here; a registry pull is parallel,
# resumable and layer-cached).
#
# ---------------------------------------------------------------------------------------
# BENCH
#   sudo docker build --target bench -t bulb:25.7-cc90 \
#        --build-arg CUDA_ARCH=90 --build-arg TARGET_CPU=sapphirerapids .
#   sudo docker run --rm -it --gpus all --ipc=host \
#        -v /tmp/nvidia-mps:/tmp/nvidia-mps -v "$PWD/runs:/opt/bulb/runs" \
#        bulb:25.7-cc90 ./bench.sh verify
#
# DEV  (toolchain only -- builds nothing; you bind-mount the project)
#   sudo docker build --target dev -t bulb:dev --build-arg CUDA_ARCH=90 .
#   sudo docker run --rm -it --gpus all --ipc=host --cap-add=SYS_ADMIN \
#        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
#        -v "$PWD:$PWD" -w "$PWD" -e CCACHE_DIR="$PWD/.ccache" bulb:dev
#
#   Mount the project at its OWN absolute path (-v "$PWD:$PWD" -w "$PWD"), not at
#   /opt/bulb. NEURON bakes absolute paths into RPATHs, nrnivmodl and NMODLHOME, so an
#   existing host-side install/ and model/ keep working only if the path is unchanged --
#   mount it elsewhere and you must rebuild from scratch.
#
#   # if the host tree is already built (as after setup_all.sh): nothing to do, just run
#   # from scratch, once:  ./00_fetch_sources.sh && ./01_setup_python.sh
#   #                      ./02_build_neuron.sh && ./03_build_model.sh
#   # per C++ edit:        ./dev_rebuild.sh       (incremental; see that script)
#   # per .py edit:        nothing -- Python is interpreted; just re-run
#
#   Build type defaults to Release, matching the validated benchmark configuration. For
#   debug info (nsys/compute-sanitizer source attribution) override per build:
#        -e CMAKE_BUILD_TYPE=RelWithDebInfo ./02_build_neuron.sh --clean
#   Do not benchmark a RelWithDebInfo build: nvc++ -g can inhibit optimisation.
#
# Why the extra run flags:
#   --gpus all                 GPU access (nvidia container runtime).
#   --ipc=host + /tmp/nvidia-mps  MPS. NOT optional for >1 rank: this GPU is in
#                              Exclusive_Process mode, so without MPS multi-rank runs FAIL
#                              rather than merely running slower. MPS's daemon is root-owned
#                              with a world-writable pipe dir, so uid 1000 can attach.
#   --cap-add=SYS_ADMIN        only needed for nsys CPU/OS-runtime sampling. CUDA kernel and
#                              API tracing work without it; add it for --trace=osrt.
#
# Acceptance gate for either image: `./bench.sh verify` must print IDENTICAL
# (NEURON == CoreNEURON-CPU == CoreNEURON-GPU spikes).
# ---------------------------------------------------------------------------------------

ARG NVHPC_TAG=25.7-devel-cuda12.9-ubuntu24.04

# =======================================================================================
# base -- toolchain and OS dependencies, shared by both targets
# =======================================================================================
FROM nvcr.io/nvidia/nvhpc:${NVHPC_TAG} AS base

# Compute capability of the *target* GPU. Must be explicit: config.sh normally derives this
# from `nvidia-smi`, which does not exist during a docker build, and 02_build_neuron.sh then
# aborts with "CUDA_ARCH not set and no GPU detected". 90 = H100/GH200.
ARG CUDA_ARCH=90

# nvc++ defaults to `-tp native`, i.e. it tunes for whatever CPU ran the build. In a
# container that is a silent trap: an image built on a different CPU than it runs on either
# hits illegal instructions or quietly produces differently-optimised code, with no warning
# either way. So pin it. Valid nvc++ values include sapphirerapids, icelake, cascadelake,
# skylake, zen2..zen4; `px` is the portable lowest common denominator (slower CPU kernels).
ARG TARGET_CPU=sapphirerapids

# Where the NVHPC image puts the SDK; checked below so a base-image layout change fails
# loudly and early rather than deep inside cmake.
ARG NVHPC_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7

ENV DEBIAN_FRONTEND=noninteractive
# `time` is GNU /usr/bin/time, which run_bulb.sh invokes as `/usr/bin/time -v`. It is NOT in
# the base image, and without it every run dies with "/usr/bin/time: No such file or
# directory" after the run directory has already been created -- which looks like a model
# failure rather than a missing package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates time \
        bison flex libfl-dev libreadline-dev libncurses-dev \
        python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    test -x "${NVHPC_ROOT}/compilers/bin/nvc++" \
      || { echo "ERROR: no nvc++ under NVHPC_ROOT=${NVHPC_ROOT}."; \
           echo "Layout of this base image:"; ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/* || true; \
           echo "Rebuild with --build-arg NVHPC_ROOT=<correct path>."; exit 1; }; \
    test -x "${NVHPC_ROOT}/comm_libs/mpi/bin/mpicc" \
      || { echo "ERROR: no bundled MPI under ${NVHPC_ROOT}/comm_libs/mpi/bin."; exit 1; }; \
    "${NVHPC_ROOT}/compilers/bin/nvc++" --version

# Run as uid 1000 so files written to bind mounts are owned correctly and OpenMPI does not
# refuse to run (as it does for root, which would otherwise force --allow-run-as-root into
# MPI_LAUNCH). The NVHPC base image ALREADY has uid 1000 as `ubuntu` with home /home/ubuntu
# -- which matches a stock Ubuntu host -- so creating a user here fails with
# "useradd: UID 1000 is not unique". Reuse whatever is there instead of inventing one.
RUN set -eux; \
    id -u 1000 >/dev/null 2>&1 \
      || useradd --uid 1000 --create-home --shell /bin/bash bulb; \
    echo "running as uid 1000 = $(id -un 1000), home $(getent passwd 1000 | cut -d: -f6)"

# Fixed path. Absolute paths get baked into RPATHs, nrnivmodl and NMODLHOME, so the location
# must be stable -- this is what makes a relocatable tarball of a bare-metal build awkward,
# and what a container fixes by construction.
WORKDIR /opt/bulb
RUN chown 1000:1000 /opt/bulb

# config.sh reads all of these from the environment, so none of the numbered scripts need
# editing, and an interactive shell behaves the same way the build did.
ENV NVHPC_ROOT=${NVHPC_ROOT} \
    CUDA_ARCH=${CUDA_ARCH} \
    TARGET_CPU=${TARGET_CPU} \
    MPI_BIN=${NVHPC_ROOT}/comm_libs/mpi/bin \
    HWLOC_COMPONENTS=-gl

USER 1000

# =======================================================================================
# bench -- everything baked at pinned commits
# =======================================================================================
FROM base AS bench

ENV EXTRA_CMAKE_ARGS="-DCMAKE_C_FLAGS=-tp=${TARGET_CPU} -DCMAKE_CXX_FLAGS=-tp=${TARGET_CPU}"

# Copied in dependency order so editing a run/bench script does not invalidate the
# ~25-minute NEURON build layer.
COPY --chown=1000:1000 config.sh env.sh requirements.txt ./
COPY --chown=1000:1000 00_fetch_sources.sh 01_setup_python.sh ./
RUN ./00_fetch_sources.sh && ./01_setup_python.sh

COPY --chown=1000:1000 02_build_neuron.sh ./
COPY --chown=1000:1000 patches/nrn/ ./patches/nrn/
RUN ./02_build_neuron.sh

COPY --chown=1000:1000 03_build_model.sh bulb_bench.py profile_setup.py conn_cache.py ./
COPY --chown=1000:1000 patches/ ./patches/
RUN ./03_build_model.sh

COPY --chown=1000:1000 run_bulb.sh bench.sh profile_bulb.sh compare_spikes.sh \
                       summarize_runs.py dev_rebuild.sh analyze_nvtx.py \
                       gpu_roundtrip_check.cu ./

RUN printf 'nrn %s\nolfactory-bulb-3d %s\n' \
      "$(git -C src/nrn rev-parse HEAD)" \
      "$(git -C src/olfactory-bulb-3d rev-parse HEAD)" > /opt/bulb/COMMITS.txt \
    && cat /opt/bulb/COMMITS.txt

LABEL org.opencontainers.image.title="NEURON+CoreNEURON GPU / olfactory-bulb-3d (bench)" \
      org.opencontainers.image.description="Prebuilt for cc90 (H100). Verify with ./bench.sh verify."
CMD ["bash"]

# =======================================================================================
# dev -- toolchain only; bind-mount the project at its own absolute path
# =======================================================================================
FROM base AS dev
USER root

# ccache: the point of the dev image. nvc++ is slow and CoreNEURON is a large compile.
# Honest caveat: ccache's nvc/nvc++ support is real but hit rates with `-acc -cuda` are not
# guaranteed -- check `ccache -s` after your second build and set `-e CCACHE_DISABLE=1` if it
# is not earning its keep. gdb for debugging; compute-sanitizer ships in ${NVHPC_ROOT}/cuda/bin.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache gdb less vim-tiny file \
    && rm -rf /var/lib/apt/lists/*
USER 1000

# Release by default so a build made in this image is directly comparable to the validated
# benchmark numbers: nvc++ -g (as RelWithDebInfo adds) can inhibit optimisation. Override
# per-build with `-e CMAKE_BUILD_TYPE=RelWithDebInfo` when you want host-side source
# attribution in nsys or compute-sanitizer. GPU device code already carries
# `-gpu=...,lineinfo` in either build type, so device attribution needs no override.
ARG CMAKE_BUILD_TYPE=Release
ENV CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    CCACHE_MAXSIZE=20G

# ccache launchers live in the image ENV rather than in .bashrc: a non-interactive
# `docker run ... ./02_build_neuron.sh` never sources .bashrc, so a shell-rc approach would
# silently not apply. To bypass ccache use its own switch, `-e CCACHE_DISABLE=1`, which
# makes it a transparent pass-through without changing these flags.
# CCACHE_DIR is intentionally NOT set here -- it would default into the container's
# ephemeral home. Pass `-e CCACHE_DIR="$PWD/.ccache"` so the cache persists on the host
# (.ccache/ is in .dockerignore).
ENV EXTRA_CMAKE_ARGS="-DCMAKE_C_FLAGS=-tp=${TARGET_CPU} -DCMAKE_CXX_FLAGS=-tp=${TARGET_CPU} \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# Bind-mounted repos are owned by the host user; without this git refuses with "detected
# dubious ownership" whenever uids do not line up, which breaks 00_fetch_sources.sh and
# run_bulb.sh's provenance header.
RUN git config --global --add safe.directory '*'

# Deliberately NO COPY and no build steps: baking sources into a dev image is exactly what
# makes it go stale, and the bind mount would shadow them anyway.
LABEL org.opencontainers.image.title="NEURON+CoreNEURON GPU / olfactory-bulb-3d (dev toolchain)" \
      org.opencontainers.image.description="Toolchain only. Bind-mount the project at its own absolute path: -v \$PWD:\$PWD -w \$PWD"
CMD ["bash"]
