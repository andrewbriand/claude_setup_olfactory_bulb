# env.sh -- source this before building models or running:  source env.sh
# Puts NVHPC compilers, MPI, the NEURON install and the Python venv on the path.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/config.sh"

export PATH="$NVHPC_ROOT/compilers/bin:$PATH"
[ -n "$MPI_BIN" ] && export PATH="$MPI_BIN:$PATH"

# Make MPI compiler wrappers use the NVIDIA compilers (needed by nrnivmodl -coreneuron).
export OMPI_CC=nvc OMPI_CXX=nvc++
# hwloc's GL plugin probes X11 displays for GPUs; under WSL2/WSLg that connect() never
# returns and mpirun hangs silently before launching anything. Harmless elsewhere.
export HWLOC_COMPONENTS=-gl

if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi

export PATH="$PREFIX/bin:$PATH"
# NMODL (CoreNEURON's code generator) dlopen()s libpython + $NMODLHOME/lib/libpywrapper.so
# for its sympy-based solver passes. pip wheels set these automatically; source builds do not.
export NMODLHOME="$PREFIX"
if [ -z "${NMODL_PYLIB:-}" ] && [ -x "$VENV/bin/find_libpython" ]; then
  export NMODL_PYLIB="$("$VENV/bin/find_libpython")"
fi
export PYTHONPATH="$PREFIX/lib/python${PYTHONPATH:+:$PYTHONPATH}"
unset _here
