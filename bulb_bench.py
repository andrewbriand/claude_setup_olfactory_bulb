"""Benchmark driver: identical to the model's sim/bulb3dtest.py, but the set of
glomeruli to build is selectable instead of hard-coded, so problem size can scale.

  --gloms 5,37,32,78,7   explicit glomerulus ids (default = bulb3dtest.py's set)
  --gloms 5              NEURON CI test size
  --gloms first:N        glomeruli 0..N-1
  --gloms all            full bulb (127 glomeruli)

All other options (--tstop, --coreneuron, --gpu, --filemode, ...) are parsed by
the model's own args.py.
"""
import argparse

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gloms', default='5,37,32,78,7')
_gloms = _p.parse_known_args()[0].gloms

import params
import runsim
import odors
import parrun

# Upstream parrun.printperf divides by NEURON's step time, which is 0 when CoreNEURON
# did the solve -> ZeroDivisionError on 1 rank (after the sim, before weights are saved).
# Its numbers are meaningless under CoreNEURON anyway; use CoreNEURON's "Solver Time".
_printperf = parrun.printperf
def _safe_printperf(p):
    try:
        _printperf(p)
    except ZeroDivisionError:
        print('printperf: skipped (NEURON step time is 0 under CoreNEURON)')
parrun.printperf = _safe_printperf

# Fast exit. After util.finish() prints "total elapsed time" every output is written and
# closed, and the process exits through Python finalization, which frees the model object
# by object. NEURON's NetCvode::presyn_disconnect() does a linear find + erase in a vector
# of every PreSyn on the rank, so that teardown is O(P^2): ~22 s for a quarter bulb at 4
# ranks, ~12 min for the full bulb at 4 ranks on the H100. Pinning an extra reference on
# every model-module global keeps finalization from freeing the graph -- the OS reclaims
# it at exit instead -- while NEURON's normal exit path, MPI_Finalize included, still runs.
# OB_FAST_EXIT=0 restores the upstream behaviour.
import ctypes
import os
import sys
import util

_MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
_finish = util.finish

def _finish_without_teardown():
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and os.path.dirname(os.path.realpath(f)) == _MODEL_DIR:
            for v in list(vars(m).values()):
                ctypes.pythonapi.Py_IncRef(ctypes.py_object(v))
    _finish()

if os.environ.get('OB_FAST_EXIT', '1') != '0':
    util.finish = _finish_without_teardown

params.sniff_invl_min = params.sniff_invl_max = 500
params.training_exc = params.training_inh = True

from neuron import h
h('sigslope_AmpaNmda=5')
h('sigslope_FastInhib=5')
h('sigexp_AmpaNmda=4')
params.odor_sequence = [('Onion', 50, 1000, 1e+9)]

if _gloms == 'all':
    gloms = list(range(params.Ngloms))
elif _gloms.startswith('first:'):
    gloms = list(range(int(_gloms.split(':')[1])))
else:
    gloms = [int(g) for g in _gloms.split(',')]
if runsim.rank == 0:
    print('bulb_bench: building %d glomeruli: %s' % (len(gloms), _gloms))

runsim.build_part_model(gloms, [])
runsim.run()
