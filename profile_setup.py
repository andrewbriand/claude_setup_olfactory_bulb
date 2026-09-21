"""cProfile the model's network construction (the dominant part of setup time).

No GPU or simulation involved -- this is pure Python/HOC model building.

  <special> -mpi -python profile_setup.py --gloms first:8 [--profile-out setup.prof]

Writes <out>.<rank> for each rank and prints the top functions on rank 0.
"""
import argparse
import cProfile
import pstats

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--gloms', default='first:8')
_p.add_argument('--profile-out', default='setup.prof')
_p.add_argument('--sort', default='tottime')
_a = _p.parse_known_args()[0]

import params
import runsim

params.sniff_invl_min = params.sniff_invl_max = 500
params.training_exc = params.training_inh = True

from neuron import h
h('sigslope_AmpaNmda=5')
h('sigslope_FastInhib=5')
h('sigexp_AmpaNmda=4')
params.odor_sequence = [('Onion', 50, 1000, 1e+9)]

g = _a.gloms
if g == 'all':
    gloms = list(range(params.Ngloms))
elif g.startswith('first:'):
    gloms = list(range(int(g.split(':')[1])))
else:
    gloms = [int(x) for x in g.split(',')]

pr = cProfile.Profile()
pr.enable()
runsim.build_part_model(gloms, [])
pr.disable()

out = '%s.%d' % (_a.profile_out, runsim.rank)
pr.dump_stats(out)
if runsim.rank == 0:
    print('\n===== cProfile: build_part_model, %d glomeruli, %d rank(s) ====='
          % (len(gloms), runsim.nhost))
    pstats.Stats(pr).sort_stats(_a.sort).print_stats(22)
print('profile written to', out)
h.quit()
