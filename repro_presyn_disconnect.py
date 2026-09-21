"""Standalone reproducer: deleting N NetCon sources costs O(N^2) in NEURON.

NetCvode::presyn_disconnect() does a linear find + erase on a vector of every PreSyn
(and, for threshold sources, a second linear search over the per-thread psl_th_ lists),
so freeing N of them one at a time is quadratic. This is what made the olfactory bulb's
post-run teardown take minutes; see README "Setup and teardown performance".
Needs only stock NEURON.

  source env.sh
  python repro_presyn_disconnect.py             # IntFire1 sources (psl_ search only)
  python repro_presyn_disconnect.py threshold   # voltage-threshold sources (both searches)
"""
import sys
import time

from neuron import h


def artificial(n):
    cells = [h.IntFire1() for _ in range(n)]
    ncs = [h.NetCon(c, None) for c in cells]      # one PreSyn per source
    t = time.perf_counter()
    del ncs, cells                                  # frees every PreSyn
    return time.perf_counter() - t


def threshold(n):
    secs = [h.Section(name="s%d" % i) for i in range(n)]
    ncs = [h.NetCon(s(0.5)._ref_v, None, sec=s) for s in secs]
    h.finitialize(-65)                              # populates psl_th_
    t = time.perf_counter()
    del ncs, secs
    return time.perf_counter() - t


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "artificial"
    trial, sizes = {
        "artificial": (artificial, (10000, 20000, 40000, 80000, 160000)),
        "threshold": (threshold, (10000, 20000, 40000, 80000)),
    }[mode]
    print("NEURON %s, %s sources" % (h.nrnversion(5), mode))
    prev = None
    for n in sizes:
        dt = trial(n)
        ratio = "" if prev is None else "   x%.2f vs N/2" % (dt / prev)
        print("N=%7d  delete %8.3f s  %6.2f us/PreSyn%s" % (n, dt, dt / n * 1e6, ratio))
        prev = dt


if __name__ == "__main__":
    main()
