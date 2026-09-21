#!/usr/bin/env python3
"""Attribute CUDA API calls, GPU work and MPI time to CoreNEURON's NVTX phases.

nsys's built-in reports summarize NVTX ranges and CUDA calls separately; this joins them.
Each CUDA API call (and MPI call) is charged to the innermost NVTX range open on the same
thread; kernels and copies are charged to the range whose API call launched them (via
correlationId). Per phase it reports, per timestep:

  incl / excl   wall time inside the range, and excluding nested ranges
  sync          cu*Synchronize time (host blocked on the GPU) and call count
  d2h / h2d     host<->device copy API time and count
  launch        kernel-launch API time and count
  mpi           MPI time
  host          excl minus all of the above: pure CPU work in this phase
  gpu           kernel execution time on the device for work this phase launched

Usage: ./analyze_nvtx.py runs/<prof dir>/rank0.nsys-rep   (or the .sqlite nsys exports)
Requires NVTX ranges from patches/nrn/01-nvtx-ranges-for-coreneuron-phases.patch.
"""
import collections
import os
import sqlite3
import subprocess
import sys

SYNC = ('cuStreamSynchronize', 'cuEventSynchronize', 'cuCtxSynchronize',
        'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize')
D2H = ('cuMemcpyDtoH', 'cudaMemcpyDtoH')
H2D = ('cuMemcpyHtoD', 'cudaMemcpyHtoD')
LAUNCH = ('cuLaunchKernel', 'cudaLaunchKernel')


def category(api):
    if api.startswith(SYNC):
        return 'sync'
    if api.startswith(D2H):
        return 'd2h'
    if api.startswith(H2D):
        return 'h2d'
    if api.startswith(LAUNCH):
        return 'launch'
    return 'other'


def open_db(path):
    if path.endswith('.nsys-rep'):
        db = path[:-len('.nsys-rep')] + '.sqlite'
        if not os.path.exists(db):
            subprocess.run(['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true',
                            '-o', db, path], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        path = db
    return sqlite3.connect(path)


def mpi_events(c):
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    for t in sorted(tables):
        if t.startswith('MPI_') and t.endswith('_EVENTS'):
            cols = {r[1] for r in c.execute(f'pragma table_info({t})')}
            if {'start', 'end', 'globalTid'} <= cols:
                yield from c.execute(f'select start, end, globalTid from {t}')


def main(path):
    c = open_db(path)
    names = dict(c.execute('select id, value from StringIds'))

    # NVTX push/pop ranges (eventType 59), per thread
    ranges = collections.defaultdict(list)
    for start, end, text, tid in c.execute(
            'select start, end, text, globalTid from NVTX_EVENTS '
            'where eventType = 59 and end is not null'):
        ranges[tid].append([start, end, text.lstrip(':') if text else '?'])

    # everything to attribute: (start, end, tid, kind, category, correlationId)
    calls = collections.defaultdict(list)
    for start, end, tid, corr, nid in c.execute(
            'select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        api = names.get(nid, '?')
        calls[tid].append((start, end, category(api), corr))
    for start, end, tid in mpi_events(c):
        calls[tid].append((start, end, 'mpi', None))

    gpu_by_corr = collections.defaultdict(int)
    for start, end, corr in c.execute(
            'select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        gpu_by_corr[corr] += end - start

    stat = collections.defaultdict(lambda: collections.defaultdict(float))
    parent_of = collections.defaultdict(collections.Counter)
    for tid, rs in ranges.items():
        rs.sort(key=lambda r: (r[0], -r[1]))
        # nesting: parent index per range, and child time per range
        parent = [None] * len(rs)
        child_time = [0] * len(rs)
        stack = []
        for i, (s, e, name) in enumerate(rs):
            while stack and rs[stack[-1]][1] <= s:
                stack.pop()
            if stack:
                parent[i] = stack[-1]
                child_time[stack[-1]] += e - s
            stack.append(i)
        for i, (s, e, name) in enumerate(rs):
            st = stat[name]
            st['n'] += 1
            st['incl'] += e - s
            st['excl'] += e - s - child_time[i]
            parent_of[name][rs[parent[i]][2] if parent[i] is not None else '-'] += 1

        # innermost open range for each call: sweep ranges and calls by start time
        cs = sorted(calls.get(tid, ()))
        stack = []
        ri = 0
        for s, e, cat, corr in cs:
            while ri < len(rs) and rs[ri][0] <= s:
                while stack and rs[stack[-1]][1] <= rs[ri][0]:
                    stack.pop()
                stack.append(ri)
                ri += 1
            while stack and rs[stack[-1]][1] < e:
                stack.pop()
            name = rs[stack[-1]][2] if stack else '(outside any range)'
            st = stat[name]
            st[cat] += e - s
            st[cat + '#'] += 1
            if corr is not None and corr in gpu_by_corr:
                st['gpu'] += gpu_by_corr[corr]

    steps = stat['timestep']['n'] or 1
    cats = ('sync', 'd2h', 'h2d', 'launch', 'mpi', 'other')
    for name, st in stat.items():
        st['host'] = st['excl'] - sum(st[k] for k in cats)

    ms = 1e6 * steps  # ns -> ms per timestep
    print('%d timesteps; all times are ms per timestep (counts per timestep in parentheses)\n' % steps)
    hdr = ('phase', 'parent', 'incl', 'excl', 'host', 'sync', 'd2h', 'h2d', 'launch', 'mpi', 'gpu')
    print('%-30s %-16s %7s %7s %7s %13s %13s %13s %13s %6s %7s' % hdr)
    for name, st in sorted(stat.items(), key=lambda kv: -kv[1]['incl']):
        if name == '(outside any range)' and not st['n']:
            pass
        def cc(k):
            return '%6.3f(%5.1f)' % (st[k] / ms, st[k + '#'] / steps)
        par = parent_of[name].most_common(1)[0][0] if parent_of[name] else '-'
        print('%-30s %-16s %7.3f %7.3f %7.3f %13s %13s %13s %13s %6.3f %7.3f' % (
            name[:30], par[:16], st['incl'] / ms, st['excl'] / ms, st['host'] / ms,
            cc('sync'), cc('d2h'), cc('h2d'), cc('launch'), st['mpi'] / ms, st['gpu'] / ms))

    ts = stat['timestep']
    tot = {k: sum(s[k] for n, s in stat.items() if n != '(outside any range)')
           for k in ('host', 'sync', 'd2h', 'h2d', 'launch', 'mpi', 'other', 'gpu')}
    print('\nPer timestep, over all phases: wall %.3f ms = host %.3f + sync %.3f + d2h %.3f + h2d %.3f '
          '+ launch %.3f + mpi %.3f + other %.3f;  GPU kernels %.3f ms (%.0f%% of wall)' % (
              ts['incl'] / ms, tot['host'] / ms, tot['sync'] / ms, tot['d2h'] / ms, tot['h2d'] / ms,
              tot['launch'] / ms, tot['mpi'] / ms, tot['other'] / ms, tot['gpu'] / ms,
              100 * tot['gpu'] / ts['incl'] if ts['incl'] else 0))
    cnt = {k: sum(s[k + '#'] for n, s in stat.items() if n != '(outside any range)') / steps
           for k in ('sync', 'launch', 'd2h', 'h2d', 'other', 'mpi')}
    print('Per timestep, counts: %.1f syncs, %.1f kernel launches, %.1f device->host copies, '
          '%.1f host->device copies, %.1f other CUDA calls, %.1f MPI calls' % (
              cnt['sync'], cnt['launch'], cnt['d2h'], cnt['h2d'], cnt['other'], cnt['mpi']))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
