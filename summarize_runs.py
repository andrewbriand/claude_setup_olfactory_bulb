#!/usr/bin/env python3
"""Summarize runs/*/run.log: prints a table of the given runs (default: all) and
always rewrites runs/summary.csv with every run.

Usage: ./summarize_runs.py [RUN_DIR ...]
"""
import csv
import glob
import os
import re
import sys

TOP = os.path.dirname(os.path.abspath(__file__))

PATTERNS = {
    # CoreNEURON prints "Solver Time : X"; plain NEURON prints "Solver time : X".
    'solver_s':     r'Solver [Tt]ime : ([\d.eE+-]+)',
    'runtime_s':    r'^runtime = ([\d.eE+-]+)',
    'setup_s':      r'^total setup time\s+([\d.eE+-]+)',
    'wall':         r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)',
    'max_rss_kb':   r'Maximum resident set size \(kbytes\): (\d+)',
    'cells':        r'Number of cells: (\d+)',
    'compartments': r'Total # compartments =\s+(\d+)',
    'exit':         r'Exit status: (\d+)',
    'gpu':          r'^# gpu: (.*)',
    'host':         r'host=(\S+)',
}
NAME = re.compile(r'_(neuron|cpu|gpu)_np(\d+)_t([\d.]+)_g([^_]+)(?:_(.*))?$')


def wall_to_s(w):
    s = 0.0
    for part in w.split(':'):
        s = s * 60 + float(part)
    return s


def parse(run_dir):
    text = open(os.path.join(run_dir, 'run.log'), errors='replace').read()
    row = {'run': os.path.basename(run_dir)}
    m = NAME.search(row['run'])
    if m:
        row.update(mode=m[1], np=int(m[2]), tstop=float(m[3]), gloms=m[4], extra=m[5] or '')
    for key, pat in PATTERNS.items():
        m = re.search(pat, text, re.M)
        row[key] = m[1].strip() if m else ''
    if row['wall']:
        row['wall_s'] = round(wall_to_s(row['wall']), 2)
    spk = os.path.join(run_dir, 'olfactory_bulb.spikes.dat.000')
    row['spikes'] = sum(1 for _ in open(spk)) if os.path.exists(spk) else ''
    return row


def main():
    all_dirs = sorted(d for d in glob.glob(os.path.join(TOP, 'runs', '*'))
                      if os.path.isfile(os.path.join(d, 'run.log')))
    all_rows = [parse(d) for d in all_dirs]
    wanted = {os.path.basename(os.path.normpath(d)) for d in sys.argv[1:]}
    rows = [r for r in all_rows if not wanted or r['run'] in wanted]
    cols = ['run', 'host', 'gpu', 'mode', 'np', 'tstop', 'gloms', 'extra', 'cells', 'compartments',
            'spikes', 'setup_s', 'solver_s', 'runtime_s', 'wall_s', 'max_rss_kb', 'exit']
    out = os.path.join(TOP, 'runs', 'summary.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    show = ['mode', 'np', 'tstop', 'gloms', 'extra', 'cells', 'spikes', 'solver_s', 'wall_s', 'exit']
    print(' '.join('%-14s' % c for c in show))
    for r in rows:
        print(' '.join('%-14s' % str(r.get(c, ''))[:14] for c in show))
    print('\nwrote', out)


if __name__ == '__main__':
    main()
