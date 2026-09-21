"""Rank-independent cache of the mitral -> granule connections.

The candidate search in determine_connections.mk_mconnection_info() dominates model setup,
and its result -- model.mconnections, per mitral gid a list of
    (mgid, isec, xm, ggid, gisec, gx, (px, py, pz))
tuples -- is all the rest of the build needs from it. With OB_CONN_CACHE=1:

  * cache miss: the search runs as usual, then every rank writes its connections (lossless:
    64-bit floats, original order) to  $OB_CONN_CACHE_DIR/<key>/part<rank>.npz
  * cache hit: the search is skipped; every rank reads all parts and keeps the connections
    of its own mitral cells.

The cache key covers the glomerulus list and the contents of the model's source and data
files (so model patches, e.g. the optional sampler, get their own cache), but *not* the rank
count: a cache generated at N ranks can be loaded at any rank count, which then simulates the
same network. Loaded at N ranks it reproduces the computed network exactly.
"""
import hashlib
import json
import os

import numpy as np

import params
import util
from common import pc, rank, nhost

_HERE = os.path.dirname(os.path.realpath(__file__))
_TOP = os.path.dirname(_HERE)
# files in model/ that belong to our drivers, not to the model
_NOT_MODEL = {'bulb_bench.py', 'conn_cache.py', 'profile_setup.py'}
_FIELDS = ('mgid', 'isec', 'xm', 'ggid', 'gisec', 'gx', 'px', 'py', 'pz')


def enabled():
    return os.environ.get('OB_CONN_CACHE', '0') not in ('', '0')


def cache_path(gloms):
    h = hashlib.sha256()
    h.update(json.dumps(sorted(gloms)).encode())
    for name in sorted(os.listdir(_HERE)):
        if name in _NOT_MODEL or not name.endswith(('.py', '.hoc', '.txt', '.dic', '.mod')):
            continue
        h.update(name.encode())
        with open(os.path.join(_HERE, name), 'rb') as f:
            h.update(f.read())
    base = os.environ.get('OB_CONN_CACHE_DIR') or os.path.join(_TOP, 'conncache')
    return os.path.join(base, h.hexdigest()[:16])


def _save(model, path):
    cis = [ci for mgid in model.mconnections for ci in model.mconnections[mgid]]
    for ci in cis:
        assert len(ci) == 7 and len(ci[6]) == 3, ci
    cols = list(zip(*cis)) if cis else [()] * 7
    arrays = {
        'mgid': np.array(cols[0], dtype=np.int64), 'isec': np.array(cols[1], dtype=np.int64),
        'xm': np.array(cols[2], dtype=np.float64), 'ggid': np.array(cols[3], dtype=np.int64),
        'gisec': np.array(cols[4], dtype=np.int64), 'gx': np.array(cols[5], dtype=np.float64),
    }
    pos = np.array(cols[6], dtype=np.float64).reshape(-1, 3)
    arrays.update(px=pos[:, 0], py=pos[:, 1], pz=pos[:, 2])
    os.makedirs(path, exist_ok=True)
    part = os.path.join(path, 'part%05d.npz' % rank)
    with open(part + '.tmp', 'wb') as f:
        np.savez(f, **arrays)
    os.replace(part + '.tmp', part)
    n = int(pc.allreduce(len(cis), 1))
    pc.barrier()
    if rank == 0:
        with open(os.path.join(path, 'manifest.json'), 'w') as f:
            json.dump({'generated_with_ranks': nhost, 'parts': nhost, 'connections': n,
                       'glomeruli': params.Ngloms}, f)
    pc.barrier()


def _load(model, path):
    with open(os.path.join(path, 'manifest.json')) as f:
        manifest = json.load(f)
    local = np.array(sorted(model.mitrals.keys()), dtype=np.int64)
    conns = model.mconnections
    for r in range(manifest['parts']):
        d = np.load(os.path.join(path, 'part%05d.npz' % r))
        keep = np.isin(d['mgid'], local)
        c = {k: d[k][keep].tolist() for k in _FIELDS}
        for mgid, isec, xm, ggid, gisec, gx, px, py, pz in zip(*(c[k] for k in _FIELDS)):
            conns.setdefault(mgid, []).append((mgid, isec, xm, ggid, gisec, gx, (px, py, pz)))
    return manifest


def install(gloms):
    """Wrap determine_connections.mk_mconnection_info with the cache (if OB_CONN_CACHE=1)."""
    if not enabled():
        return
    import determine_connections as dc
    compute = dc.mk_mconnection_info
    path = cache_path(gloms)

    def mk_mconnection_info(model):
        # all ranks must take the same branch: the search is full of collectives
        hit = int(pc.allreduce(int(os.path.exists(os.path.join(path, 'manifest.json'))), 2))
        if hit:
            manifest = _load(model, path)
            n = int(pc.allreduce(sum(len(v) for v in model.mconnections.values()), 1))
            util.elapsed('Mitral %d cells connection infos. loaded from %s (generated with %d ranks)'
                         % (n, path, manifest['generated_with_ranks']))
        else:
            compute(model)
            _save(model, path)
            if rank == 0:
                print('connection cache written: %s' % path)

    dc.mk_mconnection_info = mk_mconnection_info
