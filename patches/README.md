# Model patches

Applied by `03_build_model.sh` to the *copy* of the model in `model/`; the upstream
checkout in `src/olfactory-bulb-3d` is never modified (`NO_PATCHES=1` skips them, which
is how you reproduce the unpatched baseline).

* **01-fast-granule-candidate-search.patch** — speeds up the mitral->granule candidate
  search in `lateral_connections.py`, which is ~83% of model setup time. Three changes:
  a list comprehension over hoisted offsets, a bounded per-voxel `lru_cache`
  (`OB_NEIGHBOUR_CACHE`), and delta-insertion along the voxel path (each voxel inserts
  only the points its predecessor did not already contribute — 3.3x fewer insertions).
  Bit-identical results: the *first-insertion* order into the candidate set is unchanged,
  so `list(set)` order, the RNG stream and the network are unchanged. Verified with
  `compare_spikes.sh` against unpatched runs. See README "Setup and teardown performance" for measurements.

* **03-cheaper-object-creation.patch** — memoizes `gc_is_superficial()` per granule id (pure
  function, 96k calls at 17.7 us) and binds the HOC templates used by the synapse constructor
  once instead of paying a 1.55 us symbol lookup per `h.<Name>` access. Synapse construction
  1.38x faster. Bit-identical.

## Optional patches (`patches/optional/`)

Not applied by default. Enable by name:
`EXTRA_PATCHES=02-sample-without-materializing ./03_build_model.sh` (or `EXTRA_PATCHES=all`).

* **02-sample-without-materializing.patch** (generated against 01 + 03) — draws candidates by rejection sampling instead
  of building the ~981-point candidate set per connection. Quarter bulb, 4 ranks: setup
  32.0 s vs 52.3 s with patch 01 alone and 136.2 s unpatched; peak RSS drops to 3.30 GB/rank.
  **Changes the network realization** (same distribution, different draw order), so it is
  validated statistically, not by checksum — see README "Setup and teardown performance". Turn it off to
  reproduce spike checksums recorded with patch 01 only.

## NEURON patches (`patches/nrn/`)

Applied in order to the `src/nrn` checkout by `02_build_neuron.sh`, which works out which prefix
of the series is already applied and applies the rest (`NRN_PATCHES_UPTO=NN` stops after patch NN,
`NO_NRN_PATCHES=1` reverts all). All are bit-identical; see README "Faster spike-event delivery".

* **01-nvtx-ranges-for-coreneuron-phases.patch** — NVTX ranges for CoreNEURON's existing
  Instrumentor phases.
* **02-skip-empty-net-receive-passes.patch** — (NMODL codegen) skip a mechanism's NET_RECEIVE pass
  when nothing was delivered to it.
* **03-fewer-net-receive-syncs.patch** — (NMODL codegen) drop a redundant stream wait and a no-op
  send-count reset.
* **04-single-copy-net-receive-buffer.patch** — (CoreNEURON) receive-buffer arrays as rows of one
  pinned block; one 2D copy per upload instead of eight.
* **05-batch-net-receive-passes.patch** — (CoreNEURON + NMODL codegen) launch all mechanisms'
  NET_RECEIVE kernels, wait once, then process sent events in the original order.
