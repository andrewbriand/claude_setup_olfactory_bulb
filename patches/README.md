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
  `compare_spikes.sh` against unpatched runs. See README "Setup time" for measurements.
