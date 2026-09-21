# Model patches

Applied by `03_build_model.sh` to the *copy* of the model in `model/`; the upstream
checkout in `src/olfactory-bulb-3d` is never modified (`NO_PATCHES=1` skips them, which
is how you reproduce the unpatched baseline).

* **01-fast-granule-candidate-search.patch** — speeds up the mitral->granule candidate
  search in `lateral_connections.py`, which is ~83% of model setup time. Bit-identical
  results: the sequence of points inserted into the candidate set, and therefore the RNG
  stream in `connect_to_granule`, is unchanged. Verified with `compare_spikes.sh`.
  See README "Setup time" for measurements.
