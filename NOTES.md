# Project notes

Running log of state, open items and decisions that are not obvious from the code. Newest first.
Measurements and how-tos live in README.md; this file is for "where were we".

## 2026-09-23 — H100 session (built inside the dev container)

### Open items

1. **Rebuild and re-push the dev image.** `…/bulb:latest` (built 2026-09-20) lacks `libfl-dev`, so
   NEURON fails on `FlexLexer.h`. The `Dockerfile` fix is committed; the image is not rebuilt yet:
   `sudo docker build --target dev -t cr.eu-north1.nebius.cloud/e00y40ggy230574740/bulb:latest --build-arg CUDA_ARCH=90 .`
   then `sudo docker push …`. Consider also building `--target bench` from the same commit.
2. **Full-bulb GPU runs are not reproducible run to run** (README Gotchas). Unverified hypothesis:
   floating-point atomics / accumulation order on the GPU. Worth confirming (e.g. two CoreNEURON-CPU runs
   at full bulb should then match exactly; bisect by size to find where GPU runs start to diverge),
   because it limits how NEURON patches can be validated at scale.
3. **`NRN_PROFILE_REGIONS` is dead in embedded mode** (upstream: only standalone `special-core` calls
   `Instrumentor::init_profile()`). A small NEURON patch (lazy init in `is_region_to_track`, or call
   `init_profile()` from the embedded entry point) would fix it; also a candidate upstream bug report.
   Not needed now that patch 03 removed the per-event ranges.
4. **Not measured this session** (CLAUDE.md steps 8–9): `OB_FAST_EXIT=0` vs `1` wall clock at full bulb
   (the ~12 min teardown is inferred from the earlier unpatched run; with fast exit, teardown was
   2.7–5.8 s in every run); MPS on vs off; `bench.sh quick/full` and `runs/summary.csv`.
5. **Next solver targets** (unchanged): `check-threshold` inside `deliver-events`; the per-`dt`
   gap-junction transfer round trip. Low-overhead traces for this are described in README "NVTX ranges".
6. Post-setup, non-solver time at full bulb / 4 ranks is now ~90 s: `stdinit` + handoff ~54 s, spike
   sort/write ~24 s, weights ~12 s — next largest after the solve and setup.

### What was done

- Fresh H100 VM with no NVHPC on the host; built everything in the registry dev image via
  `sudo docker exec` (command recipe in README "Container"). `bench.sh verify`: IDENTICAL.
- Connection cache, full bulb: generated at 16 ranks (95 s setup), loaded at 4 ranks (~70 s setup).
  Wall clock at 4 ranks / 1050 ms: ~270 s (was 1368 s unpatched).
- NEURON patch 02 A/B on the H100: 112.4 s -> 106.3 s solver (5.5%), full bulb, 4 ranks.
- 1 rank: full bulb impossible (CoreNEURON int32 limit); half bulb (`first:64`) 67.3 s solver.
- Added NEURON patch 03 (no per-event `net-receive-<mech>` NVTX range; nsys overhead 29% -> ~10%)
  and `NSYS_TRACE` / `NSYS_EXTRA` in `profile_bulb.sh` (NVTX-only traces at ~5% overhead).

### Decisions

- Removed the per-event range outright (patch 03) instead of fixing `NRN_PROFILE_REGIONS`: simplest,
  and it also removes a heap allocation per delivered event from unprofiled runs.
- Patch 03 sits after 02, so the `NRN_PATCHES_UPTO=01` baseline still has the per-event ranges. For
  profiling A/Bs of 02 with low overhead, either accept that or reorder 03 before 02.

### Artifacts (on the instance only; lost when it is deleted)

`runs/`, `logs/*.ts.log` (timestamped run logs), `conncache/` (full bulb `84101f4883ce6f8d`, half bulb
`7f33a392748cc568`, both generated at 16 ranks). Traces: `runs/20260923-0408*` (baseline),
`-0400*` (patch 02), `-0428*` (02, NVTX only), `-0443*` (03, NVTX only), `-0447*` (03, full),
`-0512*` (half bulb, 1 rank, 03, full). Copy anything worth keeping off the instance before deleting it.
