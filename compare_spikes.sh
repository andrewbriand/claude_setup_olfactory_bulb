#!/usr/bin/env bash
# Check that two or more runs produced identical spikes (what NEURON's own CI test for
# this model checks). Spikes are compared as sorted (time, gid) sets: with 1 rank the
# model's spike2file.hoc leaves simultaneous spikes in arbitrary order.
# Runs must use the same -n/-t/-g: the network construction depends on the MPI rank count.
#
# Usage: ./compare_spikes.sh runs/<dirA> runs/<dirB> [...]
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,7p' "$0"; exit 1; }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
i=0 status=0
for d in "$@"; do
  sort -k1,1g -k2,2n "$d/olfactory_bulb.spikes.dat.000" > "$tmp/$i"
  printf '%-60s %8s spikes  %s\n' "$(basename "$d")" "$(wc -l < "$tmp/$i")" "$(md5sum < "$tmp/$i" | cut -c1-12)"
  if [ $i -gt 0 ] && ! cmp -s "$tmp/0" "$tmp/$i"; then
    echo "  -> DIFFERS from $(basename "$1") ($(diff "$tmp/0" "$tmp/$i" | grep -c '^[<>]') differing lines)"
    status=1
  fi
  i=$((i + 1))
done
[ $status -eq 0 ] && echo "IDENTICAL" || echo "MISMATCH"
exit $status
