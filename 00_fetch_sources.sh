#!/usr/bin/env bash
# Clone NEURON and the olfactory bulb model at the commits pinned in config.sh.
set -euo pipefail
source "$(dirname "$0")/config.sh"

fetch() {  # fetch <url> <dir> <commit>
  if [ ! -d "$2/.git" ]; then
    git clone "$1" "$2"
  fi
  git -C "$2" fetch --quiet origin
  git -C "$2" checkout --quiet "$3"
  echo "$(basename "$2"): $(git -C "$2" log -1 --format='%h %cd %s' --date=short)"
}

mkdir -p "$SRC_DIR"
fetch "$NRN_REPO" "$SRC_DIR/nrn" "$NRN_COMMIT"
git -C "$SRC_DIR/nrn" submodule update --init --recursive --quiet
fetch "$OB_REPO" "$SRC_DIR/olfactory-bulb-3d" "$OB_COMMIT"
