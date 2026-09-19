#!/usr/bin/env bash
# One-shot setup: fetch sources, Python venv, build NEURON (GPU), build model, verify.
# Every step is idempotent and can also be run on its own. Settings: config.sh.
set -euo pipefail
cd "$(dirname "$0")"

step() { echo; echo "=== $* ==="; }
step "0/4 fetch sources";        ./00_fetch_sources.sh
step "1/4 python venv";          ./01_setup_python.sh
step "2/4 build NEURON";         ./02_build_neuron.sh > logs/02_build_neuron.out 2>&1 \
                                   || { tail -30 logs/02_build_neuron.out; exit 1; }
                                 tail -1 logs/02_build_neuron.out
step "3/4 build model";          ./03_build_model.sh > logs/03_build_model.out 2>&1 \
                                   || { tail -30 logs/03_build_model.out; exit 1; }
                                 tail -1 logs/03_build_model.out
step "4/4 verify (neuron vs coreneuron-cpu vs coreneuron-gpu)"
./bench.sh verify
