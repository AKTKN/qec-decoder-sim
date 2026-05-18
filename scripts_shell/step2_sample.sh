#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

"$PYTHON_BIN" -m qec_decoder_sim.sampling \
	--circuit generated_circuits/bivariate_bicycle/code=bivariate_bicycle_code_Z,d=6,rounds=6,noisemodel=uniform,p=0.003.stim \
	--out_dir sampled_data/bb_d6_p003 \
	--shots 10000 \
	--seed 0 \
	--batch_size 2000
