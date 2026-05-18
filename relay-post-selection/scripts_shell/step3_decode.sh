#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

"$PYTHON_BIN" -m relay_post_selection step3 \
	--data_dir sampled_data/bb_d6_p003 \
	--out_dir results/bb_d6_p003_thresh-1 \
	--threshold 1.0 \
	--processes 4 \
	--decoder_type relay \
	--gamma0 0.125 \
	--gamma_dist_interval -0.24 0.66 \
	--pre_iter 60 \
	--num_sets 50 \
	--set_max_iter 60 \
	--stop_nconv 1 \
	--llr_clip_min -8.0 \
	--llr_clip_max 12.0 \
	--verbose \

