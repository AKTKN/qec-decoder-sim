#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

# Single threshold — plot summary metrics.
# "$PYTHON_BIN" -m relay_post_selection step4 \
# 	--results_dir results/bb_d6_p001_thresh-2 \
# 	--out_dir plots/bb_d6_p001

# Threshold sweep — compare multiple thresholds.
"$PYTHON_BIN" -m relay_post_selection step4 \
	--sweep_dirs \
		results/bb_d6_p003_thresh-1 \
	--sweep_thresholds 1.0 \
	--out_dir plots/bb_d6_p003_thresh-1 \
	--label "BB d=6 p=0.001" \
	--results_dir sampled_data \
