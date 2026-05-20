#!/usr/bin/env bash

set -e
set -o pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT_DIR/src"

# surface_code_Z
"$PYTHON_BIN" -m qec_decoder_sim.circuits.generate_circuits \
	--code surface_code_Z \
	--out_dir generated_circuits/surface \
	--noise_model si1000 \
	--rounds 1 \
	--distance 3\
	--noise_strength 0.001

# # superdense_color_code_Z
# "$PYTHON_BIN" -m qec_decoder_sim.circuits.generate_circuits \
# 	--code superdense_color_code_Z \
# 	--out_dir generated_circuits/superdense_color \
# 	--noise_model si1000 \
# 	--rounds 2 \
# 	--distance 3 \
# 	--noise_strength 0.001

# # bivariate_bicycle_code_Z (example parameters)
# "$PYTHON_BIN" -m qec_decoder_sim.circuits.generate_circuits \
# 	--code bivariate_bicycle_code_Z \
# 	--out_dir generated_circuits/bivariate_bicycle \
# 	--noise_model uniform \
# 	--rounds d \
# 	--distance 6 \
# 	--noise_strength 0.002 0.001 \
# 	--l 6 --m 6 \
# 	--a_x_pows 3 \
# 	--a_y_pows 1,2 \
# 	--b_x_pows 1,2 \
# 	--b_y_pows 3 \
# 	--use_both \
# 	--ibm_reproduce
