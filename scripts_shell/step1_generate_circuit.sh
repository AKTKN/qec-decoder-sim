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
	--noise_model phenomenological \
	--rounds "d" \
	--distance 7\
	--noise_strength 0.005

# # superdense_color_code_Z
# "$PYTHON_BIN" -m qec_decoder_sim.circuits.generate_circuits \
# 	--code midout_color_code_Z \
# 	--out_dir generated_circuits/midout_color \
# 	--noise_model uniform \
# 	--rounds "d" \
# 	--distance 5 7 9 11 \
# 	--noise_strength 0.001 0.002 0.003 0.005

# # bivariate_bicycle_code_Z (example parameters)
# "$PYTHON_BIN" -m qec_decoder_sim.circuits.generate_circuits \
# 	--code bivariate_bicycle_code_Z \
# 	--out_dir generated_circuits/bivariate_bicycle \
# 	--noise_model phenomenological \
# 	--rounds "d" \
# 	--distance 12 \
# 	--noise_strength 0.003 \
# 	--l 12 --m 6 \
# 	--a_x_pows 3 \
# 	--a_y_pows 1,2 \
# 	--b_x_pows 1,2 \
# 	--b_y_pows 3 \


# =====  BB code params list =====
# [[18, 4, 4]]
# code, A_list, B_list = create_bivariate_bicycle_codes(3, 3, [1], [0,2], [0,2], [1])
# d = 6

# [[72,12,6]]
# code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
# d = 6

# [[90,8,10]]
# code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0])
# d = 10

# [[108,8,10]]
# code, A_list, B_list = create_bivariate_bicycle_codes(9, 6, [3], [1,2], [1,2], [3])
# d = 10

# [[144,12,12]]
# code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
# d = 12

# [[288,12,18]]
# code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])
# d = 18
