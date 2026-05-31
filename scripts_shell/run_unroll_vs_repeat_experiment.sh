#!/usr/bin/env bash
# Run the unroll-vs-repeat BB circuit comparison experiment.
#
# Usage:
#   ./scripts_shell/run_unroll_vs_repeat_experiment.sh [config.yaml]
#
# If no config path is given, scripts_shell/config_unroll_vs_repeat_example.yaml
# is used.  The experiment generates two sinter tasks per (p, decoder)
# combination — one for the explicitly-unrolled IBM circuit and one for the
# REPEAT-block IBM circuit — and writes results with a circuit_variant column.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=${1:-"${REPO_ROOT}/scripts_shell/config_unroll_vs_repeat_example.yaml"}

python -m qec_decoder_sim.runner.run_unroll_vs_repeat_experiment --config "${CONFIG_PATH}"
