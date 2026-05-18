#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH=${1:-"${REPO_ROOT}/scripts_shell/config_example.yaml"}

python -m qec_decoder_sim.runner.run_memory_experiment --config "${CONFIG_PATH}"
