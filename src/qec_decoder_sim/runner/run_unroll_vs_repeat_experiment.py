"""Entry point for the unroll-vs-repeat BB circuit comparison experiment.

Usage::

    python -m qec_decoder_sim.runner.run_unroll_vs_repeat_experiment \\
        --config path/to/config.yaml

Or via the shell wrapper::

    scripts_shell/run_unroll_vs_repeat_experiment.sh [config.yaml]

The experiment generates two sinter tasks per (p, decoder) combination —
one for the explicitly-unrolled IBM circuit and one for the REPEAT-block
IBM circuit — and collects their logical error rates in a single sinter run.
Results are written to CSV files under ``output.output_root`` in the YAML
config, with a ``circuit_variant`` column distinguishing the two circuit
implementations.

Config format is identical to the standard memory experiment config (same
YAML schema), with the following constraints:
  - ``experiment.circuit`` must be ``bivariate_bicycle_code_Z``
  - ``experiment.noise_mode`` must be ``theoretical``
The ``experiment.ibm_reproduce`` and ``experiment.noise_model`` fields are
accepted but ignored; the experiment always builds both IBM circuit variants
with inline IBM-style noise.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


def main() -> None:
    repo_root = _ensure_src_on_path()

    parser = argparse.ArgumentParser(
        description=(
            "Run the unroll-vs-repeat BB circuit comparison experiment via sinter. "
            "Generates two tasks per (p, decoder) combination — one for the "
            "explicitly-unrolled IBM circuit and one for the REPEAT-block IBM circuit — "
            "and compares their logical error rates."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    from qec_decoder_sim.runner.config import load_config
    from qec_decoder_sim.runner.unroll_vs_repeat_collecter import collect_and_write
    from qec_decoder_sim.runner.unroll_vs_repeat_task_generator import build_tasks_for_comparison

    default_config = repo_root / "scripts_shell" / "config_unroll_vs_repeat_example.yaml"
    config_path = args.config or default_config
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    cfg, decoder_variants, _write_plots = load_config(config_path, repo_root=repo_root)

    tasks, custom_decoders = build_tasks_for_comparison(
        cfg=cfg, decoder_variants=decoder_variants
    )
    summary = collect_and_write(
        tasks=tasks,
        custom_decoders=custom_decoders,
        cfg=cfg,
        decoder_variants=decoder_variants,
        started_at=datetime.now(timezone.utc),
    )

    summary["config_path"] = str(config_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
