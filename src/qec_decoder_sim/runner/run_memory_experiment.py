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

    parser = argparse.ArgumentParser(description="Run a bivariate bicycle memory experiment via sinter.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    from qec_decoder_sim.runner.collecter import collect_and_write
    from qec_decoder_sim.runner.config import default_config_path, load_config
    from qec_decoder_sim.runner.task_genertor import build_tasks_and_decoders

    config_path = args.config or default_config_path(repo_root=repo_root)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    cfg, decoder_variants, _write_plots = load_config(config_path, repo_root=repo_root)

    tasks, custom_decoders = build_tasks_and_decoders(cfg=cfg, decoder_variants=decoder_variants)
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
