from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sinter

from qec_decoder_sim.runner.config import DecoderVariant, ExperimentConfig


def collect_and_write(
	*,
	tasks: list[sinter.Task],
	custom_decoders: dict[str, sinter.Decoder],
	cfg: ExperimentConfig,
	decoder_variants: dict[str, DecoderVariant],
	started_at: datetime | None = None,
) -> dict[str, Any]:
	started_at = started_at or datetime.now(timezone.utc)

	cfg.output_root.mkdir(parents=True, exist_ok=True)
	experiment_dir = cfg.output_root / _experiment_dir_name(cfg, started_at)
	experiment_dir.mkdir(parents=True, exist_ok=True)

	save_resume_path = experiment_dir / "sinter_collect_resume.csv"
	stats = sinter.collect(
		tasks=tasks,
		custom_decoders=custom_decoders,
		num_workers=cfg.processes,
		print_progress=cfg.print_progress,
		save_resume_filepath=save_resume_path,
	)

	rows = _task_stats_to_records(list(stats))
	all_table_path = experiment_dir / "sinter_stats_table.csv"
	_write_csv(all_table_path, rows)

	artifacts: dict[str, dict[str, str]] = {}
	for decoder_name in cfg.decoders:
		decoder_dir = experiment_dir / f"decoder={decoder_name}"
		decoder_dir.mkdir(parents=True, exist_ok=True)

		decoder_rows = [row for row in rows if row.get("decoder") == decoder_name]
		decoder_table_path = decoder_dir / "sinter_stats_table.csv"
		_write_csv(decoder_table_path, decoder_rows)

		artifacts[decoder_name] = {
			"sinter_stats_table_csv": str(decoder_table_path),
		}

	summary = {
		"started_at_utc": started_at.isoformat(),
		"finished_at_utc": datetime.now(timezone.utc).isoformat(),
		"experiment_dir": str(experiment_dir),
		"all_stats_table_csv": str(all_table_path),
		"config": _build_config_summary(cfg),
		"decoder_variants": _build_decoder_variant_summary(decoder_variants),
		"artifacts": artifacts,
	}

	summary_path = experiment_dir / "summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return summary


def _task_stats_to_records(stats: list[sinter.TaskStats]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for stat in stats:
		metadata = stat.json_metadata if isinstance(stat.json_metadata, dict) else {}
		shots = int(stat.shots)
		errors = int(stat.errors)
		num_rounds = int(metadata.get("num_rounds", 1) or 1)
		ler = errors / max(shots, 1)
		ler_per_round = 1.0 - (1.0 - ler) ** (1.0 / max(num_rounds, 1))

		rows.append(
			{
				"decoder": stat.decoder,
				"p": float(metadata.get("p", float("nan"))),
				"shots": shots,
				"errors": errors,
				"discards": int(stat.discards),
				"seconds": float(stat.seconds),
				"basis": metadata.get("basis"),
				"num_rounds": num_rounds,
				"l": metadata.get("l"),
				"m": metadata.get("m"),
				"noise_mode": metadata.get("noise_mode"),
				"noise_model": metadata.get("noise_model"),
				"noise_sigma_rel": metadata.get("noise_sigma_rel"),
				"num_circuit_samples": metadata.get("num_circuit_samples"),
				"circuit_sample_index": metadata.get("circuit_sample_index"),
				"circuit_seed": metadata.get("circuit_seed"),
				"logical_error_rate": ler,
				"logical_error_rate_per_round": ler_per_round,
			}
		)
	return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	if not rows:
		path.write_text("", encoding="utf-8")
		return
	fieldnames = list(rows[0].keys())
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _build_config_summary(cfg: ExperimentConfig) -> dict[str, Any]:
	summary = asdict(cfg)
	summary["output_root"] = str(cfg.output_root)
	return summary


def _build_decoder_variant_summary(variants: dict[str, DecoderVariant]) -> dict[str, Any]:
	return {
		alias: {
			"type": variant.decoder_type,
			"params": variant.params,
		}
		for alias, variant in variants.items()
	}


def _experiment_dir_name(cfg: ExperimentConfig, started_at: datetime) -> str:
	ts = started_at.strftime("%Y%m%d-%H%M%S")
	return f"memory_bb_l{cfg.l}_m{cfg.m}_{cfg.basis}_{ts}"
