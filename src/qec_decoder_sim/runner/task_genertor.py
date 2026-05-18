from __future__ import annotations

from dataclasses import replace
from typing import Any

import sinter
import stim

from qec_decoder_sim.circuits import BivariateBicycleCode
from qec_decoder_sim.codes.code_factory import BBCodeSpec
from qec_decoder_sim.decoder.sinter_registry import build_custom_sinter_decoders
from qec_decoder_sim.noise.noise_model import (
	NoiseParameters,
	insert_realistic_noise,
	si1000_circuit_level,
	uniform_circuit_level,
)
from qec_decoder_sim.runner.config import DecoderVariant, ExperimentConfig

_NOISE_MODEL_ALIASES = {
	"uniform": "uniform-cln",
	"uniform-cln": "uniform-cln",
	"si1000": "si1000",
	"code_capacity": "code_capacity",
}


def build_tasks_and_decoders(
	*,
	cfg: ExperimentConfig,
	decoder_variants: dict[str, DecoderVariant],
) -> tuple[list[sinter.Task], dict[str, sinter.Decoder]]:
	custom_decoders = _build_custom_decoders(cfg.decoders, decoder_variants)

	tasks: list[sinter.Task] = []
	for p_index, p in enumerate(cfg.p_values):
		shots_for_p = cfg.shots_by_p[p_index]
		num_samples = cfg.num_circuit_samples if cfg.noise_mode == "realistic" else 1
		shots_split = _split_shots(int(shots_for_p), int(num_samples))

		if cfg.noise_mode == "realistic":
			if cfg.circuit_seeds is not None:
				sample_seeds = cfg.circuit_seeds
			else:
				sample_seeds = tuple(int(cfg.circuit_seed_base + p_index * num_samples + i) for i in range(num_samples))
		else:
			sample_seeds = (None,)  # type: ignore[assignment]

		for sample_index in range(num_samples):
			circuit_seed = None if cfg.noise_mode != "realistic" else int(sample_seeds[sample_index])  # type: ignore[index]
			circuit, dem = _build_problem(cfg=cfg, p=float(p), circuit_seed=circuit_seed)

			base_meta = _build_task_metadata(
				cfg=cfg,
				p=float(p),
			)
			if cfg.noise_mode == "realistic":
				base_meta = {
					**base_meta,
					"noise_sigma_rel": float(cfg.noise_sigma_rel),
					"num_circuit_samples": int(num_samples),
					"circuit_sample_index": int(sample_index),
					"circuit_seed": int(circuit_seed) if circuit_seed is not None else None,
				}

			for decoder_name in cfg.decoders:
				tasks.append(
					sinter.Task(
						circuit=circuit,
						detector_error_model=dem,
						decoder=decoder_name,
						json_metadata=base_meta,
						collection_options=sinter.CollectionOptions(max_shots=int(shots_split[sample_index])),
					)
				)

	return tasks, custom_decoders


def _build_custom_decoders(
	aliases: tuple[str, ...],
	variants: dict[str, DecoderVariant],
) -> dict[str, sinter.Decoder]:
	custom_decoders: dict[str, sinter.Decoder] = {}
	for alias in aliases:
		variant = variants[alias]
		decoders = build_custom_sinter_decoders(
			selected_decoders=(variant.decoder_type,),
			decoder_params={variant.decoder_type: variant.params},
		)
		custom_decoders[alias] = decoders[variant.decoder_type]
	return custom_decoders


def _build_problem(
	*,
	cfg: ExperimentConfig,
	p: float,
	circuit_seed: int | None,
) -> tuple[stim.Circuit, stim.DetectorErrorModel]:
	noise_model = _resolve_noise_model(cfg.noise_model)
	spec = _build_bbcode_spec(cfg)

	if cfg.noise_mode == "realistic":
		if noise_model == "code_capacity":
			raise ValueError("noise_model='code_capacity' is not supported with noise_mode='realistic'")
		ideal_builder = BivariateBicycleCode(
			bbcode_spec=spec,
			noise_model=noise_model,
			p=p,
			num_rounds=cfg.num_rounds,
			basis=cfg.basis,
		)
		ideal_result = ideal_builder.build_ideal()
		noise = _resolve_noise_parameters(noise_model, p=p, sigma_rel=cfg.noise_sigma_rel)
		circuit = insert_realistic_noise(
			ideal_result.circuit,
			noise,
			seed=circuit_seed,
			data_qubits=set(ideal_result.metadata.data_qubits),
		)
	else:
		builder = BivariateBicycleCode(
			bbcode_spec=spec,
			noise_model=noise_model,
			p=p,
			num_rounds=cfg.num_rounds,
			basis=cfg.basis,
		)
		circuit = builder.build().circuit

	dem = circuit.detector_error_model(
		decompose_errors=False,
		ignore_decomposition_failures=True,
	)
	return circuit, dem


def _build_bbcode_spec(cfg: ExperimentConfig) -> BBCodeSpec:
	return BBCodeSpec(
		l=cfg.l,
		m=cfg.m,
		a_x_pows=cfg.a_x_pows,
		a_y_pows=cfg.a_y_pows,
		b_x_pows=cfg.b_x_pows,
		b_y_pows=cfg.b_y_pows,
	)


def _build_task_metadata(*, cfg: ExperimentConfig, p: float) -> dict[str, Any]:
	return {
		"p": p,
		"basis": cfg.basis,
		"num_rounds": cfg.num_rounds,
		"l": cfg.l,
		"m": cfg.m,
		"a_x_pows": list(cfg.a_x_pows),
		"a_y_pows": list(cfg.a_y_pows),
		"b_x_pows": list(cfg.b_x_pows),
		"b_y_pows": list(cfg.b_y_pows),
		"noise_mode": cfg.noise_mode,
		"noise_model": cfg.noise_model,
	}


def _split_shots(total: int, parts: int) -> tuple[int, ...]:
	if parts < 1:
		raise ValueError("num_circuit_samples must be >= 1")
	if total < 1:
		raise ValueError("shots must be >= 1")
	base = total // parts
	remainder = total % parts
	return tuple(base + (1 if i < remainder else 0) for i in range(parts))


def _resolve_noise_model(raw: str) -> str:
	key = raw.strip().lower()
	if key not in _NOISE_MODEL_ALIASES:
		raise ValueError(
			"experiment.noise_model must be one of "
			f"{sorted(_NOISE_MODEL_ALIASES.keys())}; got {raw!r}"
		)
	return _NOISE_MODEL_ALIASES[key]


def _resolve_noise_parameters(noise_model: str, *, p: float, sigma_rel: float) -> NoiseParameters:
	if noise_model == "uniform-cln":
		noise = uniform_circuit_level(p)
	elif noise_model == "si1000":
		noise = si1000_circuit_level(p)
	else:
		raise ValueError("Realistic noise mode only supports 'uniform' or 'si1000' models")
	return replace(noise, sigma_rel=sigma_rel)
