"""Circuit construction for the memory experiment runner.

This module is the single place that knows *how* to build a Stim circuit from
an ExperimentConfig.  It decouples circuit construction from task scheduling
so that task_genertor.py can focus on sinter task/decoder assembly.
"""
from __future__ import annotations

from dataclasses import replace

import stim

from qec_decoder_sim.circuits import BivariateBicycleCode
from qec_decoder_sim.circuits._topological_memory_circuits import make_topological_memory_circuit
from qec_decoder_sim.codes.code_factory import BBCodeSpec
from qec_decoder_sim.noise.noise_model import (
    NoiseParameters,
    insert_realistic_noise,
    si1000_circuit_level,
    uniform_circuit_level,
)
from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE, ExperimentConfig

_NOISE_MODEL_ALIASES: dict[str, str] = {
    "uniform": "uniform-cln",
    "uniform-cln": "uniform-cln",
    "si1000": "si1000",
    "code_capacity": "code_capacity",
    "bitflip": "bitflip",
    "phenomenological": "phenomenological",
}

_TOPOLOGICAL_NOISE_MAP: dict[str, str] = {
    "uniform-cln": "uniform",
    "si1000": "si1000",
    "bitflip": "bitflip",
    "phenomenological": "phenomenological",
}


def resolve_num_rounds(spec: str, *, distance: int) -> int:
    """Resolve a num_rounds spec string to a concrete integer for one distance.

    Args:
        spec: One of: a positive integer string (e.g. ``"12"``), ``"d"``, or
              ``"k*d"`` / ``"d*k"`` (e.g. ``"3*d"``).
        distance: The code diameter / distance used when the spec contains ``"d"``.

    Returns:
        The resolved number of rounds as a positive integer.
    """
    s = spec.strip().lower().replace(" ", "")
    if s.isdigit():
        return int(s)
    if s == "d":
        return distance
    if s.endswith("*d") and s[:-2].isdigit():
        return int(s[:-2]) * distance
    if s.startswith("d*") and s[2:].isdigit():
        return int(s[2:]) * distance
    raise ValueError(
        f"Invalid num_rounds spec {spec!r}. "
        "Use a positive integer, 'd', or 'k*d' (e.g. '3*d')."
    )


def build_circuit(
    cfg: ExperimentConfig,
    p: float,
    circuit_seed: int | None,
    *,
    distance: int | None,
    num_rounds: int,
) -> stim.Circuit:
    """Build and return a noisy Stim circuit for one point in the experiment sweep.

    Args:
        cfg: Experiment configuration (circuit type, noise model, code params, etc.).
        p: Noise strength (error probability) for this sweep point.
        circuit_seed: RNG seed for realistic noise mode; None in theoretical mode.
        distance: Code diameter for topological codes; ``None`` for BB codes.
        num_rounds: Resolved number of syndrome-extraction rounds (already computed
            from ``cfg.num_rounds_spec`` and ``distance`` by the caller).

    Returns:
        A noisy ``stim.Circuit`` ready to be sampled or converted to a DEM.
    """
    if cfg.circuit_type == BB_CIRCUIT_TYPE:
        return _build_bb_circuit(cfg, p, circuit_seed, num_rounds=num_rounds)
    else:
        assert distance is not None
        return _build_topological_circuit(cfg, p, distance=distance, num_rounds=num_rounds)


# ---------------------------------------------------------------------------
# BB code path
# ---------------------------------------------------------------------------

def _build_bb_circuit(
    cfg: ExperimentConfig,
    p: float,
    circuit_seed: int | None,
    *,
    num_rounds: int,
) -> stim.Circuit:
    noise_model = _resolve_noise_model(cfg.noise_model)
    spec = _build_bbcode_spec(cfg)

    if cfg.noise_mode == "realistic":
        if noise_model == "code_capacity":
            raise ValueError(
                "noise_model='code_capacity' is not supported with noise_mode='realistic'"
            )
        ideal_builder = BivariateBicycleCode(
            bbcode_spec=spec,
            noise_model=noise_model,
            p=p,
            num_rounds=num_rounds,
            basis=cfg.basis,
            use_both=cfg.use_both,
            ibm_reproduce=cfg.ibm_reproduce,
        )
        ideal_result = ideal_builder.build_ideal()
        noise = _resolve_noise_parameters(noise_model, p=p, sigma_rel=cfg.noise_sigma_rel)
        return insert_realistic_noise(
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
            num_rounds=num_rounds,
            basis=cfg.basis,
            use_both=cfg.use_both,
            ibm_reproduce=cfg.ibm_reproduce,
        )
        return builder.build().circuit


def _build_bbcode_spec(cfg: ExperimentConfig) -> BBCodeSpec:
    assert cfg.l is not None and cfg.m is not None
    assert cfg.a_x_pows is not None and cfg.a_y_pows is not None
    assert cfg.b_x_pows is not None and cfg.b_y_pows is not None
    return BBCodeSpec(
        l=cfg.l,
        m=cfg.m,
        a_x_pows=cfg.a_x_pows,
        a_y_pows=cfg.a_y_pows,
        b_x_pows=cfg.b_x_pows,
        b_y_pows=cfg.b_y_pows,
    )


# ---------------------------------------------------------------------------
# Topological code path
# ---------------------------------------------------------------------------

def _build_topological_circuit(
    cfg: ExperimentConfig,
    p: float,
    *,
    distance: int,
    num_rounds: int,
) -> stim.Circuit:
    if cfg.noise_mode == "realistic":
        raise ValueError(
            "noise_mode='realistic' is not supported for topological circuit types. "
            "Use noise_mode='theoretical'."
        )

    noise_model_key = _resolve_noise_model(cfg.noise_model)
    # make_topological_memory_circuit expects the short names "uniform"/"si1000"/"bitflip"
    topo_noise = _TOPOLOGICAL_NOISE_MAP.get(noise_model_key, noise_model_key)

    return make_topological_memory_circuit(
        style=cfg.circuit_type,
        rounds=num_rounds,
        diameter=distance,
        noise_model=topo_noise,
        noise_strength=p,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
