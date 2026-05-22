from __future__ import annotations

from typing import Any

import sinter
import stim

from qec_decoder_sim.decoder.sinter_registry import BP_DECODER_TYPES, resolve_decoder
from qec_decoder_sim.runner.circuit_builder import build_circuit, resolve_num_rounds
from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE, DecoderVariant, ExperimentConfig
from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis


def build_tasks_and_decoders(
    *,
    cfg: ExperimentConfig,
    decoder_variants: dict[str, DecoderVariant],
) -> tuple[list[sinter.Task], dict[str, sinter.Decoder]]:
    """Build sinter Tasks and a custom-decoder registry for one experiment run.

    Sweeps over all combinations of distance × p_value × circuit_sample × decoder.
    Each sinter.Task's ``json_metadata`` contains the full circuit identity so that
    results can be restored and grouped without re-reading config files.

    The DEM is generated with ``decompose_errors=True`` for MWPM-type decoders and
    ``decompose_errors=False`` (with ``ignore_decomposition_failures=True``) for all
    others.

    When ``cfg.xyz_decoding`` is ``False`` and the decoder is BP-type, the circuit
    is filtered to remove opposite-basis detectors before the DEM is built.
    """
    resolved = {alias: resolve_decoder(decoder_variants[alias]) for alias in cfg.decoders}

    # Collect custom decoder objects (built-in decoders have decoder=None)
    custom_decoders: dict[str, sinter.Decoder] = {}
    for alias, res in resolved.items():
        if res.decoder is not None:
            custom_decoders[res.sinter_name] = res.decoder

    tasks: list[sinter.Task] = []
    distance_sweep = _distance_sweep(cfg)

    for d_index, distance in enumerate(distance_sweep):
        for p_index, p in enumerate(cfg.p_values):
            shots_for_p = cfg.shots_by_p[p_index]
            num_samples = cfg.num_circuit_samples if cfg.noise_mode == "realistic" else 1
            shots_split = _split_shots(int(shots_for_p), int(num_samples))

            if cfg.noise_mode == "realistic":
                if cfg.circuit_seeds is not None:
                    sample_seeds = cfg.circuit_seeds
                else:
                    # Include distance in seed so different distances get distinct samples
                    base = cfg.circuit_seed_base + (
                        d_index * len(cfg.p_values) + p_index
                    ) * num_samples
                    sample_seeds = tuple(int(base + i) for i in range(num_samples))
            else:
                sample_seeds = (None,)  # type: ignore[assignment]

            num_rounds = resolve_num_rounds(
                cfg.num_rounds_spec,
                distance=distance if distance is not None else 1,
            )

            for sample_index in range(num_samples):
                circuit_seed = (
                    None
                    if cfg.noise_mode != "realistic"
                    else int(sample_seeds[sample_index])  # type: ignore[index]
                )
                base_circuit = build_circuit(
                    cfg=cfg,
                    p=float(p),
                    circuit_seed=circuit_seed,
                    distance=distance,
                    num_rounds=num_rounds,
                )

                base_meta = _build_task_metadata(
                    cfg=cfg,
                    p=float(p),
                    distance=distance,
                    num_rounds=num_rounds,
                )
                if cfg.noise_mode == "realistic":
                    base_meta = {
                        **base_meta,
                        "noise_sigma_rel": float(cfg.noise_sigma_rel),
                        "num_circuit_samples": int(num_samples),
                        "circuit_sample_index": int(sample_index),
                        "circuit_seed": int(circuit_seed) if circuit_seed is not None else None,
                    }

                for alias in cfg.decoders:
                    variant = decoder_variants[alias]
                    res = resolved[alias]

                    task_circuit = _prepare_circuit(base_circuit, cfg, variant)
                    dem = task_circuit.detector_error_model(
                        decompose_errors=res.needs_decompose_errors,
                        ignore_decomposition_failures=not res.needs_decompose_errors,
                    )

                    tasks.append(
                        sinter.Task(
                            circuit=task_circuit,
                            detector_error_model=dem,
                            decoder=res.sinter_name,
                            json_metadata=base_meta,
                            collection_options=sinter.CollectionOptions(
                                max_shots=int(shots_split[sample_index])
                            ),
                        )
                    )

    return tasks, custom_decoders


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _distance_sweep(cfg: ExperimentConfig) -> list[int | None]:
    """Return the list of distance values to iterate over.

    For BB codes there is no distance axis, so returns ``[None]``.
    For topological codes returns the distances tuple from the config.
    """
    if cfg.circuit_type == BB_CIRCUIT_TYPE:
        return [None]
    assert cfg.distances is not None
    return list(cfg.distances)


def _prepare_circuit(
    base_circuit: stim.Circuit,
    cfg: ExperimentConfig,
    variant: DecoderVariant,
) -> stim.Circuit:
    """Optionally filter the circuit to a single Pauli basis for BP decoders.

    When xyz_decoding=False and the decoder is a BP type, detectors of the
    opposite basis are removed so that the check matrix only spans one basis.
    MWPM and Chromobius operate on both bases natively and are unaffected.
    """
    if not cfg.xyz_decoding and variant.decoder_type in BP_DECODER_TYPES:
        remove_basis = "X" if cfg.basis == "z" else "Z"
        return filter_circuit_by_basis(base_circuit, remove_basis)
    return base_circuit


def _build_task_metadata(
    *,
    cfg: ExperimentConfig,
    p: float,
    distance: int | None,
    num_rounds: int,
) -> dict[str, Any]:
    """Build the json_metadata dict for a sinter.Task.

    All information needed to reconstruct the circuit identity is included so
    that results can be analysed without re-loading the original config file.
    """
    meta: dict[str, Any] = {
        "p": p,
        "basis": cfg.basis,
        "num_rounds": num_rounds,
        "num_rounds_spec": cfg.num_rounds_spec,
        "circuit_type": cfg.circuit_type,
        "noise_mode": cfg.noise_mode,
        "noise_model": cfg.noise_model,
        "xyz_decoding": cfg.xyz_decoding,
    }
    if cfg.circuit_type == BB_CIRCUIT_TYPE:
        meta.update(
            {
                "l": cfg.l,
                "m": cfg.m,
                "a_x_pows": list(cfg.a_x_pows),  # type: ignore[arg-type]
                "a_y_pows": list(cfg.a_y_pows),  # type: ignore[arg-type]
                "b_x_pows": list(cfg.b_x_pows),  # type: ignore[arg-type]
                "b_y_pows": list(cfg.b_y_pows),  # type: ignore[arg-type]
                "ibm_reproduce": cfg.ibm_reproduce,
                "use_both": cfg.use_both,
            }
        )
    else:
        meta["distance"] = distance
    return meta


def _split_shots(total: int, parts: int) -> tuple[int, ...]:
    if parts < 1:
        raise ValueError("num_circuit_samples must be >= 1")
    if total < 1:
        raise ValueError("shots must be >= 1")
    base = total // parts
    remainder = total % parts
    return tuple(base + (1 if i < remainder else 0) for i in range(parts))
