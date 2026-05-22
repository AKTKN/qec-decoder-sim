from __future__ import annotations

from typing import Any

import sinter
import stim

from qec_decoder_sim.decoder.sinter_registry import BP_DECODER_TYPES, resolve_decoder
from qec_decoder_sim.runner.circuit_builder import build_circuit
from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE, DecoderVariant, ExperimentConfig
from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis


def build_tasks_and_decoders(
    *,
    cfg: ExperimentConfig,
    decoder_variants: dict[str, DecoderVariant],
) -> tuple[list[sinter.Task], dict[str, sinter.Decoder]]:
    """Build sinter Tasks and a custom-decoder registry for one experiment run.

    For each (p_value × circuit_sample × decoder_alias) combination a
    ``sinter.Task`` is created.  The DEM attached to each task is generated
    with ``decompose_errors=True`` for MWPM-type decoders, and
    ``decompose_errors=False`` (with ``ignore_decomposition_failures=True``)
    for all others.

    When ``cfg.xyz_decoding`` is ``False`` and the decoder is BP-type, the
    circuit is filtered to remove the opposite-basis detectors before the DEM
    is built, so that sinter's sampler only observes a single Pauli basis.
    """
    resolved = {alias: resolve_decoder(decoder_variants[alias]) for alias in cfg.decoders}

    # Collect custom decoder objects (built-in decoders have decoder=None)
    custom_decoders: dict[str, sinter.Decoder] = {}
    for alias, res in resolved.items():
        if res.decoder is not None:
            custom_decoders[res.sinter_name] = res.decoder

    tasks: list[sinter.Task] = []

    for p_index, p in enumerate(cfg.p_values):
        shots_for_p = cfg.shots_by_p[p_index]
        num_samples = cfg.num_circuit_samples if cfg.noise_mode == "realistic" else 1
        shots_split = _split_shots(int(shots_for_p), int(num_samples))

        if cfg.noise_mode == "realistic":
            if cfg.circuit_seeds is not None:
                sample_seeds = cfg.circuit_seeds
            else:
                sample_seeds = tuple(
                    int(cfg.circuit_seed_base + p_index * num_samples + i)
                    for i in range(num_samples)
                )
        else:
            sample_seeds = (None,)  # type: ignore[assignment]

        for sample_index in range(num_samples):
            circuit_seed = (
                None
                if cfg.noise_mode != "realistic"
                else int(sample_seeds[sample_index])  # type: ignore[index]
            )
            base_circuit = build_circuit(cfg=cfg, p=float(p), circuit_seed=circuit_seed)

            base_meta = _build_task_metadata(cfg=cfg, p=float(p))
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


def _build_task_metadata(*, cfg: ExperimentConfig, p: float) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "p": p,
        "basis": cfg.basis,
        "num_rounds": cfg.num_rounds,
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
        meta["distance"] = cfg.distance
    return meta


def _split_shots(total: int, parts: int) -> tuple[int, ...]:
    if parts < 1:
        raise ValueError("num_circuit_samples must be >= 1")
    if total < 1:
        raise ValueError("shots must be >= 1")
    base = total // parts
    remainder = total % parts
    return tuple(base + (1 if i < remainder else 0) for i in range(parts))
