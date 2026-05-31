"""Task generator for the unroll-vs-repeat BB circuit comparison experiment.

For each point in the sweep (p × decoder), this module generates *two*
``sinter.Task`` objects — one for the explicitly-unrolled IBM circuit and one
for the REPEAT-block IBM circuit — so their logical error rates can be compared
directly from the same sinter collection run.

Reuses:
  - ``ExperimentConfig`` / ``DecoderVariant`` from ``config.py``
  - ``resolve_decoder`` / ``BP_DECODER_TYPES`` from ``sinter_registry.py``
  - ``resolve_num_rounds`` from ``circuit_builder.py``
  - ``filter_circuit_by_basis`` from ``dem_utils.py``

New:
  - ``build_bb_ibm_unrolled`` / ``build_bb_ibm_repeat`` from the dedicated
    circuit builder ``_bbcode_unroll_vs_repeat_builder.py``
"""
from __future__ import annotations

from typing import Any

import sinter

from qec_decoder_sim.circuits._bbcode_unroll_vs_repeat_builder import (
    build_bb_ibm_repeat,
    build_bb_ibm_unrolled,
)
from qec_decoder_sim.codes.code_factory import BBCodeSpec
from qec_decoder_sim.decoder.sinter_registry import BP_DECODER_TYPES, resolve_decoder
from qec_decoder_sim.runner.circuit_builder import resolve_num_rounds
from qec_decoder_sim.runner.config import BB_CIRCUIT_TYPE, DecoderVariant, ExperimentConfig
from qec_decoder_sim.utils.dem_utils import filter_circuit_by_basis

# Label values stored in task metadata and the output CSV.
VARIANT_UNROLLED = "unrolled"
VARIANT_REPEAT = "repeat"

# IBM circuits use hzh=False (MRX directly, not H-MR-H conjugation).
_HZH = False


def build_tasks_for_comparison(
    *,
    cfg: ExperimentConfig,
    decoder_variants: dict[str, DecoderVariant],
) -> tuple[list[sinter.Task], dict[str, sinter.Decoder]]:
    """Build sinter Tasks for the unroll-vs-repeat comparison experiment.

    For each combination of p_value × decoder, two tasks are created:

    1. ``circuit_variant="unrolled"`` — IBM circuit with all rounds explicitly
       unrolled (the original implementation).
    2. ``circuit_variant="repeat"`` — IBM circuit with middle rounds inside a
       stim REPEAT block (the current production implementation).

    The ``json_metadata`` of each task includes a ``circuit_variant`` key so
    that results for the two structural variants can be separated in analysis.

    Restrictions compared to the general memory experiment:
      - Only ``circuit_type="bivariate_bicycle_code_Z"`` is supported.
      - Only ``noise_mode="theoretical"`` is supported (IBM circuits carry
        inline noise; post-processed realistic noise is out of scope here).

    Args:
        cfg: Experiment configuration.  ``circuit_type`` must be
            ``BB_CIRCUIT_TYPE`` and ``noise_mode`` must be ``"theoretical"``.
        decoder_variants: Mapping from decoder alias to ``DecoderVariant``.

    Returns:
        A tuple ``(tasks, custom_decoders)`` ready for ``sinter.collect``.
    """
    if cfg.circuit_type != BB_CIRCUIT_TYPE:
        raise ValueError(
            f"unroll-vs-repeat experiment only supports circuit_type={BB_CIRCUIT_TYPE!r}; "
            f"got {cfg.circuit_type!r}"
        )
    if cfg.noise_mode != "theoretical":
        raise ValueError(
            "unroll-vs-repeat experiment only supports noise_mode='theoretical'; "
            f"got {cfg.noise_mode!r}"
        )

    resolved = {alias: resolve_decoder(decoder_variants[alias]) for alias in cfg.decoders}

    custom_decoders: dict[str, sinter.Decoder] = {}
    for alias, res in resolved.items():
        if res.decoder is not None:
            custom_decoders[res.sinter_name] = res.decoder

    spec = _build_bbcode_spec(cfg)
    num_rounds = resolve_num_rounds(cfg.num_rounds_spec, distance=1)

    tasks: list[sinter.Task] = []

    for p_index, p in enumerate(cfg.p_values):
        shots_for_p = cfg.shots_by_p[p_index]

        unrolled_result = build_bb_ibm_unrolled(
            spec,
            num_rounds=num_rounds,
            basis=cfg.basis,
            use_both=cfg.use_both,
            hzh=_HZH,
            p=float(p),
        )
        repeat_result = build_bb_ibm_repeat(
            spec,
            num_rounds=num_rounds,
            basis=cfg.basis,
            use_both=cfg.use_both,
            hzh=_HZH,
            p=float(p),
        )

        variants = [
            (VARIANT_UNROLLED, unrolled_result.circuit),
            (VARIANT_REPEAT, repeat_result.circuit),
        ]

        base_meta = _build_base_metadata(cfg, p=float(p), num_rounds=num_rounds)

        for alias in cfg.decoders:
            variant = decoder_variants[alias]
            res = resolved[alias]

            for variant_name, base_circuit in variants:
                task_circuit = _prepare_circuit(base_circuit, cfg, variant)
                dem = task_circuit.detector_error_model(
                    decompose_errors=res.needs_decompose_errors,
                    ignore_decomposition_failures=not res.needs_decompose_errors,
                )
                meta = {**base_meta, "circuit_variant": variant_name}
                tasks.append(
                    sinter.Task(
                        circuit=task_circuit,
                        detector_error_model=dem,
                        decoder=res.sinter_name,
                        json_metadata=meta,
                        collection_options=sinter.CollectionOptions(
                            max_shots=int(shots_for_p)
                        ),
                    )
                )

    return tasks, custom_decoders


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _prepare_circuit(
    base_circuit,
    cfg: ExperimentConfig,
    variant: DecoderVariant,
):
    """Optionally filter the circuit to a single Pauli basis for BP decoders."""
    if not cfg.xyz_decoding and variant.decoder_type in BP_DECODER_TYPES:
        remove_basis = "X" if cfg.basis == "z" else "Z"
        return filter_circuit_by_basis(base_circuit, remove_basis)
    return base_circuit


def _build_base_metadata(
    cfg: ExperimentConfig,
    *,
    p: float,
    num_rounds: int,
) -> dict[str, Any]:
    """Build the json_metadata dict shared by both circuit variants at one (p, decoder) point."""
    return {
        "p": p,
        "basis": cfg.basis,
        "num_rounds": num_rounds,
        "num_rounds_spec": cfg.num_rounds_spec,
        "circuit_type": cfg.circuit_type,
        "noise_mode": cfg.noise_mode,
        "noise_model": "ibm_inline",
        "xyz_decoding": cfg.xyz_decoding,
        "l": cfg.l,
        "m": cfg.m,
        "a_x_pows": list(cfg.a_x_pows),  # type: ignore[arg-type]
        "a_y_pows": list(cfg.a_y_pows),  # type: ignore[arg-type]
        "b_x_pows": list(cfg.b_x_pows),  # type: ignore[arg-type]
        "b_y_pows": list(cfg.b_y_pows),  # type: ignore[arg-type]
        "use_both": cfg.use_both,
        "hzh": _HZH,
    }
