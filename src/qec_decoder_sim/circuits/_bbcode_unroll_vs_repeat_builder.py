"""Circuit builders for the unroll-vs-repeat comparison experiment.

Provides two IBM-style noisy BB circuit builders with identical gate
schedules and noise insertions but different structural representations:

- ``build_bb_ibm_unrolled``: explicit loop over all rounds (original approach)
- ``build_bb_ibm_repeat``:   first round flat, remaining rounds in a stim
  REPEAT block (current production approach)

Both functions have the same signature and return type.  The only
structural difference is in how the repeated syndrome-extraction rounds
are encoded in the stim circuit — the gate-level noise and qubit
connectivity are identical.

Why this matters:
  With explicit unrolling, stim merges error mechanisms that share the
  same detector pattern across rounds, producing a coarser DEM factor
  graph.  The REPEAT block keeps every round's error mechanisms separate,
  preserving more variable nodes for belief-propagation decoders.
  This module exists specifically to measure the resulting difference in
  logical error rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import stim

from qec_decoder_sim.circuits._bbcode_circuit_builder import (
    _append_round_ibm,
    compute_check_colors,
    BBCircuitBuildResult,
    BBCircuitMetadata,
)
from qec_decoder_sim.codes.code_factory import BBCodeSpec, build_bb_block_code


def _nnz_positions(matrix) -> np.ndarray:
    rows, cols = matrix.nonzero()
    return cols[np.argsort(rows)]


@dataclass
class _IBMCircuitSetup:
    circuit: stim.Circuit
    round_args: dict[str, Any]
    data_qubits: tuple[int, ...]
    n: int
    z_basis: bool
    pcm: Any
    logical_pcm: Any
    colors: list[int]
    check_offset: int


def _prepare_ibm_setup(
    spec: BBCodeSpec,
    *,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
    p: float,
) -> _IBMCircuitSetup:
    """Build the initial circuit and compute all round-invariant parameters.

    This encapsulates the setup that is identical for both the unrolled and
    REPEAT-block variants: qubit initialization, preparation errors, and the
    round-args dict passed to ``_append_round_ibm``.
    """
    block_code = build_bb_block_code(spec)
    colors_z = compute_check_colors(block_code.code.hz)
    colors_x = compute_check_colors(block_code.code.hx)

    if len(block_code.a_blocks) != 3 or len(block_code.b_blocks) != 3:
        raise ValueError(
            "Current BB Stim schedule requires exactly three A blocks and three B blocks."
        )

    z_basis = basis.lower() == "z"
    n = block_code.code.hx.shape[1]

    A1, A2, A3 = (_nnz_positions(block_code.a_blocks[i]) for i in range(3))
    B1, B2, B3 = (_nnz_positions(block_code.b_blocks[i]) for i in range(3))
    A1_T, A2_T, A3_T = (_nnz_positions(block_code.a_blocks[i].T) for i in range(3))
    B1_T, B2_T, B3_T = (_nnz_positions(block_code.b_blocks[i].T) for i in range(3))

    x_check_offset = 0
    l_data_offset = n // 2
    r_data_offset = n
    z_check_offset = 3 * n // 2

    circuit = stim.Circuit()

    for i in range(n // 2):
        circuit.append("R", [x_check_offset + i])
        circuit.append("R", [z_check_offset + i])
        circuit.append("X_ERROR", [x_check_offset + i], [p])
        circuit.append("X_ERROR", [z_check_offset + i], [p])

    for i in range(n):
        circuit.append("R" if z_basis else "RX", [l_data_offset + i])
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", [l_data_offset + i], [p])

    data_qubits = tuple(range(l_data_offset, l_data_offset + n))
    circuit.append("TICK")

    round_args: dict[str, Any] = dict(
        n=n,
        A1=A1, A2=A2, A3=A3,
        B1=B1, B2=B2, B3=B3,
        A1_T=A1_T, A2_T=A2_T, A3_T=A3_T,
        B1_T=B1_T, B2_T=B2_T, B3_T=B3_T,
        x_check_offset=x_check_offset,
        l_data_offset=l_data_offset,
        r_data_offset=r_data_offset,
        z_check_offset=z_check_offset,
        z_basis=z_basis,
        use_both=use_both,
        hzh=hzh,
        colors_z=colors_z,
        colors_x=colors_x,
        p=p,
    )

    pcm = block_code.code.hz if z_basis else block_code.code.hx
    logical_pcm = block_code.code.logicals.lz if z_basis else block_code.code.logicals.lx
    colors = colors_z if z_basis else colors_x
    check_offset = z_check_offset if z_basis else x_check_offset

    return _IBMCircuitSetup(
        circuit=circuit,
        round_args=round_args,
        data_qubits=data_qubits,
        n=n,
        z_basis=z_basis,
        pcm=pcm,
        logical_pcm=logical_pcm,
        colors=colors,
        check_offset=check_offset,
    )


def _finalize_ibm_circuit(setup: _IBMCircuitSetup, *, num_rounds: int) -> BBCircuitBuildResult:
    """Append final data-qubit measurements, end-of-experiment detectors, and
    logical-observable declarations; return the completed build result."""
    circuit = setup.circuit
    n = setup.n
    z_basis = setup.z_basis
    l_data_offset: int = setup.round_args["l_data_offset"]
    x_check_offset: int = setup.round_args["x_check_offset"]
    z_check_offset: int = setup.round_args["z_check_offset"]

    for i in range(n):
        circuit.append("M" if z_basis else "MX", [l_data_offset + i])

    for i, stabilizer in enumerate(setup.pcm):
        nnz = np.nonzero(stabilizer)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        targets.append(
            stim.target_rec(-n - n + i) if z_basis else stim.target_rec(-n - n // 2 + i)
        )
        circuit.append("DETECTOR", targets, [setup.check_offset + i, num_rounds, setup.colors[i]])

    for i, logical in enumerate(setup.logical_pcm):
        nnz = np.nonzero(logical)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        circuit.append("OBSERVABLE_INCLUDE", targets, [i])

    metadata = BBCircuitMetadata(
        data_qubits=setup.data_qubits,
        x_ancilla_qubits=tuple(range(x_check_offset, x_check_offset + n // 2)),
        z_ancilla_qubits=tuple(range(z_check_offset, z_check_offset + n // 2)),
        num_rounds=num_rounds,
        basis="z" if z_basis else "x",
    )
    return BBCircuitBuildResult(circuit=circuit, metadata=metadata)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_bb_ibm_unrolled(
    spec: BBCodeSpec,
    *,
    num_rounds: int,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
    p: float,
) -> BBCircuitBuildResult:
    """Build a noisy BB IBM circuit by explicitly unrolling all syndrome rounds.

    This is the original implementation before the REPEAT-block refactor.
    Every round is appended as a flat sequence of instructions.  Stim merges
    error mechanisms that share the same detector pattern across rounds,
    producing a coarser DEM factor graph.  This can reduce the number of
    variable nodes available to belief-propagation decoders.

    Apart from the flat structure, the circuit is identical to
    ``build_bb_ibm_repeat``: same gate schedule, same inline noise insertions,
    same final measurements, detectors, and logical observables.
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    setup = _prepare_ibm_setup(spec, basis=basis, use_both=use_both, hzh=hzh, p=p)

    for round_index in range(num_rounds):
        _append_round_ibm(
            setup.circuit, **setup.round_args,
            repeat=round_index > 0, round_index=round_index,
        )

    return _finalize_ibm_circuit(setup, num_rounds=num_rounds)


def build_bb_ibm_repeat(
    spec: BBCodeSpec,
    *,
    num_rounds: int,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
    p: float,
) -> BBCircuitBuildResult:
    """Build a noisy BB IBM circuit with middle rounds inside a stim REPEAT block.

    The first round is appended flat (``repeat=False``).  All subsequent
    rounds are placed inside a ``(num_rounds - 1) * body`` REPEAT block where
    the body contains one round with ``repeat=True`` followed by
    ``SHIFT_COORDS(0, 1, 0)`` to keep the round coordinate correct in
    DETECTOR visualisations.

    Because stim treats each REPEAT iteration as structurally independent,
    error mechanisms from different rounds are *not* merged in the DEM.  This
    preserves more variable nodes for belief-propagation decoders compared to
    the explicitly unrolled variant.
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    setup = _prepare_ibm_setup(spec, basis=basis, use_both=use_both, hzh=hzh, p=p)

    _append_round_ibm(setup.circuit, **setup.round_args, repeat=False, round_index=0)
    if num_rounds > 1:
        rep_body = stim.Circuit()
        _append_round_ibm(rep_body, **setup.round_args, repeat=True, round_index=0)
        rep_body.append("SHIFT_COORDS", [], [0.0, 1.0, 0.0])
        setup.circuit += (num_rounds - 1) * rep_body

    return _finalize_ibm_circuit(setup, num_rounds=num_rounds)
