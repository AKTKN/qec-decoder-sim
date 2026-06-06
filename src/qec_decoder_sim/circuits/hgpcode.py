from __future__ import annotations

import concurrent.futures
import os
from typing import Union

import numpy as np
import stim

from quits import CircuitBuildOptions, ErrorModel, HgpCode
from quits.gf2_util import compute_code_distance

_MR_NAMES = {"MR", "MRZ", "MRX", "MRY"}
_MEAS_NAMES = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}

_DISTANCE_TIMEOUT_SEC = 60.0


def load_pcm(path: Union[str, os.PathLike]) -> np.ndarray:
    """Load a space-separated binary parity-check matrix from a text file."""
    return np.loadtxt(path, dtype=int)


def compute_hgp_distance(
    h: np.ndarray,
    *,
    timeout: float = _DISTANCE_TIMEOUT_SEC,
) -> int:
    """Compute the distance of the symmetric HGP code built from h.

    For a symmetric HgpCode(h, h) the quantum code distance equals the
    classical code distance of h, i.e. the minimum Hamming weight of a
    non-zero codeword in ker(h).  This is computed via quits'
    ``compute_code_distance(h)`` in a worker thread so the call can be
    interrupted after *timeout* seconds.

    Raises:
        TimeoutError: if the computation exceeds *timeout* seconds.
        ValueError: if ker(h) is trivial (distance infinite).
    """
    def _compute() -> int | float:
        return compute_code_distance(h)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_compute)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"HGP code distance computation exceeded {int(timeout)} s. "
                "Provide a smaller parity-check matrix or increase the timeout."
            )

    if np.isinf(result):
        raise ValueError(
            "The parity-check matrix h has a trivial kernel; "
            "HgpCode(h, h) encodes no logical qubits."
        )
    return int(result)


def build_hgp_stim_circuit(
    h: np.ndarray,
    *,
    basis: str,
    num_rounds: int,
    p: float,
    use_both: bool = False,
    build_graph_seed: int = 22,
) -> stim.Circuit:
    """Build a Stim memory circuit for the symmetric HGP code HgpCode(h, h).

    Args:
        h: Classical parity-check matrix; the HGP code is HgpCode(h, h).
        basis: Logical storage basis, ``"z"`` or ``"x"`` (case-insensitive).
        num_rounds: Number of noisy syndrome-extraction rounds.
        p: Uniform physical error rate applied to all gate/idle/SPAM channels.
        use_both: If True, include detectors for both X and Z stabilizers
            (passed to CircuitBuildOptions.get_all_detectors).
        build_graph_seed: Seed for the cardinal edge-orientation step.

    Returns:
        A noisy ``stim.Circuit``.
    """
    code = HgpCode(h, h)
    error_model = ErrorModel(
        idle_error=p,
        sqgate_error=p,
        tqgate_error=p,
        spam_error=p,
    )
    circuit_opts = CircuitBuildOptions(
        get_all_detectors=use_both,
        noisy_zeroth_round=True,
        noisy_final_meas=True,
    )
    return code.build_circuit(
        strategy="cardinal",
        error_model=error_model,
        num_rounds=num_rounds,
        basis=basis.upper(),
        circuit_build_options=circuit_opts,
        seed=build_graph_seed,
    )


def build_hgp_phenomenological_circuit(
    h: np.ndarray,
    *,
    basis: str,
    num_rounds: int,
    p: float,
    use_both: bool = False,
    build_graph_seed: int = 22,
) -> stim.Circuit:
    """Build a phenomenological noise Stim circuit for the symmetric HGP code.

    Noise model: data qubit bit-flip X_ERROR(p) per round + measurement error p.
    Gate and reset errors are absent. Time boundaries are noiseless.

    Specifically:
    - X_ERROR(p) is prepended at the start of every noisy round (REPEAT body).
    - X_ERROR(p) is appended just before the final data-qubit measurement.
    - Measurement error is kept (X_ERROR before MR, from spam_error=p in quits).
    - Reset error (X_ERROR after MR) is removed.
    """
    code = HgpCode(h, h)
    error_model = ErrorModel(
        idle_error=0,
        sqgate_error=0,
        tqgate_error=0,
        spam_error=p,
    )
    circuit_opts = CircuitBuildOptions(
        get_all_detectors=use_both,
        noisy_zeroth_round=False,
        noisy_final_meas=False,
    )
    base_circuit = code.build_circuit(
        strategy="cardinal",
        error_model=error_model,
        num_rounds=num_rounds,
        basis=basis.upper(),
        circuit_build_options=circuit_opts,
        seed=build_graph_seed,
    )
    return _convert_to_phenomenological(base_circuit, p=p)


# ---------------------------------------------------------------------------
# Phenomenological noise helpers
# ---------------------------------------------------------------------------

def _collect_ancilla_qubits(circuit: stim.Circuit) -> set[int]:
    """Return the set of qubit indices that are destructively measured (MR) in the REPEAT block."""
    ancilla: set[int] = set()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            for inner in inst.body_copy():
                if isinstance(inner, stim.CircuitInstruction) and inner.name in _MR_NAMES:
                    for t in inner.targets_copy():
                        if t.is_qubit_target:
                            ancilla.add(t.qubit_value)
            break
    return ancilla


def _transform_repeat_body_phenomenological(
    body: stim.Circuit,
    *,
    data_qubits: list[int],
    p: float,
) -> stim.Circuit:
    """Return a new REPEAT body with phenomenological noise applied.

    Changes made to the original spam-only body:
    1. Prepend X_ERROR(p) on data qubits (idle bit-flip per round).
    2. Remove each X_ERROR that immediately follows an MR instruction (reset error).
    All other instructions are kept as-is.
    """
    new_body = stim.Circuit()
    if data_qubits and p > 0:
        new_body.append("X_ERROR", data_qubits, [p])

    instructions = list(body)
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        new_body.append(inst)
        if (
            isinstance(inst, stim.CircuitInstruction)
            and inst.name in _MR_NAMES
            and i + 1 < len(instructions)
            and isinstance(instructions[i + 1], stim.CircuitInstruction)
            and instructions[i + 1].name == "X_ERROR"
        ):
            i += 2  # skip the post-MR X_ERROR (reset error)
            continue
        i += 1
    return new_body


def _convert_to_phenomenological(circuit: stim.Circuit, *, p: float) -> stim.Circuit:
    """Transform a quits spam-only circuit into a phenomenological noise circuit."""
    all_qubits = set(range(circuit.num_qubits))
    ancilla_qubits = _collect_ancilla_qubits(circuit)
    data_qubits = sorted(all_qubits - ancilla_qubits)

    new_circuit = stim.Circuit()
    after_repeat = False
    final_noise_inserted = False

    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            after_repeat = True
            new_body = _transform_repeat_body_phenomenological(
                inst.body_copy(), data_qubits=data_qubits, p=p
            )
            new_circuit.append(stim.CircuitRepeatBlock(inst.repeat_count, new_body))
            continue

        if after_repeat and not final_noise_inserted:
            if isinstance(inst, stim.CircuitInstruction) and inst.name in _MEAS_NAMES:
                # Insert data-qubit idle error just before final destructive measurement.
                if data_qubits and p > 0:
                    new_circuit.append("X_ERROR", data_qubits, [p])
                final_noise_inserted = True

        new_circuit.append(inst)

    return new_circuit
