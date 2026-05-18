from __future__ import annotations

import stim
import numpy as np
import scipy.sparse as spsp
from typing import Tuple, Dict, List, FrozenSet


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    """Calculate the XOR sum of sets of detectors to identify the net syndrome."""
    out = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)

def parse_targets(targets: List[stim.DemTarget]) -> Tuple[List[List[int]], List[List[int]]]:
    """Parse stim.DemTargets into lists of detector components and observable components."""
    dets: List[List[int]] = [[]]
    frames: List[List[int]] = [[]]
    for t in targets:
        if t.is_relative_detector_id():
            dets[-1].append(t.val)
        elif t.is_logical_observable_id():
            frames[-1].append(t.val)
        elif t.is_separator():
            dets.append([])
            frames.append([])
    return dets, frames


def detector_error_model_to_check_matrices(
    model: stim.Circuit | stim.DetectorErrorModel,
    *,
    decompose_errors: bool = False,
    allow_undecomposed_hyperedges: bool = True,
    **converter_kwargs,
):
    try:
        from beliefmatching import detector_error_model_to_check_matrices as converter
    except ImportError as exc:
        raise ImportError(
            "beliefmatching is required for DEM matrix extraction. "
            "If import fails with an error like 'partially initialized module beliefmatching', "
            "it is often due to a circular import between beliefmatching and ldpc (e.g. ldpc.sinter_decoders). "
            "Try removing any local ldpc checkouts from PYTHONPATH/sys.path and using a released ldpc package, "
            "then reinstall beliefmatching."
        ) from exc

    if isinstance(model, stim.Circuit):
        model = model.detector_error_model(decompose_errors=decompose_errors)
    return converter(model, allow_undecomposed_hyperedges=allow_undecomposed_hyperedges, **converter_kwargs)


def filter_detectors_by_basis(
    circuit: stim.Circuit,
    basis: str,
    qubits: list[int] | None = None,
) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
    """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.

    Args:
        circuit: The original circuit
        basis: "X" or "Z"
        qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
            to automatically detected data qubits which may not be robust.

    returns:
        The filtered circuit
    """
    assert basis in ("X", "Z")

    pauli_error = "Z" if basis == "X" else "X"

    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, reference_observables = sampler.sample(
        1, separate_observables=True
    )
    reference_detectors = reference_detectors[0, :]
    reference_observables = reference_observables[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

    if qubits is None:
        from relay_bp.stim.noise import detect_data_qubits
        to_test = detect_data_qubits(noiseless_circuit)
    else:
        to_test = qubits

    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
            )

            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, injected_observables = injected_sampler.sample(
                1, separate_observables=True
            )
            injected_detectors = injected_detectors[0, :]
            injected_observables = injected_observables[0, :]

            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            # Is a reset we must inject errors after
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            to_insert = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not to_insert:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit