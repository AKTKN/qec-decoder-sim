from __future__ import annotations

import stim
import numpy as np
import scipy.sparse as spsp
from typing import Tuple, Dict, List, FrozenSet

# Default mapping from the 4th detector coordinate to Pauli basis,
# following the chromobius convention used in surface_code and superdense_color_code circuits.
# color_offset (r=0, g=1, b=2) + basis_offset (X=0, Z=3) -> basis
CHROMOBIUS_COORD_BASIS_MAP: dict[int, str] = {
    0: "X", 1: "X", 2: "X",
    3: "Z", 4: "Z", 5: "Z",
}


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


# === not robust === use filter_dem_by_basis instead
# def filter_detectors_by_basis(
#     circuit: stim.Circuit,
#     basis: str,
#     qubits: list[int] | None = None,
# ) -> stim.Circuit | tuple[stim.Circuit, list[str]]:
#     """Return a new circuit filtering any detectors which do not detect the specified basis for the input qubits.

#     Args:
#         circuit: The original circuit
#         basis: "X" or "Z"
#         qubits: Data qubits to inject test errors on. Should typically be data qubits. Defaults
#             to automatically detected data qubits which may not be robust.

#     returns:
#         The filtered circuit
#     """
#     assert basis in ("X", "Z")

#     pauli_error = "Z" if basis == "X" else "X"

#     circuit = circuit.flattened()

#     noiseless_circuit = circuit.without_noise()
#     sampler = noiseless_circuit.compile_detector_sampler()
#     reference_detectors, reference_observables = sampler.sample(
#         1, separate_observables=True
#     )
#     reference_detectors = reference_detectors[0, :]
#     reference_observables = reference_observables[0, :]
#     num_detectors = len(reference_detectors)

#     detector_is_sensitive = np.full(num_detectors, False, dtype=bool)

#     if qubits is None:
#         from relay_bp.stim.noise import detect_data_qubits
#         to_test = detect_data_qubits(noiseless_circuit)
#     else:
#         to_test = qubits

#     to_test_set = set(to_test)

#     inst_idx = 0
#     while to_test:
#         for qubit in to_test:
#             injected_circuit = stim.Circuit()
#             injected_circuit += noiseless_circuit
#             injected_circuit.insert(
#                 inst_idx,
#                 stim.CircuitInstruction(f"{pauli_error}_ERROR", [qubit], [1.0]),
#             )

#             injected_sampler = injected_circuit.compile_detector_sampler()
#             injected_detectors, injected_observables = injected_sampler.sample(
#                 1, separate_observables=True
#             )
#             injected_detectors = injected_detectors[0, :]
#             injected_observables = injected_observables[0, :]

#             detectors_flipped = np.where(reference_detectors != injected_detectors)
#             detector_is_sensitive[detectors_flipped] = True

#         to_test = []
#         for inst in noiseless_circuit[inst_idx:]:
#             # Is a reset we must inject errors after
#             inst_idx += 1
#             if inst.name.startswith("R") or inst.name.startswith("M"):
#                 to_test = list(to_test_set)
#                 break

#     filtered_circuit = stim.Circuit()
#     detector_idx = 0
#     for inst in circuit:
#         if inst.name == "DETECTOR":
#             to_insert = detector_is_sensitive[detector_idx]
#             detector_idx += 1
#             if not to_insert:
#                 continue
#         filtered_circuit.append(inst)
#     return filtered_circuit


def filter_circuit_by_basis(
    circuit: stim.Circuit,
    remove_basis: str,
    coord_basis_map: dict[int, str] | None = None,
) -> stim.Circuit:
    """Return a new circuit with DETECTOR instructions of the specified basis removed.

    Detector basis is determined by the 4th coordinate of each DETECTOR instruction,
    mapped through coord_basis_map (defaults to CHROMOBIUS_COORD_BASIS_MAP).
    Uses circuit.get_detector_coordinates() which resolves SHIFT_COORDS offsets, so
    coordinates are always absolute.

    Unlike filter_dem_by_basis (which operates on a DetectorErrorModel), this function
    operates on the raw Stim circuit.  Removing detectors here ensures that sinter's
    detection-event sampler never sees the removed detectors — necessary when xyz_decoding
    is False and BP decoders should only observe one Pauli basis.

    Args:
        circuit: The source Stim circuit.  Must contain DETECTOR instructions with at
            least 4 coordinate arguments for basis identification.
        remove_basis: ``"X"`` or ``"Z"`` — detectors whose 4th coordinate maps to this
            basis are removed from the returned circuit.
        coord_basis_map: Maps each 4th-coordinate integer value to ``"X"`` or ``"Z"``.
            Defaults to ``CHROMOBIUS_COORD_BASIS_MAP``.  Detectors whose 4th coordinate
            is absent from the map are kept.

    Returns:
        A new ``stim.Circuit`` (flattened) with the identified DETECTOR instructions
        removed.  All other instructions are preserved unchanged.
    """
    assert remove_basis in ("X", "Z"), f"remove_basis must be 'X' or 'Z', got {remove_basis!r}"
    if coord_basis_map is None:
        coord_basis_map = CHROMOBIUS_COORD_BASIS_MAP

    # get_detector_coordinates() returns absolute coords (SHIFT_COORDS already applied)
    det_coords: dict[int, tuple[float, ...]] = circuit.get_detector_coordinates()

    detectors_to_remove: set[int] = {
        idx
        for idx, coords in det_coords.items()
        if len(coords) >= 4 and coord_basis_map.get(int(coords[3])) == remove_basis
    }

    new_circuit = stim.Circuit()
    detector_idx = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if detector_idx not in detectors_to_remove:
                new_circuit.append(instruction)
            detector_idx += 1
        else:
            new_circuit.append(instruction)
    return new_circuit


def filter_dem_by_basis(
    dem: stim.DetectorErrorModel,
    remove_basis: str,
    coord_basis_map: dict[int, str] | None = None,
) -> stim.DetectorErrorModel:
    """Return a new DEM with all detectors of the specified basis removed.

    Detectors are identified by their 4th coordinate (index 3), which is mapped to
    "X" or "Z" via coord_basis_map.  Error mechanisms that reference removed detectors
    have those detector IDs dropped from each component.  An error instruction is
    omitted entirely only when its first component (the actual syndrome, before any
    decomposition separator "^") would become completely empty.

    Remaining detector IDs are remapped to a contiguous range starting from 0.

    Args:
        dem: Source DetectorErrorModel, typically obtained from
            ``circuit.detector_error_model()``.  Must include detector coordinate
            annotations (i.e., the originating circuit used DETECTOR instructions
            with coordinate arguments).
        remove_basis: ``"X"`` or ``"Z"`` — detectors whose 4th coordinate maps to
            this basis are removed.
        coord_basis_map: Maps each possible 4th-coordinate integer value to ``"X"``
            or ``"Z"``.  Defaults to ``CHROMOBIUS_COORD_BASIS_MAP`` which encodes
            the convention used by surface_code and superdense_color_code circuits:
            coordinates 0–2 → X basis, 3–5 → Z basis.
            Detectors whose 4th coordinate is absent from the map are kept.

    Returns:
        A new ``stim.DetectorErrorModel`` with:
        - Detector instructions for the removed basis omitted.
        - All remaining detector IDs remapped to [0, n_kept).
        - Error instructions adjusted to reference only kept detector IDs.
    """
    assert remove_basis in ("X", "Z"), f"remove_basis must be 'X' or 'Z', got {remove_basis!r}"
    if coord_basis_map is None:
        coord_basis_map = CHROMOBIUS_COORD_BASIS_MAP

    # --- Step 1: collect per-detector coordinates from "detector" instructions ---
    detector_coords: dict[int, list[float]] = {}
    for instruction in dem.flattened():
        if instruction.type == "detector":
            targets = instruction.targets_copy()
            if targets:
                detector_coords[targets[0].val] = list(instruction.args_copy())

    # --- Step 2: identify detector IDs to remove ---
    detectors_to_remove: set[int] = set()
    for det_id, coords in detector_coords.items():
        if len(coords) >= 4:
            basis_coord = int(coords[3])
            if coord_basis_map.get(basis_coord) == remove_basis:
                detectors_to_remove.add(det_id)

    # --- Step 3: build contiguous ID remapping for kept detectors ---
    remaining = [i for i in range(dem.num_detectors) if i not in detectors_to_remove]
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(remaining)}

    # --- Step 4: rebuild the DEM ---
    new_dem = stim.DetectorErrorModel()

    for instruction in dem.flattened():
        itype = instruction.type

        if itype == "detector":
            targets = instruction.targets_copy()
            if not targets:
                continue
            det_id = targets[0].val
            if det_id in detectors_to_remove:
                continue
            new_dem.append(
                "detector",
                instruction.args_copy(),
                [stim.target_relative_detector_id(old_to_new[det_id])],
            )

        elif itype == "error":
            p = instruction.args_copy()[0]

            # Split targets into components at "^" separators.
            # Component 0 is the actual (possibly hyperedge) error;
            # subsequent components are its edge decomposition.
            components: list[list[stim.DemTarget]] = [[]]
            for t in instruction.targets_copy():
                if t.is_separator():
                    components.append([])
                else:
                    components[-1].append(t)

            # Filter each component: drop removed detector refs, remap the rest.
            new_components: list[list[stim.DemTarget]] = []
            for comp in components:
                new_comp: list[stim.DemTarget] = []
                for t in comp:
                    if t.is_relative_detector_id():
                        if t.val in detectors_to_remove:
                            continue
                        new_comp.append(stim.target_relative_detector_id(old_to_new[t.val]))
                    else:
                        # logical observable target — always keep
                        new_comp.append(t)
                new_components.append(new_comp)

            # Drop the entire error if the main component is now empty.
            if not new_components[0]:
                continue

            # Reconstruct targets, re-inserting "^" separators.
            new_targets: list[stim.DemTarget] = []
            for i, comp in enumerate(new_components):
                if i > 0:
                    new_targets.append(stim.target_separator())
                new_targets.extend(comp)

            new_dem.append("error", float(p), new_targets)

        elif itype == "logical_observable":
            new_dem.append(
                "logical_observable",
                instruction.args_copy(),
                instruction.targets_copy(),
            )

    return new_dem