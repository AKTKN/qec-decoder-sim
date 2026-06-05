from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim
import scipy.sparse as spsp
import networkx as nx

from qec_decoder_sim.codes.code_factory import BBCodeSpec, build_bb_block_code


@dataclass(frozen=True)
class BBCircuitMetadata:
    data_qubits: tuple[int, ...]
    x_ancilla_qubits: tuple[int, ...]
    z_ancilla_qubits: tuple[int, ...]
    num_rounds: int
    basis: Literal["x", "z"]


@dataclass(frozen=True)
class BBCircuitBuildResult:
    circuit: stim.Circuit
    metadata: BBCircuitMetadata


def compute_check_colors(H: np.array | spsp.csc_matrix) -> list[int]:
    """
    Generate a co-occurrence graph from the parity check matrix H and determine the color of each check node (row).
    Rows (checks) that have non-zero elements in the same column (data qubit) will have different colors.
    """
    if not isinstance(H, spsp.csc_matrix):
        H = spsp.csc_matrix(H)
    
    num_checks = H.shape[0]
    num_data = H.shape[1]

    # Build the graph
    G = nx.Graph()
    G.add_nodes_from(range(num_checks))  

    # For each columns, add edges between connected rows
    for j in range(num_data):
        connected_checks = H.indices[H.indptr[j]:H.indptr[j+1]] 
        for i in range(len(connected_checks)):
            for k in range(i + 1, len(connected_checks)):
                G.add_edge(connected_checks[i], connected_checks[k])

    # Graph coloring (Greedy coloring based on Welsh-Powell algorithm, etc.)
    # DSATUR (saturation_largest_first) generally yields the best results (minimum number of colors)
    color_dict = nx.coloring.greedy_color(G, strategy='saturation_largest_first')
    
    # Store colors in a list ordered by node index
    colors = [color_dict[i] for i in range(num_checks)]
    return colors


def _nnz_positions(matrix) -> np.ndarray:
    rows, cols = matrix.nonzero()
    return cols[np.argsort(rows)]


def _append_detectors_for_round(
    circuit: stim.Circuit,
    *,
    n: int,
    z_basis: bool,
    use_both: bool,
    repeat: bool,
    round_index: int,
    colors: list[int] | None,
    check_offset: int,
) -> None:
    if colors is None:
        colors = [0] * (n // 2) # Default to all checks having the same color if no coloring is provided

    if z_basis:
        if repeat:
            for i in range(n // 2):
                coords = [check_offset + i, round_index, colors[i]]
                circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i), stim.target_rec(-n - n // 2 + i)], coords)
        else:
            for i in range(n // 2):
                coords = [check_offset + i, round_index, colors[i]]
                circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i)], coords)
    elif use_both and repeat:
        for i in range(n // 2):
            coords = [check_offset + i, round_index, colors[i]]
            circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i), stim.target_rec(-n - n // 2 + i)], coords)


def _append_detectors_for_round_x(
    circuit: stim.Circuit,
    *,
    n: int,
    z_basis: bool,
    use_both: bool,
    repeat: bool,
    round_index: int,
    colors: list[int] | None,
    check_offset: int,
) -> None:
    if colors is None:
        colors = [0] * (n // 2)
    if not z_basis:
        if repeat:
            for i in range(n // 2):
                coords = [check_offset + i, round_index, colors[i]]
                circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i), stim.target_rec(-n - n // 2 + i)], coords)
        else:
            for i in range(n // 2):
                coords = [check_offset + i, round_index, colors[i]]
                circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i)], coords)
    elif use_both and repeat:
        for i in range(n // 2):
            coords = [check_offset + i, round_index, colors[i]]
            circuit.append("DETECTOR", [stim.target_rec(-n // 2 + i), stim.target_rec(-n - n // 2 + i)], coords)


def _append_measure(circuit: stim.Circuit, gate: str, qubit: int) -> None:
    circuit.append(gate, [qubit])


def _append_round(
    circuit: stim.Circuit,
    *,
    n: int,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    A1_T: np.ndarray,
    A2_T: np.ndarray,
    A3_T: np.ndarray,
    B1_T: np.ndarray,
    B2_T: np.ndarray,
    B3_T: np.ndarray,
    x_check_offset: int,
    l_data_offset: int,
    r_data_offset: int,
    z_check_offset: int,
    z_basis: bool,
    use_both: bool,
    hzh: bool,
    repeat: bool,
    round_index: int,
    colors_z: list[int],
    colors_x: list[int],    
) -> None:
    if not repeat:
        for i in range(n // 2):
            circuit.append("H", [x_check_offset + i])
    elif hzh:
        for i in range(n // 2):
            circuit.append("H", [x_check_offset + i])

    for i in range(n // 2):
        circuit.append("CNOT", [r_data_offset + int(A1_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A2[i])])
        circuit.append("CNOT", [r_data_offset + int(A3_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B2[i])])
        circuit.append("CNOT", [l_data_offset + int(B1_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B1[i])])
        circuit.append("CNOT", [l_data_offset + int(B2_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B3[i])])
        circuit.append("CNOT", [l_data_offset + int(B3_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A1[i])])
        circuit.append("CNOT", [r_data_offset + int(A2_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A3[i])])
        _append_measure(circuit, "MR", z_check_offset + i)

    _append_detectors_for_round(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_z, check_offset=z_check_offset)
    circuit.append("TICK")

    for i in range(n // 2):
        if hzh:
            circuit.append("H", [x_check_offset + i])
            _append_measure(circuit, "MR", x_check_offset + i)
        else:
            _append_measure(circuit, "MRX", x_check_offset + i)

    _append_detectors_for_round_x(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_x, check_offset=x_check_offset)
    circuit.append("TICK")


def _append_round_ibm(
    circuit: stim.Circuit,
    *,
    n: int,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    A1_T: np.ndarray,
    A2_T: np.ndarray,
    A3_T: np.ndarray,
    B1_T: np.ndarray,
    B2_T: np.ndarray,
    B3_T: np.ndarray,
    x_check_offset: int,
    l_data_offset: int,
    r_data_offset: int,
    z_check_offset: int,
    z_basis: bool,
    use_both: bool,
    hzh: bool,
    repeat: bool,
    round_index: int,
    colors_z: list[int],
    colors_x: list[int],
    p: float,
) -> None:
    # preparation errors and H gates at start of round (before first TICK of this round)
    if not repeat:
        for i in range(n // 2):
            circuit.append("H", [x_check_offset + i])
            if hzh:
                circuit.append("DEPOLARIZE1", [x_check_offset + i], [p])
    else:
        for i in range(n // 2):
            circuit.append("X_ERROR", [z_check_offset + i], [p])
            if hzh:
                circuit.append("X_ERROR", [x_check_offset + i], [p])
                circuit.append("H", [x_check_offset + i])
                circuit.append("DEPOLARIZE1", [x_check_offset + i], [p])
            else:
                circuit.append("Z_ERROR", [x_check_offset + i], [p])
            circuit.append("DEPOLARIZE1", [r_data_offset + i], [p])

    for i in range(n // 2):
        circuit.append("CNOT", [r_data_offset + int(A1_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [r_data_offset + int(A1_T[i]), z_check_offset + i], [p])
        circuit.append("DEPOLARIZE1", [l_data_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A2[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, l_data_offset + int(A2[i])], [p])
        circuit.append("CNOT", [r_data_offset + int(A3_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [r_data_offset + int(A3_T[i]), z_check_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B2[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, r_data_offset + int(B2[i])], [p])
        circuit.append("CNOT", [l_data_offset + int(B1_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [l_data_offset + int(B1_T[i]), z_check_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B1[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, r_data_offset + int(B1[i])], [p])
        circuit.append("CNOT", [l_data_offset + int(B2_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [l_data_offset + int(B2_T[i]), z_check_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B3[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, r_data_offset + int(B3[i])], [p])
        circuit.append("CNOT", [l_data_offset + int(B3_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [l_data_offset + int(B3_T[i]), z_check_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A1[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, l_data_offset + int(A1[i])], [p])
        circuit.append("CNOT", [r_data_offset + int(A2_T[i]), z_check_offset + i])
        circuit.append("DEPOLARIZE2", [r_data_offset + int(A2_T[i]), z_check_offset + i], [p])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A3[i])])
        circuit.append("DEPOLARIZE2", [x_check_offset + i, l_data_offset + int(A3[i])], [p])
        circuit.append("X_ERROR", [z_check_offset + i], [p])
        circuit.append("MR", [z_check_offset + i])

    _append_detectors_for_round(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_z, check_offset=z_check_offset)
    circuit.append("TICK")

    for i in range(n // 2):
        if hzh:
            circuit.append("H", [x_check_offset + i])
            circuit.append("DEPOLARIZE1", [x_check_offset + i], [p])
            circuit.append("X_ERROR", [x_check_offset + i], [p])
            circuit.append("MR", [x_check_offset + i])
        else:
            circuit.append("Z_ERROR", [x_check_offset + i], [p])
            circuit.append("MRX", [x_check_offset + i])

    _append_detectors_for_round_x(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_x, check_offset=x_check_offset)
    circuit.append("TICK")


def _append_round_phenom(
    circuit: stim.Circuit,
    *,
    n: int,
    A1: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    A1_T: np.ndarray,
    A2_T: np.ndarray,
    A3_T: np.ndarray,
    B1_T: np.ndarray,
    B2_T: np.ndarray,
    B3_T: np.ndarray,
    x_check_offset: int,
    l_data_offset: int,
    r_data_offset: int,
    z_check_offset: int,
    z_basis: bool,
    use_both: bool,
    hzh: bool,
    repeat: bool,
    round_index: int,
    colors_z: list[int],
    colors_x: list[int],
    data_qubits: tuple[int, ...],
    p: float,
) -> None:
    """One syndrome extraction round with phenomenological noise.

    X_ERROR(p) is applied to all data qubits at the start of the round.
    X_ERROR(p) / Z_ERROR(p) is inserted immediately before each ancilla
    measurement to model imperfect syndrome readout.
    """
    circuit.append("X_ERROR", list(data_qubits), [p])

    if not repeat:
        for i in range(n // 2):
            circuit.append("H", [x_check_offset + i])
    elif hzh:
        for i in range(n // 2):
            circuit.append("H", [x_check_offset + i])

    for i in range(n // 2):
        circuit.append("CNOT", [r_data_offset + int(A1_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A2[i])])
        circuit.append("CNOT", [r_data_offset + int(A3_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B2[i])])
        circuit.append("CNOT", [l_data_offset + int(B1_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B1[i])])
        circuit.append("CNOT", [l_data_offset + int(B2_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, r_data_offset + int(B3[i])])
        circuit.append("CNOT", [l_data_offset + int(B3_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A1[i])])
        circuit.append("CNOT", [r_data_offset + int(A2_T[i]), z_check_offset + i])
    circuit.append("TICK")

    for i in range(n // 2):
        circuit.append("CNOT", [x_check_offset + i, l_data_offset + int(A3[i])])
        circuit.append("X_ERROR", [z_check_offset + i], [p])
        circuit.append("MR", [z_check_offset + i])

    _append_detectors_for_round(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_z, check_offset=z_check_offset)
    circuit.append("TICK")

    for i in range(n // 2):
        if hzh:
            circuit.append("H", [x_check_offset + i])
            circuit.append("X_ERROR", [x_check_offset + i], [p])
            circuit.append("MR", [x_check_offset + i])
        else:
            circuit.append("Z_ERROR", [x_check_offset + i], [p])
            circuit.append("MRX", [x_check_offset + i])

    _append_detectors_for_round_x(circuit, n=n, z_basis=z_basis, use_both=use_both, repeat=repeat, round_index=round_index, colors=colors_x, check_offset=x_check_offset)
    circuit.append("TICK")


def build_bb_phenom_stim_circuit(
    spec: BBCodeSpec,
    *,
    num_rounds: int,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
    p: float,
) -> BBCircuitBuildResult:
    """Build a BB circuit under the phenomenological noise model.

    Each syndrome extraction round receives:
    - X_ERROR(p) on all data qubits (idle error during the round)
    - X_ERROR(p) before Z-ancilla measurements (MR)
    - Z_ERROR(p) before X-ancilla measurements (MRX), or X_ERROR(p) after H
      when hzh=True

    Gate operations themselves are ideal (no two-qubit gate noise).
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    block_code = build_bb_block_code(spec)
    colors_z = compute_check_colors(block_code.code.hz)
    colors_x = compute_check_colors(block_code.code.hx)

    if len(block_code.a_blocks) != 3 or len(block_code.b_blocks) != 3:
        raise ValueError("Current BB Stim schedule requires exactly three A blocks and three B blocks.")

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

    for i in range(n):
        circuit.append("R" if z_basis else "RX", [l_data_offset + i])

    data_qubits = tuple(range(l_data_offset, l_data_offset + n))

    circuit.append("TICK")

    round_args = dict(
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
        data_qubits=data_qubits,
        p=p,
    )

    for round_index in range(num_rounds):
        _append_round_phenom(circuit, **round_args, repeat=round_index > 0, round_index=round_index)

    circuit.append("X_ERROR", list(data_qubits), [p])
    for i in range(n):
        circuit.append("M" if z_basis else "MX", [l_data_offset + i])

    pcm = block_code.code.hz if z_basis else block_code.code.hx
    logical_pcm = block_code.code.logicals.lz if z_basis else block_code.code.logicals.lx

    colors = colors_z if z_basis else colors_x
    check_offset = z_check_offset if z_basis else x_check_offset

    for i, stabilizer in enumerate(pcm):
        nnz = np.nonzero(stabilizer)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        targets.append(stim.target_rec(-n - n + i) if z_basis else stim.target_rec(-n - n // 2 + i))
        circuit.append("DETECTOR", targets, [check_offset + i, num_rounds, colors[i]])

    for i, logical in enumerate(logical_pcm):
        nnz = np.nonzero(logical)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        circuit.append("OBSERVABLE_INCLUDE", targets, [i])

    metadata = BBCircuitMetadata(
        data_qubits=data_qubits,
        x_ancilla_qubits=tuple(range(x_check_offset, x_check_offset + n // 2)),
        z_ancilla_qubits=tuple(range(z_check_offset, z_check_offset + n // 2)),
        num_rounds=num_rounds,
        basis="z" if z_basis else "x",
    )
    return BBCircuitBuildResult(circuit=circuit, metadata=metadata)


def build_bb_ibm_noisy_stim_circuit(
    spec: BBCodeSpec,
    *,
    num_rounds: int,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
    p: float,
) -> BBCircuitBuildResult:
    """Build a noisy BB circuit using IBM's inline noise insertion.

    Matches the noise model in SlidingWindowDecoder/src/build_circuit.py:
    DEPOLARIZE2 after each CNOT, X_ERROR/Z_ERROR for reset and measurement
    errors, and DEPOLARIZE1 for idle data qubits — all inserted inline rather
    than post-processed. Requires uniform noise (all channels use the same p).
    """
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    block_code = build_bb_block_code(spec)
    colors_z = compute_check_colors(block_code.code.hz)
    colors_x = compute_check_colors(block_code.code.hx)

    if len(block_code.a_blocks) != 3 or len(block_code.b_blocks) != 3:
        raise ValueError("Current BB Stim schedule requires exactly three A blocks and three B blocks.")

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

    round_args = dict(
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

    # --- Original implementation (explicit unroll) ---
    # Kept here for reference; see comment below for why this was replaced.
    #
    # for round_index in range(num_rounds):
    #     _append_round_ibm(circuit, **round_args, repeat=round_index > 0, round_index=round_index)

    # --- Current implementation (stim REPEAT block) ---
    # Using a REPEAT block instead of explicit unrolling so that stim keeps
    # independent error mechanisms from different gates separate in the DEM.
    # With explicit unrolling, stim merges error mechanisms that share the same
    # detector pattern, producing a coarser factor-graph (~1584 fewer variable
    # nodes for d=12) which degrades BP decoding performance.
    # SHIFT_COORDS(0, 1, 0) at the end of the body increments the round
    # coordinate so DETECTOR visualisation remains correct across iterations.
    _append_round_ibm(circuit, **round_args, repeat=False, round_index=0)
    if num_rounds > 1:
        rep_body = stim.Circuit()
        _append_round_ibm(rep_body, **round_args, repeat=True, round_index=0)
        rep_body.append("SHIFT_COORDS", [], [0.0, 1.0, 0.0])
        circuit += (num_rounds - 1) * rep_body

    for i in range(n):
        circuit.append("M" if z_basis else "MX", [l_data_offset + i])

    pcm = block_code.code.hz if z_basis else block_code.code.hx
    logical_pcm = block_code.code.logicals.lz if z_basis else block_code.code.logicals.lx

    colors = colors_z if z_basis else colors_x
    check_offset = z_check_offset if z_basis else x_check_offset

    for i, stabilizer in enumerate(pcm):
        nnz = np.nonzero(stabilizer)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        targets.append(stim.target_rec(-n - n + i) if z_basis else stim.target_rec(-n - n // 2 + i))
        circuit.append("DETECTOR", targets, [check_offset + i, num_rounds, colors[i]])

    for i, logical in enumerate(logical_pcm):
        nnz = np.nonzero(logical)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        circuit.append("OBSERVABLE_INCLUDE", targets, [i])

    metadata = BBCircuitMetadata(
        data_qubits=data_qubits,
        x_ancilla_qubits=tuple(range(x_check_offset, x_check_offset + n // 2)),
        z_ancilla_qubits=tuple(range(z_check_offset, z_check_offset + n // 2)),
        num_rounds=num_rounds,
        basis="z" if z_basis else "x",
    )
    return BBCircuitBuildResult(circuit=circuit, metadata=metadata)


def build_bb_ideal_stim_circuit(
    spec: BBCodeSpec,
    *,
    num_rounds: int,
    basis: Literal["x", "z"],
    use_both: bool,
    hzh: bool,
) -> BBCircuitBuildResult:
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")

    block_code = build_bb_block_code(spec)
    colors_z = compute_check_colors(block_code.code.hz)
    colors_x = compute_check_colors(block_code.code.hx)

    if len(block_code.a_blocks) != 3 or len(block_code.b_blocks) != 3:
        raise ValueError("Current BB Stim schedule requires exactly three A blocks and three B blocks.")

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

    for i in range(n):
        circuit.append("R" if z_basis else "RX", [l_data_offset + i])

    data_qubits = tuple(range(l_data_offset, l_data_offset + n))

    circuit.append("TICK")

    for round_index in range(num_rounds):
        _append_round(
            circuit,
            n=n,
            A1=A1,
            A2=A2,
            A3=A3,
            B1=B1,
            B2=B2,
            B3=B3,
            A1_T=A1_T,
            A2_T=A2_T,
            A3_T=A3_T,
            B1_T=B1_T,
            B2_T=B2_T,
            B3_T=B3_T,
            x_check_offset=x_check_offset,
            l_data_offset=l_data_offset,
            r_data_offset=r_data_offset,
            z_check_offset=z_check_offset,
            z_basis=z_basis,
            use_both=use_both,
            hzh=hzh,
            repeat=round_index > 0,
            round_index=round_index,
            colors_z=colors_z,
            colors_x=colors_x,
        )

    for i in range(n):
        circuit.append("M" if z_basis else "MX", [l_data_offset + i])

    pcm = block_code.code.hz if z_basis else block_code.code.hx
    logical_pcm = block_code.code.logicals.lz if z_basis else block_code.code.logicals.lx

    colors = colors_z if z_basis else colors_x
    check_offset = z_check_offset if z_basis else x_check_offset

    for i, stabilizer in enumerate(pcm):
        nnz = np.nonzero(stabilizer)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        targets.append(stim.target_rec(-n - n + i) if z_basis else stim.target_rec(-n - n // 2 + i))
        circuit.append("DETECTOR", targets, [check_offset + i, num_rounds, colors[i]])

    for i, logical in enumerate(logical_pcm):
        nnz = np.nonzero(logical)[0]
        targets = [stim.target_rec(-n + int(index)) for index in nnz]
        circuit.append("OBSERVABLE_INCLUDE", targets, [i])

    metadata = BBCircuitMetadata(
        data_qubits=data_qubits,
        x_ancilla_qubits=tuple(range(x_check_offset, x_check_offset + n // 2)),
        z_ancilla_qubits=tuple(range(z_check_offset, z_check_offset + n // 2)),
        num_rounds=num_rounds,
        basis="z" if z_basis else "x",
    )
    return BBCircuitBuildResult(circuit=circuit, metadata=metadata)