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