from __future__ import annotations

import stim

from qec_decoder_sim.noise.noise_model import (
    ABSTRACT_1Q_GATE_DECOMPOSITIONS,
    decompose_abstract_1q_gates,
    insert_arbitrary_noise,
    si1000_circuit_level,
)


def _tableau_for(gates: tuple[str, ...]) -> stim.Tableau:
    result = stim.Tableau(1)
    for gate in gates:
        result.append(stim.Tableau.from_named_gate(gate), [0])
    return result


def test_abstract_1q_gate_decomposition_table_is_equivalent():
    for abstract_gate, physical_gates in ABSTRACT_1Q_GATE_DECOMPOSITIONS.items():
        assert _tableau_for((abstract_gate,)) == _tableau_for(physical_gates)


def test_decompose_abstract_1q_gates_keeps_h_and_expands_cyclic_cliffords():
    circuit = stim.Circuit(
        """
        C_XYZ 0 1
        C_ZYX 2
        H 3
        TICK
        REPEAT 2 {
            SQRT_Y 4
            TICK
        }
        """
    )

    decomposed = decompose_abstract_1q_gates(circuit)
    text = str(decomposed)

    assert "C_XYZ" not in text
    assert "C_ZYX" not in text
    assert "SQRT_Y" not in text
    assert "H 3" in text
    assert "SQRT_X 0 1" in text
    assert "S 0 1" in text
    assert "S_DAG 2" in text
    assert "SQRT_X_DAG 2" in text


def test_noise_insertion_decomposes_abstract_gates_before_applying_1q_noise():
    noisy = insert_arbitrary_noise(
        stim.Circuit(
            """
            C_XYZ 0
            H 1
            TICK
            """
        ),
        si1000_circuit_level(0.01),
    )
    text = str(noisy)

    assert "C_XYZ" not in text
    assert "SQRT_X 0" in text
    assert "S 0" in text
    assert "H 1" in text
    assert text.count("DEPOLARIZE1(0.001)") >= 3
