from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import stim


@dataclass(frozen=True)
class NoiseParameters:
    idle_dpl: float = 0.0
    one_q_gate_dpl: float = 0.0
    two_q_gate_dpl: float = 0.0
    measurement_flip: float = 0.0
    reset_flip: float = 0.0
    reasonable_idle_dpl: float = 0.0
    # Relative standard deviation used by the realistic noise inserter.
    # For a channel with mean probability mu, we sample from N(mu, (sigma_rel*mu)^2)
    # and then clip into [0, 1].
    sigma_rel: float = 0.0


def uniform_circuit_level(p: float) -> NoiseParameters:
    return NoiseParameters(
        idle_dpl=p,
        one_q_gate_dpl=p,
        two_q_gate_dpl=p,
        measurement_flip=p,
        reset_flip=p,
        reasonable_idle_dpl=p,
    )


def si1000_circuit_level(p: float) -> NoiseParameters:
    return NoiseParameters(
        idle_dpl=p / 10,
        one_q_gate_dpl=p / 10,
        two_q_gate_dpl=p,
        measurement_flip=5 * p,
        reset_flip=2 * p,
        reasonable_idle_dpl=2 * p,
    )


ABSTRACT_1Q_GATE_DECOMPOSITIONS: dict[str, tuple[str, ...]] = {
    "C_XYZ": ("SQRT_X", "S"),
    "C_ZYX": ("S_DAG", "SQRT_X_DAG"),
    "H_XY": ("S_DAG", "SQRT_X", "SQRT_X"),
    "H_YZ": ("S", "S", "SQRT_X_DAG"),
    "SQRT_Y": ("S", "SQRT_X_DAG", "S_DAG"),
    "SQRT_Y_DAG": ("S", "SQRT_X", "S_DAG"),
}


def decompose_abstract_1q_gates(circuit: stim.Circuit) -> stim.Circuit:
    """Expand abstract one-qubit Clifford gates before noise insertion.

    The circuit generators and Chromobius-derived optimizers may use compact
    Clifford names such as C_XYZ as internal representatives of a rotation.
    These names are not the intended physical gate basis for circuit-level
    noise. Expand them after all circuit optimizations have run, but before
    adding noise, so each physical gate receives its own one-qubit noise event.

    H is intentionally not decomposed.
    """
    out = stim.Circuit()
    moment: list[stim.CircuitInstruction] = []
    annotation_names = {
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "MPAD",
    }

    def flush_moment() -> None:
        nonlocal moment, out
        if not moment:
            return
        layer_count = max(
            len(ABSTRACT_1Q_GATE_DECOMPOSITIONS.get(inst.name, (inst.name,)))
            for inst in moment
        )
        layers = [stim.Circuit() for _ in range(layer_count)]
        for inst in moment:
            replacement = ABSTRACT_1Q_GATE_DECOMPOSITIONS.get(inst.name)
            if replacement is None:
                layers[0].append(inst)
                continue
            args = inst.gate_args_copy()
            if args:
                raise ValueError(f"Cannot decompose {inst.name} with gate arguments: {args}")
            targets = inst.targets_copy()
            for k, gate in enumerate(replacement):
                layers[k].append(gate, targets)

        for k, layer in enumerate(layers):
            out += layer
            if k + 1 < len(layers):
                out.append("TICK")
        moment = []

    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            flush_moment()
            out.append(
                stim.CircuitRepeatBlock(
                    inst.repeat_count,
                    decompose_abstract_1q_gates(inst.body_copy()),
                )
            )
            continue

        if inst.name == "TICK":
            flush_moment()
            out.append(inst)
            continue
        if inst.name in annotation_names:
            flush_moment()
            out.append(inst)
            continue
        moment.append(inst)
    flush_moment()
    return out


class CircuitNoiseInserter:
    """Injects noise into a stim circuit by walking its instruction stream.

    This file originally provided a set of standalone functions. We now implement
    the logic inside a class to improve maintainability and provide an extension
    point for more realistic, per-target non-uniform noise.

    Important compatibility note:
        The existing top-level function `insert_arbitrary_noise(...)` remains
        available and keeps identical semantics. It simply forwards to
        `CircuitNoiseInserter(...).insert_arbitrary_noise(...)`.
    """

    def __init__(
        self,
        noise: NoiseParameters,
        *,
        data_qubits: set[int] | None = None,
        perfect_logical_initialization: bool = False,
        perfect_logical_measurement: bool = False,
        perfect_first_syndrome_extraction: bool = False,
        syndrome_ticks_per_round: int = 8,
    ) -> None:
        self.noise = noise
        self.data_qubits = set() if data_qubits is None else set(data_qubits)
        self.perfect_logical_initialization = perfect_logical_initialization
        self.perfect_logical_measurement = perfect_logical_measurement
        self.perfect_first_syndrome_extraction = perfect_first_syndrome_extraction
        self.syndrome_ticks_per_round = syndrome_ticks_per_round

    @staticmethod
    def _instruction_qubits(inst: stim.CircuitInstruction) -> list[int]:
        return [target.qubit_value for target in inst.targets_copy() if target.is_qubit_target]

    @staticmethod
    def _append_measurement_flip(
        noisy_circuit: stim.Circuit,
        measurement_name: str,
        qubits: list[int],
        probability: float,
        *,
        skip_qubits: set[int] | None = None,
    ) -> None:
        filtered_qubits = qubits if skip_qubits is None else [qubit for qubit in qubits if qubit not in skip_qubits]
        if probability <= 0 or not filtered_qubits:
            return
        if measurement_name in {"M", "MZ", "MR", "MRZ"}:
            noisy_circuit.append("X_ERROR", filtered_qubits, [probability])
        elif measurement_name in {"MX", "MRX"}:
            noisy_circuit.append("Z_ERROR", filtered_qubits, [probability])
        elif measurement_name in {"MY", "MRY"}:
            noisy_circuit.append("X_ERROR", filtered_qubits, [probability])

    @staticmethod
    def _append_reset_flip(
        noisy_circuit: stim.Circuit,
        reset_name: str,
        qubits: list[int],
        probability: float,
        *,
        skip_qubits: set[int] | None = None,
    ) -> None:
        filtered_qubits = qubits if skip_qubits is None else [qubit for qubit in qubits if qubit not in skip_qubits]
        if probability <= 0 or not filtered_qubits:
            return
        if reset_name in {"R", "RZ", "MR", "MRZ"}:
            noisy_circuit.append("X_ERROR", filtered_qubits, [probability])
        elif reset_name in {"RX", "MRX"}:
            noisy_circuit.append("Z_ERROR", filtered_qubits, [probability])
        elif reset_name in {"RY", "MRY"}:
            noisy_circuit.append("X_ERROR", filtered_qubits, [probability])

    def _step_is_perfect(self, step: int) -> bool:
        if self.perfect_logical_initialization and step == 0:
            return True
        if self.perfect_first_syndrome_extraction and 1 <= step <= self.syndrome_ticks_per_round:
            return True
        return False

    @staticmethod
    def _clip_probability(p: float) -> float:
        # stim expects probabilities in [0, 1]. Clip defensively.
        return float(np.clip(p, 0.0, 1.0))

    def _sample_probability(self, *, rng: np.random.Generator, mean: float) -> float:
        mu = self._clip_probability(mean)
        sigma_rel = float(self.noise.sigma_rel)
        if sigma_rel <= 0 or mu <= 0:
            return mu
        sigma = sigma_rel * mu
        return self._clip_probability(float(rng.normal(loc=mu, scale=sigma)))

    @staticmethod
    def _rng_from_seed_or_rng(*, seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
        if seed is not None and rng is not None:
            raise ValueError("Specify either seed or rng, not both")
        if rng is not None:
            return rng
        return np.random.default_rng(seed)

    @staticmethod
    def _measurement_error_gate(measurement_name: str) -> str | None:
        if measurement_name in {"M", "MZ", "MR", "MRZ"}:
            return "X_ERROR"
        if measurement_name in {"MX", "MRX"}:
            return "Z_ERROR"
        if measurement_name in {"MY", "MRY"}:
            return "X_ERROR"
        return None

    @staticmethod
    def _reset_error_gate(reset_name: str) -> str | None:
        if reset_name in {"R", "RZ", "MR", "MRZ"}:
            return "X_ERROR"
        if reset_name in {"RX", "MRX"}:
            return "Z_ERROR"
        if reset_name in {"RY", "MRY"}:
            return "X_ERROR"
        return None

    def insert_arbitrary_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """Insert the existing uniform-per-channel noise model.

        This keeps the original behavior of the pre-refactor implementation.
        """
        circuit = decompose_abstract_1q_gates(circuit)
        noisy_circuit = stim.Circuit()
        all_qubits = set(range(circuit.num_qubits))

        used_qubits_this_step: set[int] = set()
        measured_or_reset_qubits_this_step: set[int] = set()
        has_measure_or_reset_this_step = False
        step_index = 0

        def flush_step_noise(*, apply_noise: bool) -> None:
            nonlocal used_qubits_this_step, measured_or_reset_qubits_this_step, has_measure_or_reset_this_step

            if apply_noise:
                if has_measure_or_reset_this_step:
                    resonator_idle_targets = sorted(all_qubits - measured_or_reset_qubits_this_step)
                    if resonator_idle_targets and self.noise.reasonable_idle_dpl > 0:
                        noisy_circuit.append("DEPOLARIZE1", resonator_idle_targets, [self.noise.reasonable_idle_dpl])
                else:
                    idle_targets = sorted(all_qubits - used_qubits_this_step)
                    if idle_targets and self.noise.idle_dpl > 0:
                        noisy_circuit.append("DEPOLARIZE1", idle_targets, [self.noise.idle_dpl])

            used_qubits_this_step = set()
            measured_or_reset_qubits_this_step = set()
            has_measure_or_reset_this_step = False

        for inst in circuit:
            if isinstance(inst, stim.CircuitRepeatBlock):
                flush_step_noise(apply_noise=not self._step_is_perfect(step_index))
                nested = CircuitNoiseInserter(
                    self.noise,
                    data_qubits=self.data_qubits,
                    perfect_logical_initialization=False,
                    perfect_logical_measurement=self.perfect_logical_measurement,
                    perfect_first_syndrome_extraction=False,
                    syndrome_ticks_per_round=self.syndrome_ticks_per_round,
                )
                nested_noisy = nested.insert_arbitrary_noise(inst.body_copy())
                noisy_circuit.append(stim.CircuitRepeatBlock(inst.repeat_count, nested_noisy))
                continue

            qubits = self._instruction_qubits(inst)
            name = inst.name
            step_is_perfect = self._step_is_perfect(step_index)

            if name == "TICK":
                flush_step_noise(apply_noise=not step_is_perfect)
                noisy_circuit.append(inst)
                step_index += 1
                continue

            if qubits:
                used_qubits_this_step.update(qubits)

            if name in {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}:
                skip_measurement_qubits: set[int] = set()
                if self.perfect_logical_measurement and name in {"M", "MZ", "MX", "MY"}:
                    skip_measurement_qubits = self.data_qubits
                if not step_is_perfect:
                    self._append_measurement_flip(
                        noisy_circuit,
                        name,
                        qubits,
                        self.noise.measurement_flip,
                        skip_qubits=skip_measurement_qubits,
                    )
                noisy_circuit.append(inst)
                measured_or_reset_qubits_this_step.update(qubits)
                has_measure_or_reset_this_step = True
                skip_reset_qubits: set[int] = set()
                if self.perfect_logical_initialization and step_index == 0:
                    skip_reset_qubits = self.data_qubits
                if not step_is_perfect:
                    self._append_reset_flip(
                        noisy_circuit,
                        name,
                        qubits,
                        self.noise.reset_flip,
                        skip_qubits=skip_reset_qubits,
                    )
                continue

            noisy_circuit.append(inst)

            if name in {"R", "RZ", "RX", "RY"}:
                measured_or_reset_qubits_this_step.update(qubits)
                has_measure_or_reset_this_step = True
                skip_reset_qubits: set[int] = set()
                if self.perfect_logical_initialization and step_index == 0:
                    skip_reset_qubits = self.data_qubits
                if not step_is_perfect:
                    self._append_reset_flip(
                        noisy_circuit,
                        name,
                        qubits,
                        self.noise.reset_flip,
                        skip_qubits=skip_reset_qubits,
                    )
                continue

            if name in {
                "H",
                "H_XY",
                "H_YZ",
                "S",
                "S_DAG",
                "SQRT_X",
                "SQRT_X_DAG",
                "SQRT_Y",
                "SQRT_Y_DAG",
                "C_XYZ",
                "C_ZYX",
                "X",
                "Y",
                "Z",
            }:
                if not step_is_perfect and self.noise.one_q_gate_dpl > 0 and qubits:
                    noisy_circuit.append("DEPOLARIZE1", qubits, [self.noise.one_q_gate_dpl])
                continue

            if name in {"CNOT", "CX", "CZ", "CY", "ISWAP", "ISWAP_DAG", "SWAP", "CXSWAP", "CZSWAP", "SWAPCX"}:
                if not step_is_perfect and self.noise.two_q_gate_dpl > 0 and qubits:
                    noisy_circuit.append("DEPOLARIZE2", qubits, [self.noise.two_q_gate_dpl])
                continue

        flush_step_noise(apply_noise=not self._step_is_perfect(step_index))
        return noisy_circuit

    def insert_realistic_noise(
        self,
        circuit: stim.Circuit,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        unroll_repeat_blocks: bool = True,
    ) -> stim.Circuit:
        """Insert non-uniform, per-target noise.

        Differences from `insert_arbitrary_noise`:
        - For each noise event we sample an independent probability per target qubit.
        - For 2q gates, probability is sampled per physical gate (i.e. per target pair).
        - When `unroll_repeat_blocks=True`, REPEAT blocks are expanded so each iteration
          receives independently sampled noise (more realistic time variation).
        """
        circuit = decompose_abstract_1q_gates(circuit)
        rng = self._rng_from_seed_or_rng(seed=seed, rng=rng)
        noisy_circuit = stim.Circuit()
        all_qubits = set(range(circuit.num_qubits))

        used_qubits_this_step: set[int] = set()
        measured_or_reset_qubits_this_step: set[int] = set()
        has_measure_or_reset_this_step = False
        step_index = 0

        def _iter_instructions() -> list[stim.CircuitInstruction]:
            if not unroll_repeat_blocks:
                # In realistic mode, we still support repeat blocks, but then
                # noise is sampled once for the body and repeated identically.
                return list(circuit)
            if hasattr(circuit, "flattened"):
                # stim provides circuit.flattened() that expands REPEAT blocks.
                return list(circuit.flattened())

            # Fallback: manual unroll.
            out: list[stim.CircuitInstruction] = []
            for inst in circuit:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    body = inst.body_copy()
                    for _ in range(inst.repeat_count):
                        out.extend(list(body))
                else:
                    out.append(inst)
            return out

        inst_stream = _iter_instructions()

        def flush_step_noise(*, apply_noise: bool) -> None:
            nonlocal used_qubits_this_step, measured_or_reset_qubits_this_step, has_measure_or_reset_this_step

            if apply_noise:
                if has_measure_or_reset_this_step:
                    resonator_idle_targets = sorted(all_qubits - measured_or_reset_qubits_this_step)
                    if resonator_idle_targets and self.noise.reasonable_idle_dpl > 0:
                        for q in resonator_idle_targets:
                            p = self._sample_probability(rng=rng, mean=self.noise.reasonable_idle_dpl)
                            if p > 0:
                                noisy_circuit.append("DEPOLARIZE1", [q], [p])
                else:
                    idle_targets = sorted(all_qubits - used_qubits_this_step)
                    if idle_targets and self.noise.idle_dpl > 0:
                        for q in idle_targets:
                            p = self._sample_probability(rng=rng, mean=self.noise.idle_dpl)
                            if p > 0:
                                noisy_circuit.append("DEPOLARIZE1", [q], [p])

            used_qubits_this_step = set()
            measured_or_reset_qubits_this_step = set()
            has_measure_or_reset_this_step = False

        for inst in inst_stream:
            if isinstance(inst, stim.CircuitRepeatBlock):
                # This should not occur when unrolling. Keep a safe fallback.
                flush_step_noise(apply_noise=not self._step_is_perfect(step_index))
                nested = CircuitNoiseInserter(
                    self.noise,
                    data_qubits=self.data_qubits,
                    perfect_logical_initialization=False,
                    perfect_logical_measurement=self.perfect_logical_measurement,
                    perfect_first_syndrome_extraction=False,
                    syndrome_ticks_per_round=self.syndrome_ticks_per_round,
                )
                nested_noisy = nested.insert_realistic_noise(
                    inst.body_copy(),
                    rng=rng,
                    unroll_repeat_blocks=unroll_repeat_blocks,
                )
                noisy_circuit.append(stim.CircuitRepeatBlock(inst.repeat_count, nested_noisy))
                continue

            qubits = self._instruction_qubits(inst)
            name = inst.name
            step_is_perfect = self._step_is_perfect(step_index)

            if name == "TICK":
                flush_step_noise(apply_noise=not step_is_perfect)
                noisy_circuit.append(inst)
                step_index += 1
                continue

            if qubits:
                used_qubits_this_step.update(qubits)

            if name in {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}:
                if not step_is_perfect and self.noise.measurement_flip > 0 and qubits:
                    skip: set[int] = set()
                    if self.perfect_logical_measurement and name in {"M", "MZ", "MX", "MY"}:
                        skip = self.data_qubits

                    err_gate = self._measurement_error_gate(name)
                    if err_gate is not None:
                        for q in qubits:
                            if q in skip:
                                continue
                            p = self._sample_probability(rng=rng, mean=self.noise.measurement_flip)
                            if p > 0:
                                noisy_circuit.append(err_gate, [q], [p])

                noisy_circuit.append(inst)
                measured_or_reset_qubits_this_step.update(qubits)
                has_measure_or_reset_this_step = True

                if not step_is_perfect and self.noise.reset_flip > 0 and qubits:
                    skip_reset: set[int] = set()
                    if self.perfect_logical_initialization and step_index == 0:
                        skip_reset = self.data_qubits

                    err_gate = self._reset_error_gate(name)
                    if err_gate is not None:
                        for q in qubits:
                            if q in skip_reset:
                                continue
                            p = self._sample_probability(rng=rng, mean=self.noise.reset_flip)
                            if p > 0:
                                noisy_circuit.append(err_gate, [q], [p])
                continue

            noisy_circuit.append(inst)

            if name in {"R", "RZ", "RX", "RY"}:
                measured_or_reset_qubits_this_step.update(qubits)
                has_measure_or_reset_this_step = True
                if not step_is_perfect and self.noise.reset_flip > 0 and qubits:
                    skip_reset: set[int] = set()
                    if self.perfect_logical_initialization and step_index == 0:
                        skip_reset = self.data_qubits

                    err_gate = self._reset_error_gate(name)
                    if err_gate is not None:
                        for q in qubits:
                            if q in skip_reset:
                                continue
                            p = self._sample_probability(rng=rng, mean=self.noise.reset_flip)
                            if p > 0:
                                noisy_circuit.append(err_gate, [q], [p])
                continue

            if name in {
                "H",
                "H_XY",
                "H_YZ",
                "S",
                "S_DAG",
                "SQRT_X",
                "SQRT_X_DAG",
                "SQRT_Y",
                "SQRT_Y_DAG",
                "C_XYZ",
                "C_ZYX",
                "X",
                "Y",
                "Z",
            }:
                if not step_is_perfect and self.noise.one_q_gate_dpl > 0 and qubits:
                    for q in qubits:
                        p = self._sample_probability(rng=rng, mean=self.noise.one_q_gate_dpl)
                        if p > 0:
                            noisy_circuit.append("DEPOLARIZE1", [q], [p])
                continue

            if name in {"CNOT", "CX", "CZ", "CY", "ISWAP", "ISWAP_DAG", "SWAP", "CXSWAP", "CZSWAP", "SWAPCX"}:
                if not step_is_perfect and self.noise.two_q_gate_dpl > 0 and qubits:
                    if len(qubits) % 2 != 0:
                        raise ValueError(f"2q gate '{name}' has odd number of qubit targets: {qubits}")
                    for i in range(0, len(qubits), 2):
                        a, b = qubits[i], qubits[i + 1]
                        p = self._sample_probability(rng=rng, mean=self.noise.two_q_gate_dpl)
                        if p > 0:
                            noisy_circuit.append("DEPOLARIZE2", [a, b], [p])
                continue

        flush_step_noise(apply_noise=not self._step_is_perfect(step_index))
        return noisy_circuit


def insert_arbitrary_noise(
    circuit: stim.Circuit,
    noise: NoiseParameters,
    *,
    data_qubits: set[int] | None = None,
    perfect_logical_initialization: bool = False,
    perfect_logical_measurement: bool = False,
    perfect_first_syndrome_extraction: bool = False,
    syndrome_ticks_per_round: int = 8,
) -> stim.Circuit:
    inserter = CircuitNoiseInserter(
        noise,
        data_qubits=data_qubits,
        perfect_logical_initialization=perfect_logical_initialization,
        perfect_logical_measurement=perfect_logical_measurement,
        perfect_first_syndrome_extraction=perfect_first_syndrome_extraction,
        syndrome_ticks_per_round=syndrome_ticks_per_round,
    )
    return inserter.insert_arbitrary_noise(circuit)


def insert_realistic_noise(
    circuit: stim.Circuit,
    noise: NoiseParameters,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    unroll_repeat_blocks: bool = True,
    data_qubits: set[int] | None = None,
    perfect_logical_initialization: bool = False,
    perfect_logical_measurement: bool = False,
    perfect_first_syndrome_extraction: bool = False,
    syndrome_ticks_per_round: int = 8,
) -> stim.Circuit:
    """Top-level wrapper for the realistic per-target noise inserter."""
    inserter = CircuitNoiseInserter(
        noise,
        data_qubits=data_qubits,
        perfect_logical_initialization=perfect_logical_initialization,
        perfect_logical_measurement=perfect_logical_measurement,
        perfect_first_syndrome_extraction=perfect_first_syndrome_extraction,
        syndrome_ticks_per_round=syndrome_ticks_per_round,
    )
    return inserter.insert_realistic_noise(
        circuit,
        seed=seed,
        rng=rng,
        unroll_repeat_blocks=unroll_repeat_blocks,
    )
