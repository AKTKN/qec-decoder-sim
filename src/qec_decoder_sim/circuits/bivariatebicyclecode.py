from __future__ import annotations

from typing import Literal

import stim

from qec_decoder_sim.circuits._bbcode_circuit_builder import (
    BBCircuitBuildResult,
    build_bb_ibm_noisy_stim_circuit,
    build_bb_ideal_stim_circuit,
    build_bb_phenom_stim_circuit,
)
from qec_decoder_sim.codes.code_factory import BBCodeSpec
from qec_decoder_sim.noise.noise_model import (
	NoiseParameters,
	insert_arbitrary_noise,
	si1000_circuit_level,
	uniform_circuit_level,
)

NoiseModelName = Literal["code_capacity", "uniform-cln", "si1000", "phenomenological", "custom"]
Basis = Literal["x", "z"]
NumRounds = int | Literal["d"]


class BivariateBicycleCode:
	"""User-facing circuit builder for bivariate bicycle codes.

	Notes
	-----
	- This class is intended to be called directly by users.
	- The underlying circuit schedule generation is delegated to
	  `build_bb_ideal_stim_circuit` (ideal/noiseless circuit only).
	- Noise insertion is handled in this class (not in the builder module).
	"""

	def __init__(
		self,
		*,
		code_parameter: tuple[int, int, int] | None = None,
		bbcode_spec: BBCodeSpec | None = None,
		noise_model: NoiseModelName = "code_capacity",
		p: float | None = None,
		noise_parameter: NoiseParameters | None = None,
		num_rounds: NumRounds = 1,
		basis: Basis = "z",
		hzh: bool | None = None,
		use_both: bool = False,
		ibm_reproduce: bool = False,
	) -> None:
		self.code_parameter = code_parameter
		self._bbcode_spec = bbcode_spec
		self.noise_model = noise_model
		self.p = p
		self.noise_parameter = noise_parameter
		self.num_rounds = self._resolve_num_rounds(num_rounds=num_rounds, code_parameter=code_parameter)
		self.basis: Basis = basis.lower()  # type: ignore[assignment]
		if self.basis not in {"x", "z"}:
			raise ValueError("basis must be either 'x' or 'z'.")

		self.use_both = use_both
		self.ibm_reproduce = ibm_reproduce
		self.hzh = (noise_model == "si1000") if hzh is None else hzh

		self._validate_code_args(code_parameter=code_parameter, bbcode_spec=bbcode_spec)
		self._validate_noise_args()

		# Build once at initialization to avoid recomputation in get_circuit().
		self._circuit_build_result = self.build()
		self._circuit = self._circuit_build_result.circuit

	def get_noisemodel(self) -> NoiseModelName:
		"""Return the configured noise model name."""
		return self.noise_model

	def get_circuit(self) -> stim.Circuit:
		"""Return the pre-built stim circuit for this instance."""
		return self._circuit

	@staticmethod
	def _validate_code_args(*, code_parameter: tuple[int, int, int] | None, bbcode_spec: BBCodeSpec | None) -> None:
		if code_parameter is not None and bbcode_spec is not None:
			raise ValueError("Specify either code_parameter or bbcode_spec, not both.")
		if code_parameter is None and bbcode_spec is None:
			raise ValueError("Either code_parameter or bbcode_spec must be specified.")

	def _get_bbcode_spec(self) -> BBCodeSpec:
		if self._bbcode_spec is not None:
			return self._bbcode_spec
		if self.code_parameter is None:
			raise ValueError("bbcode_spec is missing and code_parameter is not set.")
		self._bbcode_spec = self._bbcode_spec_from_code_parameter(self.code_parameter)
		return self._bbcode_spec

	@staticmethod
	def _bbcode_spec_from_code_parameter(code_parameter: tuple[int, int, int]) -> BBCodeSpec:
		"""Derive a `BBCodeSpec` from a target (n, k, d).

		This is a placeholder hook. Implementations may search a catalog of known
		BBCodeSpec entries or compute the next available spec matching the
		requested code parameters.
		"""
		n, k, d = code_parameter
		raise NotImplementedError(
			"Automatic BBCodeSpec generation from (n,k,d) is not implemented yet. "
			"For now, pass bbcode_spec explicitly. "
			f"Got code_parameter=(n={n}, k={k}, d={d})."
		)

	@staticmethod
	def _resolve_num_rounds(*, num_rounds: NumRounds, code_parameter: tuple[int, int, int] | None) -> int:
		if isinstance(num_rounds, int):
			if num_rounds < 1:
				raise ValueError("num_rounds must be >= 1")
			return num_rounds
		if num_rounds != "d":
			raise ValueError("num_rounds must be an int or 'd'.")
		if code_parameter is None:
			raise ValueError("num_rounds='d' requires code_parameter to be specified.")
		d = int(code_parameter[2])
		if d < 1:
			raise ValueError("distance d must be >= 1 when using num_rounds='d'.")
		return d

	def _validate_noise_args(self) -> None:
		if self.noise_model == "custom":
			if self.noise_parameter is None:
				raise ValueError("noise_model='custom' requires noise_parameter (NoiseParameters).")
			return

		if self.p is None:
			raise ValueError("Parameter p must be specified unless noise_model='custom'.")
		if self.p < 0:
			raise ValueError("p must be >= 0")

	def build_ideal(self) -> BBCircuitBuildResult:
		"""Build an ideal (noiseless) stim circuit."""
		spec = self._get_bbcode_spec()
		return build_bb_ideal_stim_circuit(
			spec,
			num_rounds=self.num_rounds,
			basis=self.basis,
			use_both=self.use_both,
			hzh=self.hzh,
		)

	def build(self) -> BBCircuitBuildResult:
		"""Build a (potentially) noisy stim circuit based on `noise_model`."""
		if self.ibm_reproduce:
			if self.noise_model not in {"uniform-cln"}:
				raise ValueError(
					"ibm_reproduce=True requires noise_model='uniform' (uniform-cln); "
					f"got noise_model={self.noise_model!r}"
				)
			if self.p is None:
				raise ValueError("ibm_reproduce=True requires p.")
			spec = self._get_bbcode_spec()
			return build_bb_ibm_noisy_stim_circuit(
				spec,
				num_rounds=self.num_rounds,
				basis=self.basis,
				use_both=self.use_both,
				hzh=self.hzh,
				p=self.p,
			)
		if self.noise_model == "phenomenological":
			if self.p is None:
				raise ValueError("noise_model='phenomenological' requires p.")
			spec = self._get_bbcode_spec()
			return build_bb_phenom_stim_circuit(
				spec,
				num_rounds=self.num_rounds,
				basis=self.basis,
				use_both=self.use_both,
				hzh=self.hzh,
				p=self.p,
			)
		ideal = self.build_ideal()
		noisy = self._apply_noise_model(ideal)
		return BBCircuitBuildResult(circuit=noisy, metadata=ideal.metadata)

	def _apply_noise_model(self, ideal: BBCircuitBuildResult) -> stim.Circuit:
		model = self.noise_model

		if model == "code_capacity":
			if self.p is None:
				raise ValueError("noise_model='code_capacity' requires p.")
			return self._apply_code_capacity_noise(ideal.circuit, p=self.p, data_qubits=ideal.metadata.data_qubits)

		if model == "uniform-cln":
			if self.p is None:
				raise ValueError("noise_model='uniform-cln' requires p.")
			noise = uniform_circuit_level(self.p)
			return insert_arbitrary_noise(ideal.circuit, noise, data_qubits=set(ideal.metadata.data_qubits))

		if model == "si1000":
			if self.p is None:
				raise ValueError("noise_model='si1000' requires p.")
			noise = si1000_circuit_level(self.p)
			return insert_arbitrary_noise(ideal.circuit, noise, data_qubits=set(ideal.metadata.data_qubits))

		if model == "custom":
			assert self.noise_parameter is not None
			return insert_arbitrary_noise(
				ideal.circuit,
				self.noise_parameter,
				data_qubits=set(ideal.metadata.data_qubits),
			)

		raise ValueError(f"Unknown noise_model: {model}")

	@staticmethod
	def _apply_code_capacity_noise(
		circuit: stim.Circuit,
		*,
		p: float,
		data_qubits: tuple[int, ...],
	) -> stim.Circuit:
		"""Apply a simple code-capacity noise: one-time depolarization on data qubits.

		Implementation detail:
		We inject `DEPOLARIZE1` immediately after the first top-level `TICK`.
		This matches the intended placement "after initialization, before the
		syndrome extraction schedule".
		"""
		if p <= 0:
			return circuit

		out = stim.Circuit()
		inserted = False
		targets = list(data_qubits)

		for inst in circuit:
			out.append(inst)
			if not inserted and not isinstance(inst, stim.CircuitRepeatBlock) and inst.name == "TICK":
				out.append("DEPOLARIZE1", targets, [p])
				inserted = True

		if not inserted:
			out.append("DEPOLARIZE1", targets, [p])

		return out
