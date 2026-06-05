from __future__ import annotations

import concurrent.futures
import os
from typing import Union

import numpy as np
import stim

from quits import CircuitBuildOptions, ErrorModel, HgpCode
from quits.gf2_util import compute_code_distance

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
