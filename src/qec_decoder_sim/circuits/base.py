# from abc import ABC, abstractmethod
# from typing import Tuple, Dict, Any
# import stim
# import numpy as np


# class BaseCircuit(ABC):
#     """
#     Abstract base class for quantum circuits used in error correction decoding simulations.
#     """

#     @property
#     @abstractmethod
#     def num_qubits(self) -> int:
#         """Return the number of qubits in the circuit."""
#         pass

#     @abstructmethod
#     def generate_ideal_circuit(self) -> stim.Circuit:
#         """Generate the ideal quantum circuit without noise."""
#         pass

#     @abstractmethod
#     def get_parity_check_matrix(self) -> np.ndarray:
#         """Return the parity check matrix of the code."""
#         pass
    